import argparse

import pandas as pd
import numpy as np
import openap

from pitot import isa

import utils
from polynomial import Polynomial
from correct_date import joincdates




class WrapperOpenAP:
    """
    A wrapper to OpenAP that takes a dataframe as input, this allows to deal with SI/aeronautical units stuff
    We also added a method to extract the drag as a second degree polynomial of the mass.
    """
    _dragsynonym = {
        "a21n":"a321",
        "crj9":"e75l",
        "bcs3":"a319",
        "at76":"e75l",
        "b763":"b752",
        "bcs1":"a319",
        "b39m":"b739",
        "a310":"a319",
        "a318":"a319",
        "b773":"b772",
        "c56x":"c550",
        "e290":"e190"
    }
    _thrustsynonym = _dragsynonym
    def __init__(self,ac):
        ac = ac.lower()
        self.thrust_model = openap.Thrust(self._thrustsynonym.get(ac,ac))
        self.drag_model = openap.Drag(self._dragsynonym.get(ac,ac))
        self.aircraft = openap.prop.aircraft(self._thrustsynonym.get(ac,ac))
    def convert(self,df):
        tas_kt = df.tas / utils.KTS2MS
        alt_ft = df.altitude / utils.FEET2METER
        rocd = df.vertical_rate / utils.FEET2METER * 60
        return tas_kt, alt_ft,rocd
    def drag_polynomial(self,df):
        tas,alt,_ = self.convert(df);del df
        # drag = c_0 + c_2 * m ** 2
        masses = np.array([0,10000])
        assert masses[0] == 0
        # drags: masses x points
        drags = self.drag_model.clean(masses[:,np.newaxis],tas,alt)
        # c_2 = (drag - c_0) / m**2
        drags[1,:] = (drags[1,:]-drags[0,:]) / masses[1] ** 2
        return drags.T # points x coeffs
    def thrust_climb(self, df, use_rocd=False):
        tas,alt,rocd = self.convert(df);del df
        if not use_rocd:
            rocd = 0
        return self.thrust_model.climb(tas=tas, alt=alt, roc=rocd)
    def thrust_descent(self, df):
        tas,alt,_ = self.convert(df);del df
        return self.thrust_model.descent_idle(tas=tas, alt=alt)


def energy_rate(df,periods,thresh_dt):
    '''
    Computes energy rate from trajs fils @df considering pairs of points separated by @period points.
    Roughly speaking it computes (Energy(point i) - Energy(point i+@period))/(Timestamp(point i) - Timestamp(point i+@period)).
    If (Timestamp(point i+@period) - Timestamp(point i))> @thresh_dt, then the energy rate is ruled out as np.nan.
    This allows to treat all the trajectories altogether without considering one trajectory at a time.
    '''
    tempISA = isa.temperature(df.altitude)
    tau = df.temperature / tempISA
    g_0 = 9.80665
    isdifferent = df.flight_id.shift(periods=periods) != df.flight_id
    dalt = df.altitude - df.altitude.shift(periods=periods)
    tas2 = df.tas ** 2
    dtas2 = tas2 - tas2.shift(periods=periods)
    dt = (df.timestamp - df.timestamp.shift(periods=periods)).dt.total_seconds()
    dwx = df.u_component_of_wind - df.u_component_of_wind.shift(periods=periods)
    dwy = df.v_component_of_wind - df.v_component_of_wind.shift(periods=periods)
    dewind = dwx * df.tasx + dwy * df.tasy
    e = (g_0 * dalt * tau + 0.5 * dtas2 + dewind) / dt
    e[isdifferent]=np.nan
    e[dt.abs() > thresh_dt]=np.nan
    return np.clip(e.values,-400,300)


def total_energy_polynomial_equation(thrust,drag_poly,energy_rate,tas):
    '''
    Computes the polynomial associated to the total energy rate equation:
    Polynom(mass)=(Thrust-Drag)/m*tas - energy_rate
    Physicals equation rules that Polynom(actual_mass)=0
    So the roots are of particular interest
    '''
    pthrust = Polynomial(thrust[:,np.newaxis])

    penergyshape = energy_rate.shape+(2,)
    cenergy = np.zeros(penergyshape,dtype=np.float64)
    cenergy[...,1] = energy_rate
    penergy = Polynomial(cenergy)

    dshape = drag_poly.shape[:-1]+(3,)
    cdrag = np.zeros(dshape,dtype=np.float64)
    cdrag[...,0] = drag_poly[...,0]
    cdrag[...,2] = drag_poly[...,1]
    pdrag = Polynomial(cdrag)
    ptas = Polynomial(tas[:,np.newaxis])
    power = (pthrust - pdrag) * ptas
    return power - penergy

def compute_mass(df, is_climb,periods,thresh_dt,cthrust):
    '''
    Computes the mass as the largest root of the total energy polynom
    '''
    lac = df.aircraft_type.unique()
    n = df.shape[0]
    thrust = np.empty(n)
    thrust[:]=np.nan
    drag_poly = np.empty((n,2))
    drag_poly[:]=np.nan
    for ac in lac:
        if pd.isna(ac):
            print("nan detected")
            raise Exception
        mask = (df.aircraft_type == ac).to_numpy(dtype=bool,na_value=False)
        dfac = df.query("aircraft_type == @ac")
        apmodel = WrapperOpenAP(ac)
        thrust_ac =  cthrust * apmodel.thrust_climb(dfac)
        drag_poly_ac = apmodel.drag_polynomial(dfac)
        thrust[mask] = thrust_ac
        drag_poly[mask] = drag_poly_ac
    e_rate = energy_rate(df,periods,thresh_dt)
    total_energy_poly = total_energy_polynomial_equation(thrust,drag_poly,e_rate,df.tas.values)
    sols =total_energy_poly.roots2()
    which = 0 # if is_climb else 1
    return pd.DataFrame({"masses":sols[:,which],"e_rate":e_rate},index=df.index)


def feature_climbing(trajs, flights,threshold_vr,periods,thresh_dt,is_climb,cthrust,vrate_var,altstart,altstep):
    ''' compute all the features to be added by this python code '''
    # flights["mid_flight_time"] = flights.actual_offblock_time + (flights.arrival_time-flights.actual_offblock_time) / 2
    flights["mid_flight_time"] = flights.t_adep + (flights.t_ades-flights.t_adep) / 2
    print(trajs.flight_id.nunique())
    dfjoinedin = trajs.join(flights.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")
    print(dfjoinedin.flight_id.nunique())
    dfjoinedin = dfjoinedin.merge(compute_mass(dfjoinedin,is_climb=is_climb,periods=periods,thresh_dt=thresh_dt,cthrust=cthrust),left_index=True,right_index=True)
    timecondition = "t_adep<=timestamp<mid_flight_time" if is_climb else "t_ades>timestamp>mid_flight_time"
    lvrcondition = [f"{vrate_var} > @threshold_vr",f"-@threshold_vr<= {vrate_var} <= @threshold_vr"] if is_climb else [f"{vrate_var} < @threshold_vr"]
    lalts = [np.arange(altstart,48,altstep) * 1000 * utils.FEET2METER,[-10000,2000* utils.FEET2METER]]  if is_climb else [np.arange(altstart,48,altstep) * 1000 * utils.FEET2METER]
    lprefix=["climb","takeoff"] if is_climb else ["descent"]
    dfjoinedin = dfjoinedin.query(f"{timecondition}")# and {vrcondition}")
    print(dfjoinedin.flight_id.nunique())
    results = dfjoinedin[["flight_id","aircraft_type"]].drop_duplicates().reset_index(drop=True)
    n = results.shape[0]
    mass_max = np.empty((n,))
    mass_max[:]=np.nan
    mass_min = np.empty((n,))
    mass_min[:]=np.nan
    for ac in results.aircraft_type.unique():
        if pd.isna(ac):
            print("nan detected")
            raise Exception
        mask = (results.aircraft_type == ac).to_numpy(dtype=bool,na_value=False)
        apmodel = WrapperOpenAP(ac)
        mass_max[mask] = apmodel.aircraft["limits"]["MTOW"]
        mass_min[mask] = apmodel.aircraft["limits"]["OEW"]
    results["mass_max"]=mass_max
    results["mass_min"]=mass_min
    results = results.drop(columns="aircraft_type").set_index("flight_id")
    lresults = []
    dfjoinedin = dfjoinedin.assign(DeltaT=dfjoinedin.temperature - isa.temperature(dfjoinedin.altitude))
    # iter different context climb/take_off or just descent
    for (alts,vrcondition,prefix) in zip(lalts,lvrcondition,lprefix):
        dfjoined = dfjoinedin.query(vrcondition)
        # compute stats on slices
        for (i,(hlow,hhigh)) in enumerate(zip(alts[:-1],alts[1:])):
            grouped = dfjoined.query("@hlow <= altitude < @hhigh").groupby("flight_id")
            lresults.append(grouped.masses.median().rename(f"{prefix}mass_{i}"))
            lresults.append(grouped.masses.count().fillna(0).rename(f"{prefix}masscount_{i}"))
            lresults.append((grouped.timestamp.min() - grouped.t_adep.min()).dt.total_seconds().rename(f"{prefix}massadepdate_{i}"))
            lresults.append((grouped.t_ades.max() - grouped.timestamp.max()).dt.total_seconds().rename(f"{prefix}massadesdate_{i}"))
            lresults.append((grouped.vertical_rate.max()-grouped.vertical_rate.min()).rename(f"{prefix}DeltaROCD_{i}"))
            lresults.append(grouped.vertical_rate.median().rename(f"{prefix}ROCD_{i}"))
            lresults.append(grouped.e_rate.median().rename(f"{prefix}e_rate_median_{i}"))
            lresults.append(grouped.tas.median().rename(f"{prefix}tas_median_{i}"))
            lresults.append(grouped.DeltaT.median().rename(f"{prefix}DeltaT_median_{i}"))
            lresults.append(grouped.e_rate.max().rename(f"{prefix}e_rate_max_{i}"))
            lresults.append(grouped.e_rate.min().rename(f"{prefix}e_rate_min_{i}"))
    results = pd.concat([results]+lresults,axis=1)
    print("results.shape",results.shape)
    return results


def main():
    import readers
    parser = argparse.ArgumentParser(
                    description='compute climbing/mass features using altitude-wise slices',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-f_out")
    parser.add_argument("-airports")
    parser.add_argument("-is_climb",action="store_true")
    parser.add_argument("-threshold_vr",type=float,help="in [ft/min]")
    parser.add_argument("-cthrust",type=float,help="thrust = cthrust * thrustmaxclimb")
    parser.add_argument("-periods",type=int)
    parser.add_argument("-thresh_dt",type=int,help="in [s]")
    parser.add_argument("-vrate_var",type=str)
    parser.add_argument("-altstep",type=float,help="in [FL]")
    parser.add_argument("-altstart",type=float,help="in [FL]")
    args = parser.parse_args()
    args.threshold_vr = args.threshold_vr * utils.FEET2METER / 60
    trajs = readers.read_trajectories(args.t_in).reset_index()
    airports=pd.read_parquet(args.airports)

    trajs["latitude"]=trajs["latitude"] * 180 / np.pi
    trajs["longitude"]=trajs["longitude"] * 180 / np.pi
    trajs["altitude"]=trajs["altitude"] / utils.FEET2METER
    flights = joincdates(readers.read_flights(args.f_in),airports,trajs,rod=-1000,roc=1000)
    trajs["latitude"]=trajs["latitude"] / 180 * np.pi
    trajs["longitude"]=trajs["longitude"] / 180 * np.pi
    trajs["altitude"]=trajs["altitude"] * utils.FEET2METER

    dfadded = feature_climbing(trajs, flights,
                       periods = args.periods,
                       thresh_dt = args.thresh_dt,
                       threshold_vr = args.threshold_vr,
                       is_climb = args.is_climb,
                       cthrust = args.cthrust,
                       vrate_var=args.vrate_var,
                       altstart = args.altstart,
                       altstep = args.altstep,
    )
    return dfadded.reset_index().to_parquet(args.f_out,index=False)


if __name__ == '__main__':
    main()
