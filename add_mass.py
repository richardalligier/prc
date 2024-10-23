import pandas as pd
import numpy as np
import openap
import utils
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from polynomial import Polynomial
import argparse
# import pitot
from pitot import isa
import csaps
from add_cruise_infos import joincdates




class WrapperOpenAP:
#    _thrustsynonym = {"crj9":"e75l","bcs3":"a20n","at76":"e145","bcs1":"a20n","a310":"a318","c56x":"c550","e290":"e190"}
    _dragsynonym = {"a21n":"a321","crj9":"e75l","bcs3":"a319","at76":"e75l","b763":"b752","bcs1":"a319","b39m":"b739","a310":"a319","a318":"a319","b773":"b772","c56x":"c550","e290":"e190"}
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
        c_0 = self.drag_model.clean(0,tas,alt)
        masses = np.array([0,10000])
        assert masses[0] == 0
        # drags: masses x points
        drags = self.drag_model.clean(masses[:,np.newaxis],tas,alt)
        # c_2 = (drag - c_0) / m**2
        drags[1,:] = (drags[1,:]-drags[0,:]) / masses[1] ** 2
        # return np.array([c_0, c_2]).T
        return drags.T # points x coeffs
    def thrust_climb(self, df, use_rocd=False):
        tas,alt,rocd = self.convert(df);del df
        if not use_rocd:
            rocd = 0
        # a = self.thrust_model.climb(tas=tas, alt=alt, roc=rocd)
        # b = self.thrust_model.climb(tas=tas, alt=alt, roc=0)
        # plt.plot(a)
        # plt.plot(b)
        # plt.show()
        # raise Exception
        return self.thrust_model.climb(tas=tas, alt=alt, roc=rocd)
    def thrust_descent(self, df):
        tas,alt,_ = self.convert(df);del df
        return self.thrust_model.descent_idle(tas=tas, alt=alt)

# def temperature_ISA(alt):
#     # t_0 = 288.15
#     # beta_T_inf = -0.0065
#     # alt_Hp_trop = 11000.
#     # res = t_0 + beta_T_inf * np.minimum(alt,alt_Hp_trop)
#     # res2 = isa.temperature(alt)
#     # print(np.abs(res-res2).max())
#     # assert (np.abs(res-res2).max()<2)
#     return isa.temperature(alt)#res

def energy_rate(df,periods,thresh_dt):
    # df = df.reset_index()
    tempISA = isa.temperature(df.altitude)
    tau = df.temperature / tempISA
    g_0 = 9.80665
    isdifferent = df.flight_id.shift(periods=periods) != df.flight_id
    dalt = df.altitude - df.altitude.shift(periods=periods)
    tas2 = df.tas ** 2
    dtas2 = tas2 - tas2.shift(periods=periods)
    dt = (df.timestamp - df.timestamp.shift(periods=periods)).dt.total_seconds()
    # print(df[["flight_id","timestamp"]].head())
    # plt.plot(dt)
    # plt.show()
    # print(dt.min(),dt.max())
    # assert(dt.min()>=0)
    dwx = df.u_component_of_wind - df.u_component_of_wind.shift(periods=periods)
    dwy = df.v_component_of_wind - df.v_component_of_wind.shift(periods=periods)
    dewind = dwx * df.tasx + dwy * df.tasy
    e = (g_0 * dalt * tau + 0.5 * dtas2 + dewind) / dt
    e[isdifferent]=np.nan
    e[dt.abs() > thresh_dt]=np.nan
    return np.clip(e.values,-400,300)

# def energy_rate(df,periods=5,thresh_dt=40):
#     for fid in df.flight_id.unique():
#         traj = df.query("flight_id==@fid").reset_index()
#         t = (traj.timestamp-traj.timestamp.iloc[0]).dt.seconds
#         print(t)
#         raise Exception
#     # temp = csaps.CubicSmoothingSpline(xdata=trajff.timeAtServer.values,ydata=trajff.nnpredlatitude.values,smooth=smooth).spline
#     # del df
#     tempISA = temperature_ISA(df.altitude)
#     tau = df.temperature / tempISA
#     g_0 = 9.80665
#     isdifferent = df.flight_id.shift(periods=periods) != df.flight_id
#     dalt = df.altitude - df.altitude.shift(periods=periods)
#     tas2 = df.tas ** 2
#     dtas2 = tas2 - tas2.shift(periods=periods)
#     dt = (df.timestamp - df.timestamp.shift(periods=periods)).dt.seconds
#     dwx = df.u_component_of_wind - df.u_component_of_wind.shift(periods=periods)
#     dwy = df.v_component_of_wind - df.v_component_of_wind.shift(periods=periods)
#     dewind = dwx * df.tasx + dwy * df.tasy
#     print(dt.min())
#     e = (g_0 * dalt * tau + 0.5 * dtas2 + dewind) / dt
#     e[isdifferent]=np.nan
#     e[dt > thresh_dt]=np.nan
#     return np.clip(e.values,-200,300)

# def average(df,power,periods):
#     isdifferent = df.flight_id.shift(periods=periods) != df.flight_id
#     averagepower = np.convolve(power.data,np.ones(periods)/periods,mode="valid")
#     return Polynomial(averagepower)

def total_energy_polynomial_equation(thrust,drag_poly,energy_rate,tas):
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
    # print(pthrust.shape())
    # print(pdrag.shape())
    # print(ptas.shape())
    # print(penergy.shape())
    # raise Exception
    return power - penergy

def compute_mass(df, is_climb,periods,thresh_dt,cthrust):
#    df = df.reset_index()#.set_index("flight_id")
    lac = df.aircraft_type.unique()
    n = df.shape[0]
    thrust = np.empty(n)
    thrust[:]=np.nan
    drag_poly = np.empty((n,2))
    drag_poly[:]=np.nan
    # mass_max = np.empty((n,))
    # mass_max[:]=np.nan
    # mass_min = np.empty((n,))
    # mass_min[:]=np.nan
    for ac in lac:
        # print(ac,pd.isna(ac))
        if pd.isna(ac):
            print("nan detected")
            raise Exception
        mask = (df.aircraft_type == ac).to_numpy(dtype=bool,na_value=False)
        dfac = df.query("aircraft_type == @ac")
        apmodel = WrapperOpenAP(ac)
        # mass_max[mask] = apmodel.aircraft["limits"]["MTOW"] #+ apmodel.aircraft["limits"]["OEW"]) * 0.5
        # mass_min[mask] = apmodel.aircraft["limits"]["OEW"] #+ apmodel.aircraft["limits"]["OEW"]) * 0.5
        thrust_ac =  cthrust * apmodel.thrust_climb(dfac)
        drag_poly_ac = apmodel.drag_polynomial(dfac)
        thrust[mask] = thrust_ac
        drag_poly[mask] = drag_poly_ac
    e_rate = energy_rate(df,periods,thresh_dt)
    total_energy_poly = total_energy_polynomial_equation(thrust,drag_poly,e_rate,df.tas.values)
    # return pd.DataFrame({"masses":e_rate},index=df.index)#total_energy_poly.roots2()[:,0] #
    sols =total_energy_poly.roots2()
    # selected = np.zeros(sols.shape[0],dtype=np.int64)
    # selected[e_rate > 0] = 1
    # # plt.plot(np.take_along_axis(sols,selected[:,None],axis=1)[:,0])
    #[:,0],sols[:,1])

    # print(flights.iloc[0][["actual_offblock_time","mid_flight_time","arrival_time"]])
    # raise Exception
    dshape = drag_poly.shape[:-1]+(3,)
    cdrag = np.zeros(dshape,dtype=np.float64)
    cdrag[...,0] = drag_poly[...,0]
    cdrag[...,2] = drag_poly[...,1]
    pdrag = Polynomial(cdrag)
    # c = (e_rate / df.tas.values + pdrag.eval(expected_mass))/ thrust
    # p = (thrust- pdrag.eval(expected_mass))*df.tas.values/expected_mass
#    cthrust *
    # print(c.shape)
    # mask = np.logical_and(df.timestamp.values > df.mid_flight_time.values,df.vertical_rate<-1500*utils.FEET2METER/60)
    # # plt.plot(expected_mass)
    # plt.plot(c)#[mask])
    # # plt.plot(p[mask])#[mask])
    # plt.show()
    # # plt.plot(sols[:,0])
    # # plt.plot(sols[:,1])
    # # plt.plot(sols[:,1]-sols[:,0])
    # # plt.plot(e_rate*1000)
    # # plt.plot(np.take_along_axis(sols,1-selected[:,None],axis=1)[:,0])#[:,0],sols[:,1])
    # # plt.plot(np.min(sols,axis=1))#[:,0],sols[:,1])
    # # plt.plot(np.max(sols,axis=1))#[:,0],sols[:,1])
    # # plt.show()
    # raise Exception
    which = 0 if is_climb else 1
    return pd.DataFrame({"masses":sols[:,which],"e_rate":e_rate},index=df.index)#total_energy_poly.roots2()[:,0] #


def add_mass(trajs, flights,threshold_vr,periods,thresh_dt,is_climb,prefix,cthrust,vrate_var,altstart,altstep):

    # flights["mid_flight_time"] = flights.actual_offblock_time + (flights.arrival_time-flights.actual_offblock_time) / 2
    flights["mid_flight_time"] = flights.t_adep + (flights.t_ades-flights.t_adep) / 2
    # print(flights.iloc[0][["actual_offblock_time","mid_flight_time","arrival_time"]])
    # raise Exception
    print(trajs.flight_id.nunique())
    dfjoined = trajs.join(flights.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#.query("aircraft_type=='B738'")
    print(dfjoined.flight_id.nunique())
    dfjoined = dfjoined.merge(compute_mass(dfjoined,is_climb=is_climb,periods=periods,thresh_dt=thresh_dt,cthrust=cthrust),left_index=True,right_index=True)
    timecondition = "timestamp<mid_flight_time" if is_climb else "timestamp>mid_flight_time"
    vrcondition = f"{vrate_var} > @threshold_vr" if is_climb else f"{vrate_var} < @threshold_vr"
    # print(dfjoined.dtypes)
    # threshold_vr = -1500 * utils.FEET2METER / 60
    dfjoined = dfjoined.query(f"{timecondition} and {vrcondition}")
    print(dfjoined.flight_id.nunique())
    # dfjoined = dfjoined.assign(masses = compute_mass(dfjoined))
    alts = np.arange(altstart,48,altstep) * 1000 * utils.FEET2METER
#    results = pd.DataFrame({"flight_id":dfjoined.flight_id.unique()}).set_index("flight_id")
    results = dfjoined[["flight_id","aircraft_type"]].drop_duplicates().reset_index(drop=True)
    # print(list(results),results.index)
    # raise Exception
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
        dfac = results.query("aircraft_type == @ac")
        apmodel = WrapperOpenAP(ac)
        mass_max[mask] = apmodel.aircraft["limits"]["MTOW"] #+ apmodel.aircraft["limits"]["OEW"]) * 0.5
        mass_min[mask] = apmodel.aircraft["limits"]["OEW"] #+ apmodel.aircraft["limits"]["OEW"]) * 0.5
    results["mass_max"]=mass_max
    results["mass_min"]=mass_min
    results = results.drop(columns="aircraft_type").set_index("flight_id")
    #results.join(dfjoined[["flight_id","]],on="flight_id",how="left",validate="one_to_many")
    # print(results.index)
    # raise Exception
    # print(alts)
    lresults = []
    dfjoined["DeltaT"]=dfjoined.temperature - isa.temperature(dfjoined.altitude)
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
    return results#.reset_index()


def main():
    import readers
    parser = argparse.ArgumentParser(
                    prog='addmass',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-f_out")
    parser.add_argument("-airports")
    parser.add_argument("-is_climb",action="store_true")
    parser.add_argument("-threshold_vr",type=float,help="in [ft/min]")
    parser.add_argument("-cthrust",type=float,help="thrust = cthrust * thrustmaxclimb")
    parser.add_argument("-prefix",type=str,default="")
    parser.add_argument("-periods",type=int)
    parser.add_argument("-thresh_dt",type=int)
    parser.add_argument("-vrate_var",type=str)
    parser.add_argument("-altstep",type=float)
    parser.add_argument("-altstart",type=float)
    args = parser.parse_args()
    args.threshold_vr = args.threshold_vr * utils.FEET2METER / 60
    trajs = readers.read_trajectories(args.t_in).reset_index()
    airports=pd.read_parquet(args.airports)

    # print("altitude",np.isnan(trajs.altitude.values).mean())
    # print("latitude",np.isnan(trajs.latitude.values).mean())
    # print("tas",np.isnan(trajs.tas.values).mean())
    # print("temp",np.isnan(trajs.temperature.values).mean())
    # print("groundspeed",np.isnan(trajs.groundspeed.values).mean())
    flights = joincdates(readers.read_flights(args.f_in),airports,trajs,rod=-1000,roc=1000)
    dfadded = add_mass(trajs, flights,
                       periods = args.periods,
                       thresh_dt = args.thresh_dt,
                       threshold_vr = args.threshold_vr,
                       is_climb = args.is_climb,
                       cthrust = args.cthrust,
                       prefix=args.prefix,
                       vrate_var=args.vrate_var,
                       altstart = args.altstart,
                       altstep = args.altstep,
    )
    # print(list(dfadded.reset_index()))
    # print(dfadded.shape)
    # raise Exception
    return dfadded.reset_index().to_parquet(args.f_out,index=False)

def maintest():
    import readers
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from sklearn.metrics import root_mean_squared_error
    fnametraj = "2022-01-02.parquet"
    trajs = readers.read_trajectories(os.path.join("/disk2/prc/interpolated_trajectories",fnametraj))
    # print(trajs.head())
    # trajs = readers.convert_to_SI(readers.add_features_trajectories(readers.convert_from_SI(readers.read_trajectories(os.path.join("/disk2/prc/rawtrajectories",fnametraj)))))
    # trajs = readers.read_trajectories(os.path.join("/disk2/prc/filtered_trajectories",fnametraj))
    # trajs = readers.add_features_trajectories(readers.read_trajectories(os.path.join("/disk2/prc/rawtrajectories",fnametraj)).sort_values(["flight_id","timestamp"]).reset_index(drop=True))
    # trajs = readers.read_trajectories("toto.parquet")
    #trajs = readers.read_trajectories("toto_interp.parquet")
    #trajs = trajs.sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.query("flight_id==248809247")
    # trajs = pd.read_parquet(os.path.join("/disk2/prc/interpolated_trajectories",fnametraj))
    # for v in ["flight_id", "icao24"]:
    #     trajs[v] = trajs[v].astype(np.int64)
    # tra = trajs.query("flight_id==248803487")
    # # plt.scatter(tra.timestamp, tra.altitude)
    # # plt.show()
    # # plt.scatter(tra.timestamp, tra.groundspeed)
    # plt.scatter(tra.timestamp, tra.latitude)
    # plt.scatter(tra.timestamp, tra.longitude)
    # plt.show()
    # tra = trajs.query("flight_id==248809247")
    # # # trajs = readers.read_trajectories(os.path.join("/disk2/prc/rawtrajectories",fnametraj))
    # # # trajs = readers.read_trajectories(os.path.join("/disk2/prc/trajectories",fnametraj))
    # # tra = trajs.query("flight_id==248809247")
    fid = 248784182#248765562
    # fid = 248757278
    # fid = 248759896
    # fid = 248772010
#    fid = 248761757 # ok, very initial climb
    #fid = 248762722 # ok
#    fid = 248752191 # not many data
    #fid = 248762056 # climb in cruise
    #fid = 248765562
    # fid = 248759396
    # # fid = 248803487
    # # fid = 248809247 # Ok avec nouveau filtrage#248803724
    # # #

    # threshold_vr = 1000 * utils.FEET2METER / 60
    # print("threshold_vr",threshold_vr)
    # tra = trajs.query("flight_id==@fid")#.sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.query("flight_id==248809247")
    # print(tra.head())
    # e = energy_rate(tra,periods=5,thresh_dt=40)#[tra.vertical_rate.values>threshold_vr]
    # #tra = tra.query("vertical_rate>@threshold_vr")
    # print(type(e))
    # print(e.shape)
    # print(tra.shape)
    # # raise Exception
    # print(e)
    # print(tra.timestamp.values)
    # plt.scatter(tra.timestamp.values,e)
    # plt.show()
    # plt.scatter(tra.altitude.values,e)
    # plt.show()
    # # raise Exception
    # plt.scatter(tra.timestamp, tra.tas)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.altitude)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.vertical_rate /utils.FEET2METER *60)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.gsx)
    # plt.scatter(tra.timestamp, tra.gsy)
    # plt.scatter(tra.timestamp, tra.groundspeed)
    # plt.scatter(tra.timestamp, np.sqrt(tra.gsx**2+tra.gsy**2))
    # plt.show()


    # plt.scatter(tra.timestamp, tra.track)
    # plt.scatter(tra.timestamp, tra.track_unwrapped)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.u_component_of_wind)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.v_component_of_wind)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.groundspeed)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.vertical_rate)
    # plt.show()
    # plt.scatter(tra.timestamp, tra.altitude)
    # plt.show()
    # plt.scatter(tra.latitude, tra.longitude)
    # plt.show()
    config = utils.read_config()
    flights = readers.read_flights(os.path.join(config.flights,"challenge_set.parquet"))
    # flights = readers.read_flights(os.path.join(config.flights,"submission_set.parquet"))
    dfadded = add_mass(trajs, flights,
                       periods = 10,
                       thresh_dt = 40,
                       threshold_vr = -500 * utils.FEET2METER /60,
                       is_climb = False,
                       cthrust=0.,
                       prefix="")
    # raise Exception
    joined = dfadded.join(flights.set_index("flight_id"),on="flight_id",how="left",validate="one_to_one")
    print(joined.shape)
    # print(joined.query("flight_id==@fid").iloc[0])
    # for i in range(8):
    #     print(joined.query(f"mass_{i}>80000")[["flight_id"]+[f"mass_{i}" for i in range(8)]])
    # print(joined.query("flight_id == 248809247")[["ades","adep"]])
    # joined = joined.query("aircraft_type=='B738'")
    # print(joined.query("mass_1>100000"))
    # print(joined.query("mass_2>100000"))
    # print(joined.query("mass_3>100000"))
    # print(joined.query("mass_4>100000"))
    # print(joined.query("mass_5>100000"))
    # print(joined.query("mass_6>100000"))
    for ac in joined.aircraft_type.unique():
        print(ac)
        aj = joined.query("aircraft_type==@ac")
        model = LinearRegression()
        model.fit(aj[["flown_distance"]],aj["tow"])
        print(aj.shape)
        # for i in range(8):
        #     estimass = aj[f"mass_{i}"].values#np.nanmean(aj[[f"mass_{i}" for i in range(8)]].values,axis=1)
        # estimass = np.nanmean(aj[[f"mass_{i}" for i in range(8)]].values,axis=1)
        # print(aj.iloc[np.nanargmax(estimass)])
        low = aj.tow-model.predict(aj[["flown_distance"]])+model.predict(np.zeros_like(aj[["flown_distance"]].values))
        # plt.scatter(aj.tow, low)
        # plt.show()
        # plt.scatter(aj.flown_distance, low)
        # plt.show()
        # raise Exception
        # estimass = np.nanmean(aj[[f"mass_{i}" for i in range(8)]].values,axis=1)
        estimass = np.nanmean(aj[[f"mass_{i}" for i in range(8)]].values,axis=1)
        numb = np.nansum(aj[[f"masscount_{i}" for i in range(8)]].values,axis=1)
        isok = numb > 300
        print(aj.iloc[np.nanargmax(estimass)])
        plt.scatter(low[isok], estimass[isok])
        plt.show()
    raise Exception

#    decoratedflights = flights.join(dfadded,)
    # print()
    # raise Exception
    print(flights.ades.unique())
    dfjoined = trajs.join(flights.set_index("flight_id"),on="flight_id",how="right",validate="many_to_one")
    print(dfjoined.groupby("aircraft_type").flight_id.nunique())
    print(list(flights.aircraft_type.unique()))
    print(dfjoined.shape)
    # raise Exception
    # acft = 'A20N'
    threshold_vr = 500 * utils.FEET2METER / 60
    threshold_alt = 40000 * utils.FEET2METER
    df  = dfjoined.query("vertical_rate < @threshold_vr")#.query("altitude < @threshold_alt")#query("aircraft_type == @acft").
    # total_energy_poly = total_energy_polynomial_equation(thrust,dragpoly,e_rate,df.tas.values)
    # print(total_energy_poly)
    df["estimated_mass"] = compute_mass(df)#total_energy_poly.roots2()[:,0]
    print(df["estimated_mass"])
    grouped = df.groupby("flight_id")
    count = grouped.estimated_mass.count()
    emass = grouped.estimated_mass.median()[count>100]
    vmass = grouped.tow.median()[count>100]
    print(emass)
    print(count)
    print(vmass)
    model = linear_model.Ridge(alpha=1e-10)
    x = np.array([emass]).T
    print(x.shape)
    model.fit(x,vmass)
    print(np.corrcoef(vmass,emass))
    ypred = model.predict(x)
    print(root_mean_squared_error(ypred,vmass))
    fid = emass.idxmax()
    print(fid)
    traj = trajs.query("flight_id==@fid")
    dftraj = df.query("flight_id==@fid")
    trajsf = readers.read_trajectories(os.path.join("/disk2/prc/trajectories",fnametraj))
    trajf = trajsf.query("flight_id==@fid")
    # print(f"nb flights {acft}",dfjoined.query("aircraft_type == @acft").flight_id.nunique())
    plt.scatter(traj.timestamp,traj.altitude,s=20)
    plt.scatter(trajf.timestamp,trajf.altitude,c="red",s=3)
    plt.scatter(dftraj.timestamp,dftraj.altitude,c="yellow",s=2)
    plt.show()
    plt.scatter(traj.timestamp,traj.tas)
    plt.show()
    plt.scatter(vmass,emass)
    plt.show()
    raise Exception

if __name__ == '__main__':
    main()
