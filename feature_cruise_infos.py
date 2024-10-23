import argparse
import pandas as pd
import pitot.aero
import pitot.geodesy
import readers
import numpy as np
import utils
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("error")
# def compute_dates(flights,cdates):
#     # print(list(cdates))
#     dfjoined = flights.join(cdates.set_index("flight_id"),on="flight_id",how="inner",validate="one_to_one")#.query("aircraft_type=='B738'")
#     dfjoined["t_takeoff"] = dfjoined.actual_offblock_time + pd.to_timedelta(dfjoined.taxiout_time,unit="minutes")
#     for (t,apt) in [("t_takeoff","adep"),("arrival_time","ades")]:
#         dfjoined["t_"+apt] = dfjoined[t] + pd.to_timedelta(dfjoined["timedecal_"+apt],unit="seconds")
#     dfjoined["f_duration"] = (dfjoined.t_ades -dfjoined.t_adep).dt.total_seconds()
#     return dfjoined


def joincdates(flights,airports,trajs,rod=-1000,roc=1000):
    cdates=fillnadates(flights,compute_dates(flights,airports,trajs,rod=rod,roc=roc))
    # n = flights.shape[0]
    # print(n)
    flights =flights.join(cdates.set_index("flight_id"),on="flight_id",how="inner")
    # print(flights.shape[0])
    # assert(n==flights.shape[0])
    return flights

def compute_dates(flights,airports,trajs,rod=-1000,roc=1000):
    # print(list(cdates))
    # flights["t_adep"] = flights.actual_offblock_time + pd.to_timedelta(flights.taxiout_time-6,unit="minutes")
    dfjoined = trajs.join(flights.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#
    res = dfjoined[["flight_id"]].drop_duplicates()#.set_index("flight_id")
    for apt in ["adep","ades"]:
        adf = dfjoined.join(airports[["icao_code","latitude_deg","longitude_deg"]].set_index("icao_code"),on=apt,how="left").reset_index()
        adf[f"distance_{apt}"] = pitot.geodesy.distance(adf.latitude,adf.longitude,adf.latitude_deg,adf.longitude_deg)/utils.NM2METER
        print(adf[f"latitude_deg"].isna().any())
        print(adf[f"longitude_deg"].isna().any())
        print(adf[f"latitude"].isna().any())
        print(adf[f"longitude"].isna().any())
        print(adf[f"distance_{apt}"].isna().any())
        mindist = adf.query(f"distance_{apt}<10").groupby("flight_id")
        print(mindist.flight_id.nunique())
#        print(mindist.flight_id.nunique()==flights.flight_id.nunique())

        idxdistmin = mindist["timestamp"].idxmin() if apt == "ades" else mindist["timestamp"].idxmax()
        xtime=adf.loc[idxdistmin,["flight_id","timestamp","altitude"]]
        rocd = rod if apt =="ades" else roc
        timedelta = (xtime.altitude/rocd*60).round() # 2 decimals rounding resolution:60/100 [s]
        sign = np.sign(timedelta)
        # print(sign)
        # print(timedelta)
        # print(timedelta.max())
        # print(timedelta.min())
        # feed positive timedelta, otherwise, random overflow error from one run to another on the exact same data with the exact same code
        xtime["timestamp"]=xtime["timestamp"]-sign*pd.to_timedelta(timedelta.abs(),unit="seconds")
        x = xtime[["flight_id","timestamp"]].rename(columns={"timestamp":f"t_{apt}"}).set_index("flight_id")
#        print(x)
        res = res.join(x,on="flight_id",how="left")
        # lres.append(.set_index("flight_id"))
    # res = pd.concat(lres,axis=1).reset_index()
#    print(res)
    # res["timestamp"] = res["timestamp"] + pd.to_timedelta(6,unit="minutes")
    # print(f"{res['timestamp'].isna().mean() =}")
    # res = res.rename(columns={"timestamp":"t_ades"})
    # print(list(res))
    # traj = adf.query("flight_id==256760815")
    # plt.scatter(traj.timestamp,traj.distance_ades)
    # plt.show()
    return res#dfjoined[["flight_id","t_adep","t_ades","f_duration"]]

def fillnadates(flights,cdates):
    df = flights[["flight_id","arrival_time"]].assign(t_adep=flights.actual_offblock_time + pd.to_timedelta(flights.taxiout_time,unit="minutes")).rename(columns={"arrival_time":"t_ades"})
    df = df.set_index("flight_id").loc[cdates.flight_id]#.reset_index()
    # print(df)
    # raise Exception
    return cdates.set_index("flight_id").combine_first(df).reset_index()

def add_cruise_infos(trajs,cdates,nsplit):
    cdates["f_duration"] = (cdates.t_ades -cdates.t_adep).dt.total_seconds()
    dfjoined = trajs.join(cdates.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#.query("aircraft_type=='B738'")
    dfjoined["t"] = (dfjoined.timestamp - dfjoined.t_adep).dt.total_seconds()/(dfjoined.f_duration)
    dfjoined["mach"] = pitot.aero.tas2mach(dfjoined.tas,dfjoined.altitude)
    # nsplit = 20
    slices = np.linspace(0,1,nsplit+1)
    # print(dfjoined.t.describe())
    # print(dfjoined.t.idxmax())
    # print(dfjoined[["flight_id","t_adep","t_ades","t_takeoff","arrival_time","t","timestamp"]].query("flight_id==248756344"))
    lresults = []
    results = cdates[["flight_id","f_duration"]].set_index("flight_id")#pd.DataFrame({"flight_id":dfjoined.flight_id.unique()}).set_index("flight_id")
    for i,(tdeb,tend) in enumerate(zip(slices[:-1],slices[1:])):
        # traj = dfjoined.query("@tdeb <= t <= @tend").query("flight_id==256760815")
        # plt.scatter(traj.timestamp,traj.altitude)
        # plt.scatter(traj.longitude,traj.latitude)
        grouped = dfjoined.query("@tdeb <= t <= @tend").groupby("flight_id")
        lresults.append(grouped.mach.median().rename(f"Cruisemach_{i}"))
        lresults.append(grouped.mach.count().fillna(0).rename(f"Cruisemachcount_{i}"))
        lresults.append((grouped.altitude.last()-grouped.altitude.first()).rename(f"CruiseDeltaAlt_{i}"))
        lresults.append(grouped.altitude.median().rename(f"CruiseMedianAlt_{i}"))
    # print(traj.t_adep,traj.t_ades,traj.f_duration)
    # plt.show()
    results = pd.concat([results]+lresults,axis=1)
    print("results.shape",results.shape)
    return results.reset_index()
    # flights["flight_time"] = flights.actual_offblock_time + (flights.arrival_time-flights.actual_offblock_time) / 2


def main():
    import readers
    parser = argparse.ArgumentParser(
                    prog='add_wind',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-nsplit",type=int)
    # parser.add_argument("-corrected_dates")
    parser.add_argument("-airports")
    parser.add_argument("-f_out")
    args = parser.parse_args()
    # flights = compute_dates(readers.read_flights(args.f_in),pd.read_parquet(args.corrected_dates))
    trajs = pd.read_parquet(args.t_in)
    flights = readers.read_flights(args.f_in)
    cdates = compute_dates(flights,pd.read_parquet(args.airports),trajs)
    cdates = fillnadates(flights,cdates)
 #   print(cdates)
    del flights
    # print(args.f_out)
    res = add_cruise_infos(trajs,cdates,args.nsplit)#pd.concat([add_cruise_infos(dfjoined),flights[["flight_id","f_duration"]].set_index("flight_id")],axis=1).reset_index()
    # print(res.shape)
    # print(res.CruiseDeltaAlt_0.idxmax())
    # print(res[["CruiseMedianAlt_0","Cruisemach_0","CruiseDeltaAlt_0","f_duration"]].describe())
    res.to_parquet(args.f_out)#,index=False)
if __name__ == '__main__':
    main()
