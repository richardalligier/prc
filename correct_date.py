import argparse
import pandas as pd
import pitot.geodesy
import readers
import numpy as np
import utils


# def correct_date(dfjoined,airports):
#     dfjoined["t_takeoff"] = dfjoined.actual_offblock_time + pd.to_timedelta(dfjoined.taxiout_time,unit="minutes")
#     # dfjoined = dfjoined.join(airports.set_index("icao_code"),on=apt,how="left")
#     lres = []
#     for (apt,tapt) in [("ades","arrival_time"),("adep","t_takeoff")]:
#         adf = dfjoined.join(airports[["icao_code","latitude_deg","longitude_deg"]].set_index("icao_code"),on=apt,how="left")
#         adf = adf.query("groundspeed>50")
#         adf = adf.assign(**{"distance_"+apt:pitot.geodesy.distance(adf.latitude,adf.longitude,adf.latitude_deg,adf.longitude_deg)/utils.NM2METER})
#         idxdistmin = adf.groupby("flight_id")["distance_"+apt].idxmin()
#         res = adf.loc[idxdistmin,["flight_id","timestamp","distance_"+apt]]
#         res["timedecal_"+apt] = (adf["timestamp"] - adf[tapt]).dt.total_seconds()
#         lres.append(res.drop(columns="timestamp").set_index("flight_id"))
#     return pd.concat(lres,axis=1).reset_index()

def correct_date(flights,airports,trajs):
    # print(list(cdates))
    # flights["t_adep"] = flights.actual_offblock_time + pd.to_timedelta(flights.taxiout_time-6,unit="minutes")
    dfjoined = trajs.join(flights.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#
    res = dfjoined[["flight_id"]].drop_duplicates()#.set_index("flight_id")
    for apt in ["adep","ades"]:
        adf = dfjoined.join(airports[["icao_code","latitude_deg","longitude_deg"]].set_index("icao_code"),on=apt,how="left").reset_index()
        adf[f"distance_{apt}"] = pitot.geodesy.distance(adf.latitude,adf.longitude,adf.latitude_deg,adf.longitude_deg)/utils.NM2METER
        mindist = adf.query(f"distance_{apt}<10").groupby("flight_id")
        idxdistmin = mindist["timestamp"].idxmin() if apt == "ades" else mindist["timestamp"].idxmax()
        xtime=adf.loc[idxdistmin,["flight_id","timestamp","altitude"]]
        # rocd = -1000 if apt =="ades" else 1000
        # xtime["timestamp"]=xtime["timestamp"]-pd.to_timedelta(xtime.altitude/rocd,unit="minutes")
        x = xtime[["flight_id","timestamp","altitude"]].rename(columns={"timestamp":f"t_{apt}","altitude":f"alt_{apt}"}).set_index("flight_id")
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
    return res.reset_index()#dfjoined[["flight_id","t_adep","t_ades","f_duration"]]




def main():
    import readers
    parser = argparse.ArgumentParser(
                    prog='add_wind',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-airports")
    parser.add_argument("-f_out")
    args = parser.parse_args()
    flights = readers.read_flights(args.f_in)
    trajs = pd.read_parquet(args.t_in)
    # dfjoined = trajs.join(flights.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#.query("aircraft_type=='B738'")
    airports = pd.read_parquet(args.airports)
    # correct_date(dfjoined,airports).to_parquet(args.f_out)
    correct_date(flights,airports,trajs).to_parquet(args.f_out)
if __name__ == '__main__':
    main()
