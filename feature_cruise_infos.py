import argparse
import pandas as pd
import pitot.aero
import readers
import numpy as np

from correct_date import compute_dates, fillnadates

def feature_cruise_infos(trajs,cdates,nsplit):
    cdates["f_duration"] = (cdates.t_ades - cdates.t_adep).dt.total_seconds()
    dfjoined = trajs.join(cdates.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")
    dfjoined["t"] = (dfjoined.timestamp - dfjoined.t_adep).dt.total_seconds()/(dfjoined.f_duration)
    dfjoined["mach"] = pitot.aero.tas2mach(dfjoined.tas,dfjoined.altitude)
    slices = np.linspace(0,1,nsplit+1)
    lresults = []
    results = cdates[["flight_id","f_duration"]].set_index("flight_id")
    for i,(tdeb,tend) in enumerate(zip(slices[:-1],slices[1:])):
        grouped = dfjoined.query("@tdeb <= t <= @tend").groupby("flight_id")
        lresults.append(grouped.mach.median().rename(f"Cruisemach_{i}"))
        lresults.append(grouped.mach.count().fillna(0).rename(f"Cruisemachcount_{i}"))
        lresults.append((grouped.altitude.last()-grouped.altitude.first()).rename(f"CruiseDeltaAlt_{i}"))
        lresults.append(grouped.altitude.median().rename(f"CruiseMedianAlt_{i}"))
    results = pd.concat([results]+lresults,axis=1)
    print("results.shape",results.shape)
    return results.reset_index()


def main():
    import readers
    parser = argparse.ArgumentParser(
                    description='compute features related to cruise phase by computing stats on slices aligned on time',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-nsplit",type=int)
    parser.add_argument("-airports")
    parser.add_argument("-f_out")
    args = parser.parse_args()
    trajs = pd.read_parquet(args.t_in)
    flights=readers.read_flights(args.f_in)
    cdates=fillnadates(flights,compute_dates(flights,pd.read_parquet(args.airports),trajs))
    del flights
    res = feature_cruise_infos(trajs,cdates,args.nsplit)
    print(list(res))
    res.to_parquet(args.f_out)
if __name__ == '__main__':
    main()
