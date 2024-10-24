import argparse
import pandas as pd
import pitot.aero
import pitot.geodesy
import readers
import numpy as np
import utils
from correct_date import compute_dates, fillnadates

def add_wind_effect(trajs,cdates):
    dfjoined = trajs.join(cdates.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")
    dfjoined = dfjoined.query("t_adep<=timestamp").query("timestamp<=t_ades")
    dfjoined["wind_effect"]= (dfjoined["gsx"] * dfjoined["u_component_of_wind"]+dfjoined["gsy"]* dfjoined["v_component_of_wind"])/dfjoined["groundspeed"]
    gdf = dfjoined[["flight_id","wind_effect"]].groupby("flight_id").mean()
    return gdf.reset_index()


def main():
    import readers
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
    parser.add_argument("-airports")
    parser.add_argument("-f_out")
    args = parser.parse_args()
    trajs = pd.read_parquet(args.t_in)
    flights = readers.read_flights(args.f_in)
    cdates = compute_dates(flights,pd.read_parquet(args.airports),trajs)
    cdates = fillnadates(flights,cdates)
    del flights
    res = add_wind_effect(trajs,cdates)
    # print(res)
    res.to_parquet(args.f_out)#,index=False)
if __name__ == '__main__':
    main()
