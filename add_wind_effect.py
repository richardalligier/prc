import argparse
import pandas as pd
import pitot.aero
import pitot.geodesy
import readers
import numpy as np
import utils
from add_cruise_infos import compute_dates, fillnadates

def add_wind_effect(trajs,cdates):
 #   cdates["f_duration"] = (cdates.t_ades -cdates.t_adep).dt.total_seconds()
    dfjoined = trajs.join(cdates.set_index("flight_id"),on="flight_id",how="inner",validate="many_to_one")#.query("aircraft_type=='B738'")
    dfjoined = dfjoined.query("t_adep<=timestamp").query("timestamp<=t_ades")
    dfjoined["wind_effect"]= (dfjoined["gsx"] * dfjoined["u_component_of_wind"]+dfjoined["gsy"]* dfjoined["v_component_of_wind"])/dfjoined["groundspeed"]
    gdf = dfjoined[["flight_id","wind_effect"]].groupby("flight_id").mean()
    return gdf.reset_index()
    # flights["flight_time"] = flights.actual_offblock_time + (flights.arrival_time-flights.actual_offblock_time) / 2


def main():
    import readers
    parser = argparse.ArgumentParser(
                    prog='add_wind',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-t_in")
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
    res = add_wind_effect(trajs,cdates)#pd.concat([add_cruise_infos(dfjoined),flights[["flight_id","f_duration"]].set_index("flight_id")],axis=1).reset_index()
    print(res)
    # print(res.shape)
    # print(res.CruiseDeltaAlt_0.idxmax())
    # print(res[["CruiseMedianAlt_0","Cruisemach_0","CruiseDeltaAlt_0","f_duration"]].describe())
    res.to_parquet(args.f_out)#,index=False)
if __name__ == '__main__':
    main()
