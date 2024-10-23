import pandas as pd

import os
from os.path import isfile, join
import utils
import numpy as np
import readers
import argparse


def read_METARS(dirname):
    onlyfiles = [f for f in os.listdir(dirname) if isfile(join(dirname, f))]#[:1]
    dtype = {}
    for x in ["lon","lat","elevation","tmpf","dwpf","relh","drct","sknt","p01i","alti","mslp","vsby","gust","skyl1","skyl2","skyl3","skyl4","feel"]:
        dtype[x] = np.float64
    for x in ["station","skyc1","skyc2","skyc3","skyc4","wxcodes","metar"]:
        dtype[x]="string"
    df = pd.concat([pd.read_csv(join(dirname,f),comment="#",dtype=dtype).astype({"valid":"datetime64[ns, UTC]"}) for f in onlyfiles])
    return df.sort_values("valid")

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res

def build_correspondance(metars,airports):
    dfstation = metars[["station","lon","lat"]].drop_duplicates()
    lat1 = dfstation.lat.values[:,None]
    lon1 = dfstation.lon.values[:,None]
    lat2 = airports.latitude_deg.values[None,:]
    lon2 = airports.longitude_deg.values[None,:]
    dist = haversine_distance(lat1,lon1,lat2,lon2)
    iairports=np.argmin(dist,axis=0)
    dtoairports = np.min(dist,axis=0)
    # print(dtoairports.shape)
    return pd.DataFrame({"icao_code":airports.icao_code.values,"station":dfstation.station.values[iairports],"distance_ident_station":dtoairports})


# flights["actual_offblock_time"]=flights["actual_offblock_time"].astype("datetime64[ns, UTC]")
# res=pd.merge_asof(flights.sort_values("actual_offblock_time"),metars,left_on="actual_offblock_time",right_on="valid",left_by="ades",right_by="station",right_index=False,left_index=False,direction="nearest",tolerance=pd.Timedelta("2hour"))
# res[["date","actual_offblock_time","valid","ades","station","tmpf","wxcodes"]]

def build_metars_ident_station(metars,airports):
    correspondance = build_correspondance(metars,airports)
    return metars.join(correspondance.set_index("station"),on="station")

def main():
    parser = argparse.ArgumentParser(
                    prog='METARs weather processor',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    subparsers = parser.add_subparsers(dest="command",help='sub-command help')
    parser_a = subparsers.add_parser('to_parquet', help='a help')
    parser_a.add_argument("-metars_folder_in")
    parser_a.add_argument("-metars_parquet_out")
    parser_a.add_argument("-airports")
    args = parser.parse_args()
    if args.command == "to_parquet":
        airports = pd.read_parquet(args.airports)
        # airports = airports.query("ident!='EGSY'") # drop shefield city heliport
        print(airports.dtypes,airports.shape)

        # print(airports.groupby("icao_code").count().idxmax())
        # print(usedairports-set(airports.gps_code.values))
        # print(usedairports-set(airports.icao_code.values))
        # print(airports.dtypes,airports.shape)
        # raise Exception
        metars = read_METARS(args.metars_folder_in)
        df = build_metars_ident_station(metars,airports)
        print(metars.shape)
        print(df.shape)
        print(df.dtypes)
        df.to_parquet(args.metars_parquet_out)
    # elif args.command == "a":
    #     metars = pd.read_parquet("/disk2/prc/METARs.parquet")
    #     airports = pd.read_csv("airports.csv")

main()
