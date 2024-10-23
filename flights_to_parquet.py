import pandas as pd
import argparse
import numpy as np

def read_flights(f):
    dates = ["date","actual_offblock_time","arrival_time"]
    ltypes = [
        ("string", ["adep","ades"]),#4
        ("string", ["callsign","airline"]),#32
        ("string",["wtc"]),#1
        ("string",["country_code_ades","country_code_adep","name_ades","name_adep"]),#2
        ("string",["aircraft_type"]),#4
        (np.float64, ["flight_duration","taxiout_time","flown_distance","tow"]),
        (np.int64, ["flight_id"]),
        ("datetime64[ns, UTC]", dates),
    ]
    df = pd.read_csv(f)
    for dtype,lvar in ltypes:
        for v in lvar:
            df[v]=df[v].astype(dtype)
    return df
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f_in")
    parser.add_argument("-f_out")
    args = parser.parse_args()
    read_flights(args.f_in).to_parquet(args.f_out,index=False)
main()
