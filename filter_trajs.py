import pandas as pd
import numpy as np
import utils
import argparse
from filterclassic import FilterCstPosition, FilterCstSpeed, MyFilterDerivative, FilterCstLatLon,FilterIsolated

# unwrap wrong on 248803487 of 2022-01-03
from traffic.core import Traffic
import matplotlib.pyplot as plt


def nointerpolate(x):
    return x

def read_trajectories(f, strategy):
    df = pd.read_parquet(f)
    for v in ["flight_id"]:
        df[v] = df[v].astype(np.int64)
    df = df.drop_duplicates(["flight_id","timestamp"]).sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.head(10_000)
    if strategy == "classic":
        filter = FilterCstLatLon()|FilterCstPosition()|FilterCstSpeed()|MyFilterDerivative()|FilterIsolated()
    else:
        raise Exception(f"strategy '{strategy}' not implemented")
    dftrafficin = Traffic(df).filter(filter=filter,strategy=nointerpolate).eval(max_workers=1).data
    dico_tomask = {
        # "track":["track_unwrapped"],
        "latitude":["u_component_of_wind","v_component_of_wind","temperature"],
        "altitude":["u_component_of_wind","v_component_of_wind","temperature"],
    }
    for k,lvar in dico_tomask.items():
        for v in lvar:
            dftrafficin[v] = dftrafficin[[v]].mask(dftrafficin[k].isna())
    return dftrafficin

def main():
    parser = argparse.ArgumentParser(
                    prog='trajs normalizer',
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-t_in")
    parser.add_argument("-t_out")
    parser.add_argument("-strategy")
    args = parser.parse_args()
    df = read_trajectories(args.t_in,args.strategy)
    df.to_parquet(args.t_out,index=False)



if __name__ == '__main__':
    main()
