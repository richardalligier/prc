import pandas as pd
import utils
import os
import matplotlib.pyplot as plt
from traffic.core import Traffic
import numpy as np
import argparse
import readers
import csaps
import scipy
from scipy.ndimage import gaussian_filter1d


MAX_HOLE_SIZE = 20 # seconds


DICO_HOLE_SIZE = {}

def spline(t,v,smooth,derivative=False):
    isok = np.logical_not(np.isnan(v))
    if isok.sum()>2:
        cspline = csaps.csaps(xdata=t[isok],ydata=v[isok],smooth=smooth)#.spline
        if derivative:
            return cspline(t),cspline(t,nu=1)
        else:
            return cspline(t),None
    else:
        dv = np.empty_like(v)
        dv[:]=np.nan
        return v,dv

# pd.DataFrame.interpolate does not use index values for the 'limit' parameter :-(
# so I implement my own
def compute_holes(t,inans):
    tnan = t.copy()
    tnan[inans] = np.nan
    tf = pd.DataFrame({"tf":tnan}, dtype=np.float64).ffill().values
    tb = pd.DataFrame({"tb":tnan}, dtype=np.float64).bfill().values
    return tb - tf

def interpolate(df,smooth):
    t = ((df.timestamp - df.timestamp.iloc[0]) / pd.to_timedelta(1, unit="s")).values.astype(np.float64)
    df["t"] = t
    masknan = np.isnan(df["track"].values)
    track = df[["track"]].ffill().bfill().values[:,0]
    unwraped = np.unwrap(track,period=360)
    unwraped[masknan]=np.nan
    df["track_unwrapped"] = unwraped

    assert (t[1:]>t[:-1]).all()
    ddt = {}
    lnanvar = ['latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'track_unwrapped', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity','gsx', 'gsy', 'tasx', 'tasy', 'tas', 'wind']
    df = df.drop(columns="track")
    dvar = (
        (smooth,('latitude', 'longitude')),
        (smooth,('altitude',)),
        (smooth*0.1,('groundspeed','gsx', 'gsy', 'tasx', 'tasy', 'tas', 'wind','u_component_of_wind', 'v_component_of_wind')),
        (smooth,('track_unwrapped',)),
        (smooth*0.1,('vertical_rate',)),
        (smooth,('temperature', 'specific_humidity')),
    )
    for v in lnanvar:
        ddt[v] = compute_holes(df["t"],np.isnan(df[v].values))
    dico  = {}
    for smooth,lvar in dvar:
        for v in lvar:
            deriv = v == "altitude"
            st,dst = spline(t,df[v].values,smooth,derivative=deriv)
            dico[v]=st
            if deriv:
                dico["d"+v]=dst * 60
    res = df.assign(**dico)
    for v in lnanvar:
        res[v] = res[[v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
        res[v] = res[[v]].mask(np.isnan(ddt[v]))
        if ("d"+v) in dico:
            res["d"+v] = res[["d"+v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
            res["d"+v] = res[["d"+v]].mask(np.isnan(ddt[v]))
    return res.drop(columns="t")



def main():
    parser = argparse.ArgumentParser(
                    description='sort points of each trajectory by date, and convert units to SI units, and store good dtype',
    )
    parser.add_argument("-t_in",required=True)
    parser.add_argument("-t_out",required=True)
    parser.add_argument("-smooth",type=float,required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.t_in)
    # print(list(df))
    for v in ["flight_id", "icao24"]:
        df[v] = df[v].astype(np.int64)
    df = readers.convert_from_SI(readers.add_features_trajectories(readers.convert_to_SI(df)))
    df = df.groupby("flight_id").apply(lambda x:interpolate(x,args.smooth),include_groups=False).reset_index()
    df.drop(columns="level_1").to_parquet(args.t_out,index=False)



if __name__ == '__main__':
    main()
