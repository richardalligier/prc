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


DICO_HOLE_SIZE = {}#"track_unwrapped":5}
# vol bizarre 248751045 2022-01-01.parquet

#SMOOTH = 1e-2


# def splinedzad(t,v,smooth):
#     isok = np.logical_not(np.isnan(v))
#     nok = isok.sum()
#     if nok > 2:
#         cspline = gaussian_filter1d(v[])
#         return cspline(t),cspline(x=t,nu=1)
#     else:
#         return v


def splineazdzdz(t,v,smooth):
    isok = np.logical_not(np.isnan(v))
    nok = isok.sum()
    vmax = np.nanmax(np.abs(v))
    if nok > 2:
        cspline = scipy.interpolate.UnivariateSpline(x=t[isok],y=v[isok]/vmax,w=1/nok*np.ones_like(t[isok]),s=smooth,k=2)#,xidata=t,smooth=smooth)#.spline
        return cspline(t)*vmax#,cspline(x=t,nu=1)
    else:
        return v

def spline(t,v,smooth,derivative=False):
    isok = np.logical_not(np.isnan(v))
    # print(f"{isok.sum() =}")
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
    # print(list(df))
    # raise Exception
    t = ((df.timestamp - df.timestamp.iloc[0]) / pd.to_timedelta(1, unit="s")).values.astype(np.float64)
    df["t"] = t
    # print(df["track"])
    # print(df["track_unwrapped"].describe())
    masknan = np.isnan(df["track"].values)
    track = df[["track"]].ffill().bfill().values[:,0]
    # print(track.shape)
    unwraped = np.unwrap(track,period=360)
    unwraped[masknan]=np.nan
    df["track_unwrapped"] = unwraped
    # print(df["track_unwrapped"])
    # print(t[1:] - t[:-1])
    # print(df.reset_index().iloc[-3])
    # print(df.reset_index().iloc[-2])
    # print(df.reset_index().iloc[-1])
    # icheck = np.argmax(np.logical_not(t[1:]>t[:-1]))
    # print(df.reset_index(drop=True).iloc[icheck:icheck+2])
    # print(list(df))
    # print(df.flight_id.unique())
    # print(t)
    assert (t[1:]>t[:-1]).all()
    ddt = {}
    # print(list(df))
    # raise Exception
    lnanvar = ['latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'track_unwrapped', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity','gsx', 'gsy', 'tasx', 'tasy', 'tas', 'wind']#["altitude","latitude","longitude","groundspeed","vertical_rate","track","track_unwrapped"]
    df = df.drop(columns="track")
    # print(df.shape)
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
        # print(v)
        # print(ddt[v])
    # res = df.set_index("t").interpolate(limit_area='inside', method="index").reset_index(drop=True)
    dico  = {}
    for smooth,lvar in dvar:
        for v in lvar:
            deriv = v == "altitude"
            st,dst = spline(t,df[v].values,smooth,derivative=deriv)
            dico[v]=st
            if deriv:
                dico["d"+v]=dst * 60
                # print(dst)
                # raise Exception
            # dico["d"+v]=dst
            # print(v)
    res = df.assign(**dico)
    # print(res.describe())
    # print(list(res))
    for v in lnanvar:
        # print(v)
        res[v] = res[[v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
        res[v] = res[[v]].mask(np.isnan(ddt[v]))
        if ("d"+v) in dico:
            res["d"+v] = res[["d"+v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
            res["d"+v] = res[["d"+v]].mask(np.isnan(ddt[v]))
        # res["d"+v] = res[["d"+v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
        # res["d"+v] = res[["d"+v]].mask(np.isnan(ddt[v]))
    # print(list(res))
    return res.drop(columns="t")

def maintest():
    config = utils.read_config()
#    248750944
    # fid = 248751068
    # df = pd.read_parquet(os.path.join(config.trajectories,"2022-01-01.parquet"))#.query("flight_id==@fid")
    # fid = 248759896
    fid=248750643
    df = pd.read_parquet("/disk2/prc/filtered_trajectories/2022-01-01.parquet").query("flight_id==@fid")
    df = readers.convert_from_SI(readers.add_features_trajectories(readers.convert_to_SI(df)))
#    df = pd.read_parquet("/disk2/prc/filtered_trajectories/2022-01-01.parquet")#.drop_duplicates(["flight_id","timestamp"])#.query("flight_id==@fid")
    # print(list(df))
    # raise Exception
    dfi = df.groupby("flight_id").apply(lambda x:interpolate(x,1e-2),include_groups=False).reset_index()
    # print(dfi)
    # raise Exception
    # dfi = tf.data
    # dfi = df.set_index("timestamp").interpolate(limit_area='inside',method="index").reset_index()
    print(df.flight_id.nunique())
    for fid in df.flight_id.unique():
        print(fid)
        traj = df.query("flight_id==@fid")
        traji = dfi.query("flight_id==@fid")
        plt.scatter(traji.timestamp,traji.track_unwrapped,s=20)
        plt.scatter(traj.timestamp,traj.track,s=3)
        plt.show()
        plt.scatter(traji.timestamp,traji.groundspeed,s=20)
        plt.scatter(traj.timestamp,traj.groundspeed,s=3)
        plt.show()
        plt.scatter(traji.timestamp,traji.altitude,s=20)
        plt.scatter(traj.timestamp,traj.altitude,s=3)
        plt.show()
        plt.scatter(traji.timestamp,traji.vertical_rate,s=20)
        plt.scatter(traj.timestamp,traj.vertical_rate,s=3)
        plt.show()
        # plt.scatter(traj.timestamp,traj.vertical_rate,s=3)
        # plt.scatter(traji.timestamp,traji.nana,s=3)
        # plt.scatter(traji.timestamp,traji.tb,s=3)
        # plt.scatter(traji.timestamp,np.maximum(traji.tf,traji.tb),s=3)
        # plt.scatter(traji.timestamp,traji.tb,s=3)
        # plt.scatter(traji.timestamp,traji.tf,s=3)



def main():
    parser = argparse.ArgumentParser(
                    prog='trajs normalizer',
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
