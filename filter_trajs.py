import pandas as pd
import numpy as np
import utils
import argparse
from filterclassic import FilterCstPosition, FilterCstSpeed, MyFilterDerivative, FilterCstLatLon,FilterIsolated

# unwrap wrong on 248803487 of 2022-01-03



from traffic.core import Traffic
import consistency

import matplotlib.pyplot as plt
# def interpolate(x):
#     return x.set_index("timestamp").interpolate(limit_area='inside',limit=60,limit_direction='both',method='index').reset_index(drop=False)
def nointerpolate(x):
    return x
def read_trajectories(f, strategy):
    df = pd.read_parquet(f)
    # print(list(df))
    for v in ["flight_id"]:
        df[v] = df[v].astype(np.int64)
    ###  2022-07-13####
    #df = df.query("flight_id==253412168")
    # df = df.query("flight_id==253415180")
    # df = df.query("flight_id==253384574")
    ###  2022-01-01####
    # df = df.query("flight_id==248758607")
    # df = df.query("flight_id==248759396") # 500 better
    #df = df.query("flight_id==248759896") # 4000 ok
    # df = df.query("flight_id==248757278") # 4000 better
    #df = df.query("flight_id==248752191") # way better 4000 ok
    # df = df.query("flight_id==248765562") # way better
    ###  2022-01-01####
    # df = df.query("flight_id==248784182")
    ### 2022-01-03####
    # df = df.query("flight_id==248803487") #ok fait pleins de cercles
    # df = df.query("flight_id==248809247") #ok track angle & groundspeed
#    df = df.query("flight_id==248793143") #ok
    # df = df.query("flight_id==248805448") #ok
    # df = df.query("flight_id==248795587") # repeating backward in time altitude
#    248805371
#    df = df.query("flight_id==248795587")
    #################
    # print(df.dtypes)
    # print(df.shape)
    # print(df.flight_id.nunique())
    df = df.drop_duplicates(["flight_id","timestamp"]).sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.head(10_000)
    # print(df)
    # df = df.query("flight_id==253415180")
    # df = df.query("flight_id==253384977")
    if strategy == "consistency":
        filter = consistency.FilterConsistency(
            exact_when_kept_below_verti = 0.7,
            exact_when_kept_below_track = 0.5,#0.5
            exact_when_kept_below_speed = 0.6,
            # exact_when_kept_below_verti = 1,
            # exact_when_kept_below_track = 1,
            # exact_when_kept_below_speed = 1,
            horizon = 200,
            backup_horizon = 2000,
            backup_exact = True,
            dalt_dt_error = 4000,
            dtrack_dt_error = 4,
            max_dtrack_dt = 6,
            dtrack_dt_error_extra_turn = 1.5,#2
            relative_error_on_dist = 8 / 100
        )
    elif strategy == "classic":
        filter = FilterCstLatLon()|FilterCstPosition()|FilterCstSpeed()|MyFilterDerivative()|FilterIsolated()#|MyFilterDerivative()|MyFilterDerivative()|MyFilterDerivative()|MyFilterDerivative()
    else:
        raise Exception(f"strategy '{strategy}' not implemented")
    # plt.plot(df.track.values,color="blue")#,s=30)
    # plt.show()
    # plt.scatter(df.timestamp,df.track,color="blue")#,s=30)
    # plt.show()
    dftrafficin = Traffic(df).filter(filter=filter,strategy=nointerpolate).eval(max_workers=1).data
    # print(list(dftrafficin))
    # print(dftrafficin.dtypes)
    # raise Exception
    dico_tomask = {
        # "track":["track_unwrapped"],
        "latitude":["u_component_of_wind","v_component_of_wind","temperature"],
        "altitude":["u_component_of_wind","v_component_of_wind","temperature"],
    }
    for k,lvar in dico_tomask.items():
        for v in lvar:
            # print(dftrafficin[v].isna().values.shape)
            # print(dftrafficin[v].shape)
            dftrafficin[v] = dftrafficin[[v]].mask(dftrafficin[k].isna())
    return dftrafficin
    for v in ["altitude","groundspeed","latitude","temperature"]:
        print(v,1-dftrafficin[v].isna().mean())
    # raise Exception
    import readers
    import interpolate
    df = df.copy()
    dftraffic = dftrafficin.copy()
    dfi = dftrafficin.copy()

    for fid in df.flight_id.unique():
        print(fid)
        traj = readers.add_features_trajectories(readers.convert_to_SI(df.query("flight_id==@fid")))
        trajf = readers.add_features_trajectories(readers.convert_to_SI(dftraffic.query("flight_id==@fid")))
        trajfi = readers.convert_to_SI(interpolate.interpolate(readers.convert_from_SI(readers.add_features_trajectories(readers.convert_to_SI(dfi.query("flight_id==@fid")))),smooth=1e-2))
        c ="red"
        ci = "green"
        # plt.scatter(traj.longitude, traj.latitude,color="blue")
        # plt.scatter(trajf.longitude, trajf.latitude,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.latitude,color="blue")
        # plt.scatter(trajf.timestamp, trajf.latitude,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.longitude,color="blue")
        # plt.scatter(trajf.timestamp, trajf.longitude,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.wind,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.wind,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.wind,color=ci,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.u_component_of_wind,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.u_component_of_wind,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.u_component_of_wind,color=ci,s=3)
        # plt.show()
        nanmask = np.logical_not(np.isnan(trajf.latitude.values))
        lat = trajf.latitude.values[nanmask]
        lon = trajf.latitude.values[nanmask]
        dists=consistency.distance(lat[1:],lon[1:],lat[:-1],lon[:-1])
        print(dists,dists.shape,np.sum(dists))
        # raise Exception


        plt.scatter(traj.timestamp, traj.altitude,color="blue")
        plt.scatter(trajf.timestamp, trajf.altitude,color=c,s=3)
        plt.scatter(trajfi.timestamp, trajfi.altitude,color=ci,s=1)
        plt.show()
        plt.scatter(traj.longitude, traj.latitude,color="blue",s=30)
        plt.scatter(trajf.longitude, trajf.latitude,color=c,s=10)
        plt.scatter(trajfi.longitude, trajfi.latitude,color=ci,s=3)
        plt.show()
        plt.scatter(traj.timestamp, traj.longitude,color="blue")
        plt.scatter(trajf.timestamp, trajf.longitude,color=c,s=3)
        plt.scatter(trajfi.timestamp, trajfi.longitude,color=ci,s=1)
        plt.show()
        plt.scatter(traj.timestamp, traj.latitude,color="blue")
        plt.scatter(trajf.timestamp, trajf.latitude,color=c,s=3)
        plt.scatter(trajfi.timestamp, trajfi.latitude,color=ci,s=1)
        plt.show()
        # plt.scatter(traj.timestamp, traj.track,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.track,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.track_unwrapped,color=ci,s=3)
        # plt.show()
        print(1-traj.groundspeed.isna().mean(),1-trajfi.groundspeed.isna().mean(),1-trajf.groundspeed.isna().mean())
        plt.scatter(traj.timestamp, traj.groundspeed,color="blue")
        plt.scatter(trajf.timestamp, trajf.groundspeed,color=c,s=3)
        plt.scatter(trajfi.timestamp, trajfi.groundspeed,color=ci,s=1)
        plt.show()
        # plt.scatter(traj.timestamp, traj.gsx,color="blue")
        # plt.scatter(trajf.timestamp, trajf.gsx,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.gsy,color="blue")
        # plt.scatter(trajf.timestamp, trajf.gsy,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.altitude,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.altitude,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.altitude,color=ci,s=3)
        # plt.show()
        # print("tas")
        # plt.scatter(traj.timestamp, traj.tas,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.tas,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.tas,color=ci,s=3)
        # plt.show()

        # plt.scatter(traj.timestamp, traj.vertical_rate,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.vertical_rate,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.vertical_rate,color=ci,s=3)
        # plt.show()

        # plt.scatter(traj.timestamp, traj.track,color="blue",s=30)
        # plt.scatter(traj.timestamp, traj.track_unwrapped,color="red",s=30)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.track,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.track,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.track,color=ci,s=5)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.track_unwrapped,color="blue",s=30)
        # plt.scatter(traj.timestamp, np.unwrap(traj.track_unwrapped.values),color="blue",s=30)
        # print(np.isnan(trajf.track_unwrapped.values).all())
        # print(trajf.track_unwrapped.isna().all())
        # print(np.isnan(trajfi.track_unwrapped.values).all())
        # plt.scatter(traj.timestamp, traj.track_unwrapped,color=c,s=10)
        # plt.scatter(trajf.timestamp, trajf.track_unwrapped,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.track_unwrapped,color=ci,s=5)
        # plt.show()

        # plt.scatter(traj.timestamp, traj.groundspeed,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.groundspeed,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.groundspeed,color=ci,s=5)
        # plt.show()

        # plt.scatter(traj.timestamp, traj.wind,color="blue",s=30)
        # plt.scatter(trajf.timestamp, trajf.wind,color=c,s=10)
        # plt.scatter(trajfi.timestamp, trajfi.wind,color=ci,s=5)
        # plt.show()

        # plt.scatter(traj.timestamp, traj.u_component_of_wind,color="blue")
        # plt.scatter(trajf.timestamp, trajf.u_component_of_wind,color=c,s=3)
        # plt.scatter(trajfi.timestamp, trajfi.u_component_of_wind,color=ci,s=1)
        # plt.show()


        # # plt.scatter(traj.timestamp, traj.track,color="blue")
        # # plt.scatter(trajf.timestamp, trajf.track,color=c,s=3)
        # # plt.show()

        # plt.scatter(traj.timestamp, traj.groundspeed,color="blue")
        # plt.scatter(trajf.timestamp, trajf.groundspeed,color=c,s=3)
        # plt.show()
        # plt.scatter(traj.timestamp, traj.altitude,color="blue")
        # plt.scatter(trajf.timestamp, trajf.altitude,color=c,s=3)
        # plt.show()
    # transfos = [
    #     (np.radians,["longitude", "latitude", "track", "track_unwrapped"]),
    #     (lambda x: x * utils.FEET2METER, ["altitude"]),
    #     (lambda x: x * utils.FEET2METER / 60, ["vertical_rate"]),
    #     (lambda x: x * utils.KTS2MS, ["groundspeed"]),
    # ]
    # for transfo,lvar in transfos:
    #     for v in lvar:
    #         df[v] = transfo(df[v])
    # print(df)



def plot_altitude_jump():
    import matplotlib.pyplot as plt
    config = utils.read_config()
    df = pd.read_parquet(os.path.join(config.FOLDER_DATA,"/rawtrajectories/2022-07-13.parquet")).sort_values(["flight_id","timestamp"])
    l = [253384586]
    for fid in l:
        flight = df.query("flight_id==@fid")
        plt.plot(flight.timestamp,flight.altitude)
        plt.show()

def plot_latitude_jump():
    import matplotlib.pyplot as plt
    config = utils.read_config()
    df = pd.read_parquet(os.path.join(config.FOLDER_DATA,"/rawtrajectories/2022-07-13.parquet")).sort_values(["flight_id","timestamp"])
    flight = df.query("flight_id==253384586")
    plt.plot(flight.timestamp,flight.altitude)
    plt.show()

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
