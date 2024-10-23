import pandas as pd
import numpy as np
import utils
import os


def add_features_trajectories(df):
    df = df.copy()
    # x east y north
    print("warning not using track_unwrapped")
    gsx = df.groundspeed * np.sin(df.track)
    gsy = df.groundspeed * np.cos(df.track)
    # groundspeed = airspeed + wind
    # u east v north
    tasx = gsx - df.u_component_of_wind
    tasy = gsy - df.v_component_of_wind
    tas = np.hypot(tasx,tasy)
    # energy_rate = df.vertical_rate
    return df.assign(
        gsx = gsx,
        gsy = gsy,
        tasx = tasx,
        tasy = tasy,
        tas = tas,
        wind = np.hypot(df.u_component_of_wind,df.v_component_of_wind)
    )

def applytransfo(df,lfactors,inverse=False):
    ldf = list(df)
    for factor,lvar in lfactors:
        for v in lvar:
            if v in ldf:
                df[v] = df[v] * (1 / factor if inverse else factor)
    return df

TO_SI = (
        (np.pi/180,["longitude", "latitude", "track", "track_unwrapped"]),
        (utils.FEET2METER, ["altitude"]),
        (utils.FEET2METER / 60, ["vertical_rate"]),
        (utils.KTS2MS, ["groundspeed","gsx","gsy","tasx","tasy","tas"]),
)

def convert_to_SI(df):
    df = df.copy()
    return applytransfo(df, TO_SI)


def convert_from_SI(df):
    df = df.copy()
    return applytransfo(df, TO_SI, inverse=True)



def read_trajectories(f):
    df = pd.read_parquet(f)
    for v in ["flight_id", "icao24"]:
        df[v] = df[v].astype(np.int64)
    df = convert_to_SI(df)
    # print(df.dtypes)
    return df#add_features_trajectories(df)
def add_features_flight(df):
    return df.assign(
        dayofweek = df.actual_offblock_time.dt.isocalendar().day,
        # dayofyear = df.actual_offblock_time.dt.isocalendar().dayofyear,
        weekofyear = df.actual_offblock_time.dt.isocalendar().week,
        arrival_minutes = df.arrival_time.dt.hour*60+df.arrival_time.dt.minute,
        actual_offblock_minutes = df.actual_offblock_time.dt.hour*60+df.actual_offblock_time.dt.minute,
        )
#.astype({"dayofweek":np.float64,"weekofyear":np.float64,"arrival_minutes":np.float64,"actual_offblock_minutes":np.float64})

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
    # usecols = [v for _,lvar in ltypes for v in lvar]
    # dtype = {v:t for t,lvar in ltypes for v in lvar}
    df = pd.read_parquet(f)
    for dtype,lvar in ltypes:
        for v in lvar:
            df[v]=df[v].astype(dtype)
    #,dtype=dtype,parse_dates=dates,usecols=usecols)
    # print(df.dtypes)
    # print(df.dtypes)
    return add_features_flight(df)


# def read_features(what):
#     config = utils.read_config()
#     # Flights()
#     flights = read_flights(os.path.join(config.flights,f"{what}.parquet"))
#     # masses = pd.read_parquet(os.path.join(config.FOLDER_DATA,f"masses_1e-12_1_500_1/{what}"))
#     masses = pd.read_parquet(os.path.join(config.FOLDER_DATA,f"masses_1e-2_5_500_1/{what}"))
#     print(list(masses))
#     for v in [f"Nonemassadepdate_{i}" for i in range(9)]:
#         masses[v]=masses[v].dt.seconds
#     for v in [f"Nonemasscount_{i}" for i in range(9)]:
#         masses[v]=masses[v].fillna(0)
#     # massesdesc = pd.read_parquet(os.path.join(config.FOLDER_DATA,f"massesdesc_10_1500_0/{what}"))
#     # for v in [f"descmassadesdate_{i}" for i in range(9)]:
#     #     massesdesc[v]=massesdesc[v].dt.seconds
#     # for v in [f"descmasscount_{i}" for i in range(9)]:
#     #     massesdesc[v]=massesdesc[v].fillna(0)
#     # nansum = np.sum(np.nan_to_num(massesdesc[[f"descmasscount_{i}" for i in range(9)]].values),axis=1)
#     # estimass = np.sum(np.nan_to_num(massesdesc[[f"descmass_{i}" for i in range(9)]].values*massesdesc[[f"descmasscount_{i}" for i in range(9)]].values),axis=1)/np.sum(np.nan_to_num(massesdesc[[f"descmasscount_{i}" for i in range(9)]].values),axis=1)
#     # massesdesc["count_total"] = nansum
#     # massesdesc["mass_total"]=estimass
#     icaos = pd.read_parquet(os.path.join(config.FOLDER_DATA,f"icaos/{what}"))
#     df = flights
#     for features in [masses]:#,massesdesc]:
#         df = df.join(features.set_index("flight_id"),on="flight_id",how="left")
#     return df
