import pandas as pd
import numpy as np
import utils
import os


def add_features_trajectories(df):
    '''
    add features on trajectories data
    '''
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
    '''
    used to factorize code
    '''
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
    ''' read and convert trajectories to SI units
    '''
    df = pd.read_parquet(f)
    for v in ["flight_id", "icao24"]:
        df[v] = df[v].astype(np.int64)
    df = convert_to_SI(df)
    return df#add_features_trajectories(df)
def add_features_flight(df):
    return df.assign(
        dayofweek = df.actual_offblock_time.dt.isocalendar().day,
        weekofyear = df.actual_offblock_time.dt.isocalendar().week,
        arrival_minutes = df.arrival_time.dt.hour*60+df.arrival_time.dt.minute,
        actual_offblock_minutes = df.actual_offblock_time.dt.hour*60+df.actual_offblock_time.dt.minute,
        )

def read_flights(f):
    '''
    read flights from parquet file
    '''
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
    df = pd.read_parquet(f)
    for dtype,lvar in ltypes:
        for v in lvar:
            df[v]=df[v].astype(dtype)
    return add_features_flight(df)

