import readers
import pandas as pd
from functools import wraps
import utils
import os
from pitot import geodesy
import numpy as np
from add_localtime import add_localtime

FID = "flight_id"


# def read_flights(f):
#     dates = ["date","actual_offblock_time","arrival_time"]
#     ltypes = [
#         ("string", ["adep","ades"]),#4
#         ("string", ["callsign","airline"]),#32
#         ("string",["wtc"]),#1
#         ("string",["country_code_ades","country_code_adep"]),#2
#         ("string",["aircraft_type"]),#4
#         (np.float64, ["flight_duration","taxiout_time","flown_distance","tow"]),
#         (np.int64, ["flight_id"]),
#         ("string", dates),
#     ]
#     usecols = [v for _,lvar in ltypes for v in lvar]
#     dtype = {v:t for t,lvar in ltypes for v in lvar}
#     df = pd.read_csv(f,dtype=dtype,parse_dates=dates,usecols=usecols)
#     df = add_features_flights(df)
#     print(df.dtypes)
#     return df


def sortedout(method):
    @wraps(method)
    def _sortedout(self):
        # print(method(self))
        return sorted(method(self))
    return _sortedout


class AllNumeric:
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.data = pd.read_parquet(fname)
        # print("split")
        # self.data = self.data.query("date.dt.month<=1")
        # print("done")
    @sortedout
    def categorical_features(self):
        return []
    @sortedout
    def numeric_features_not_scaled(self):
        return []
    @sortedout
    def numeric_features(self):
        return sorted([x for x in self.data if x!=FID])


class AllNumericNotScaled:
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.data = pd.read_parquet(fname)
        # print("split")
        # self.data = self.data.query("date.dt.month<=1")
        # print("done")
    @sortedout
    def categorical_features(self):
        return []
    @sortedout
    def numeric_features_not_scaled(self):
        return sorted([x for x in self.data if x!=FID])
    @sortedout
    def numeric_features(self):
        return []

class AllCategorical:
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.data = pd.read_parquet(fname)
    @sortedout
    def categorical_features(self):
        return sorted([x for x in self.data if x!=FID])
    @sortedout
    def numeric_features_not_scaled(self):
        return []
    @sortedout
    def numeric_features(self):
        return []

def round_to_hour(time):
    return (time.dt.hour+time.dt.minute/60).round().astype(np.int32)

class Flights:
    def __init__(self, config, what):
        fname = os.path.join(config.flights,f"{what}.parquet")
        self.fname = fname
        df = readers.read_flights(fname)#.query("date.dt.month<=1").query("date.dt.day<=1")
        print(f"{df.shape=}")
        airports = pd.read_parquet(os.path.join(config.FOLDER_DATA,"airports_tz.parquet"))
        df=add_localtime(airports,df,"actual_offblock_time","arrival_time")
        df["local_hour_adep"]=round_to_hour(df["local_actual_offblock_time"])
        df["local_hour_ades"]=round_to_hour(df["local_arrival_time"])
        # print(df[["local_arrival_time","local_actual_offblock_time"]])
        # print(df[["local_hour_ades","local_hour_adep"]])
        # raise Exception
        # metars = pd.read_parquet(config.METARs+".parquet")
        # # print(df.dtypes)
        # # print(airports.dtypes)
        # # raise Exception
        # # print(metars.dtypes)
        # # print(airports.dtypes)
        # for (tairport,airport) in [("actual_offblock_time","adep"),("arrival_time","ades")]:
        #     df = pd.merge_asof(df.sort_values(tairport),metars,left_on=tairport,right_on="valid",left_by=airport,right_by="icao_code",right_index=False,left_index=False,direction="nearest",tolerance=pd.Timedelta("2hour"),suffixes=('_adep','_ades'))
        # df = df.drop(columns=["icao_code_adep","icao_code_ades"])
        # print(df.query("tmpf_ades.isna()")[["flight_id"]])
        weather = pd.read_parquet(os.path.join(config.FOLDER_DATA,"weather",f"{what}.parquet"))
        # print(weather)
        # print(list(weather))
        df = df.join(weather.set_index("flight_id"),on="flight_id",how="left")
        print(f"{df.shape=}")
        # for airport in ["adep","ades"]:
        #     df= pd.merge(df,airports[["icao_code","latitude_deg","longitude_deg","elevation_ft"]],left_on=airport,right_on="icao_code",suffixes=('_adep','_ades'))
        for airport in ["adep","ades"]:
            df= pd.merge(df,airports[["icao_code","latitude_deg","longitude_deg","elevation_ft"]],how="left",left_on=airport,right_on="icao_code",suffixes=('_adep','_ades'))
        print(f"{df.shape=}")
        # print(1-df.latitude_deg_ades.isna().mean())
        # print(1-df.latitude_deg_adep.isna().mean())
        df["bird_distance"]=geodesy.distance(
            df.latitude_deg_adep.values,df.longitude_deg_adep.values,
            df.latitude_deg_ades.values,df.longitude_deg_ades.values,
        )/utils.NM2METER
 #       df["diffdist"]= df["flown_distance"]-df["bird_distance"]
#        print(df.query("diffdist<-10")[["flight_id","date","adep","ades","diffdist","bird_distance","flown_distance"]])
        # print(list(df))
        # raise Exception
        self.data=df
        # .assign(
        #     dayofweek = df.date.dt.isocalendar().day,
        #     weekofyear = df.date.dt.isocalendar().week,
        #     arrival_hour = df.arrival_time.dt.hour*60+df.arrival_time.dt.minute,
        #     actual_offblock_time_hour = df.actual_offblock_time.dt.hour*60+df.actual_offblock_time.dt.minute,
        # )
    @sortedout
    def categorical_features(self):
        return ["adep","ades","airline","aircraft_type","wtc","country_code_ades","country_code_adep","dayofweek","local_hour_ades","local_hour_adep"]#"callsign",
    @sortedout
    def numeric_features_not_scaled(self):
        aptvar = [f"{v}_{apt}" for apt in ["ades","adep"] for v in ["drct","tmpf","sknt","elevation_ft","vsby","latitude_deg","longitude_deg"]]#"alti","drct","sknt","tmpf"]]
        return ["weekofyear"] + aptvar
    @sortedout
    def numeric_features(self):
        #return ["arrival_minutes","actual_offblock_minutes"]#"flight_duration","taxiout_time","flown_distance","arrival_minutes","actual_offblock_minutes"]
        return ["flight_duration","taxiout_time","flown_distance","bird_distance"]


class Cruise(AllNumeric):
    def __init__(self,fname):
        super().__init__(fname)
        prefix = "CruiseDeltaAlt_"
        nb = max(int(x[len(prefix):])  for x in list(self.data) if x.startswith(prefix))+1
        print(nb)
        for i in range(nb):
            isclimb =  self.data[f"CruiseDeltaAlt_{i}"] > 200
            isdescent = self.data[f"CruiseDeltaAlt_{i}"] < -200
            iscount = self.data[f"Cruisemachcount_{i}"] < 100
            self.data[f"CruiseMedianAlt_{i}"]= ((self.data[f"CruiseMedianAlt_{i}"]/1000).round())*1000
            for v in [f"CruiseMedianAlt_{i}",f"Cruisemach_{i}"]:
                self.data.loc[isclimb,v]= -10000
                self.data.loc[isdescent,v]= -20000
                self.data.loc[iscount,v]= np.nan
            print(self.data[f"CruiseMedianAlt_{i}"].unique())

class Mass(AllNumeric):
    def __init__(self,norange,scale,fname):
        super().__init__(fname)
        if "index" in list(self.data):
            self.data=self.data.drop(columns="index")
        if scale:
            for v in list(self.data):
                if "mass_" in v and v not in ["mass_min","mass_max"]:
                    print(f"scaling {v}")
                    self.data[v]=(self.data[v]-self.data["mass_min"])/(self.data["mass_max"]-self.data["mass_min"])
        if norange:
            self.data = self.data.drop(columns=["mass_min","mass_max"])


class Union:
    def __init__(self,flights,lfeatures):
        self.data = flights.data#.copy()
        for i,features in enumerate(lfeatures):
            print(features.fname)
            # if i==0:
            #     features.data=features.data.drop(columns=["mass_min","mass_max"])
            # if "index" in list(features.data):
            #     print("warning aircraft_type")
            #     features.data=features.data.drop(columns="index")
            self.data = self.data.join(features.data.set_index("flight_id"),on="flight_id",how="left")
        # print("split")
        # self.data = self.data.query("date.dt.month<=6")
        # print("done")
        self.lfeatures = lfeatures
        self.flights = flights
        # for v in [f"Nonemassadepdate_{i}" for i in range(9)]:
        #     self.data[v]=self.data[v].dt.seconds
        # for v in [f"Nonemassadesdate_{i}" for i in range(9)]:
        #     self.data[v]=self.data[v].dt.seconds
        # for v in [f"Nonemasscount_{i}" for i in range(9)]:
        #     self.data[v]=self.data[v].fillna(0)
        # print(self.data.Nonemassadepdate_0.dtypes)
        # raise Exception
        self.fname = (flights.fname,)+tuple(x.fname for x in lfeatures)
    @sortedout
    def categorical_features(self):
        return [y for x in self.lfeatures for y in x.categorical_features()] + self.flights.categorical_features()
    @sortedout
    def numeric_features_not_scaled(self):
        return [y for x in self.lfeatures for y in x.numeric_features_not_scaled()] + self.flights.numeric_features_not_scaled()
    @sortedout
    def numeric_features(self):
        return [y for x in self.lfeatures for y in  x.numeric_features()] + self.flights.numeric_features()

def read_features(what):
    config = utils.read_config()
    lfeatures = [
        # AllCategorical(os.path.join(config.FOLDER_DATA,f"icaos/{what}")),
        Mass(norange=True,scale=True,fname=os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_500_40_daltitude_1_-0.5_1_masses/{what}")),
#        Mass(norange=True,scale=True,fname=os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_1500_40_daltitude_0.3_-0.5_1_massesdesc/{what}")),
        AllNumericNotScaled(os.path.join(config.FOLDER_DATA,f"thunder/{what}.parquet")),
        AllNumericNotScaled(os.path.join(config.FOLDER_DATA,f"classic__1e-2_wind/{what}")),
#         AllNumeric(os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_500_1_masses/{what}")),

        # AllNumericNotScaled(os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_500_40_daltitude_0.3_-0.5_1_massesdesc/{what}")),
        Cruise(os.path.join(config.FOLDER_DATA,f"classic__1e-2__20_cruise/{what}")),
#        AllNumeric(os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_500_1_daltitude_masses/{what}")),
        #AllNumeric(os.path.join(config.FOLDER_DATA,f"classic__1e-2__5_500_40_vectorized_trajectories/{what}")),
    ]
 #   print(lfeatures[0].data["thunder_max_pVCTS_2_ades"].describe())
    # print(list(lfeatures[0].data))
    # print(len(list(lfeatures[0].data)))
    # raise Exception
    flights = Flights(config, what)
    feat = Union(flights, lfeatures)
    print(f"{feat.data.shape=}")
    feat.data=feat.data.query("aircraft_type!='C56X'").query("aircraft_type!='A310'")#.reset_index()
    print(f"{feat.data.shape=}")
    return feat
