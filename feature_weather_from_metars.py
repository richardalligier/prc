import argparse
import pandas as pd
import pitot
import readers
from sklearn.neighbors import NearestNeighbors
import numpy as np

class FitWeather:
    """
    Fit a model that allows efficient spatio-temporal queries on a weather dataframe of spatio-temporal measurements
    """
    def __init__(self,geo_scale,hour_scale):
        self.geo_scale = geo_scale
        self.hour_scale = hour_scale
    def scale(self,df):
        vlat,vlon,vtime = self.vpostime
        for v in [vlon,vlat]:
            df[v]=df[v]/self.geo_scale
        df[vtime] = (df[vtime]-self.timeref) / np.timedelta64(1, 'h') / self.hour_scale
        return df
    def fit(self,df,vpostime,radius=1,n_neighbors=5):
        self.radius=radius
        self.n_neighbors=n_neighbors
        self.vpostime = vpostime
        vlat,vlon,vtime = vpostime
        df = df[[vlat,vlon,vtime]].copy()
        self.timeref = df[vtime].min()
        self.model = NearestNeighbors(n_neighbors=n_neighbors,radius=radius)
        self.model.fit(self.scale(df))
        return self
    def radius_neighbors(self,df,vpostime,radius=None):
        radius = self.radius if radius is None else radius
        xdf = df[list(vpostime)].copy().rename(columns={k:v for k,v in zip(vpostime,self.vpostime)})
        return self.model.radius_neighbors(self.scale(xdf),radius)
    def kneighbors(self,df,vpostime,n_neighbors=None):
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        xdf = df[list(vpostime)].copy().rename(columns={k:v for k,v in zip(vpostime,self.vpostime)})
        return self.model.kneighbors(self.scale(xdf),n_neighbors)

def feature_weather(flights,airports,metars,lvar,geo_scale,hour_scale):
    '''
    Compute the weather variables @lvar for the @flights using the @metars closest to the airport at the datetime of interest.
    The ratio between the temporal and spatial scales are specified by @geo_scale and @hour_scale
    '''
    df = flights.copy()#["flight_id"]
    for airport in ["adep","ades"]:
        df= pd.merge(df,airports[["icao_code","latitude_deg","longitude_deg"]],left_on=airport,right_on="icao_code",suffixes=('_adep','_ades'))
    df = df.reset_index()
    postimeMETARs = ("lat","lon","valid")
    model = FitWeather(geo_scale=geo_scale,hour_scale=hour_scale).fit(metars,postimeMETARs)
    postime_ades = ("latitude_deg_ades","longitude_deg_ades","arrival_time")
    postime_adep = ("latitude_deg_adep","longitude_deg_adep","actual_offblock_time")
    res = df[["flight_id"]].set_index("flight_id")
    for apt,postime in (("adep",postime_adep),("ades",postime_ades)):
        dist,index = model.kneighbors(df,postime,1)
        weather = metars.iloc[index[:,0]].drop(columns=list(postimeMETARs)).reset_index(drop=True)
        weather = weather.rename(columns={v:f"{v}_{apt}" for v in lvar})
        weather["flight_id"]=res.index
        res = res.join(weather.set_index("flight_id"),on="flight_id",how="left")
    return res.reset_index()

def main():
    import readers
    parser = argparse.ArgumentParser(
                    description='compute weather features at adep and ades',
    )
    parser.add_argument("-f_in")
    parser.add_argument("-airports")
    parser.add_argument("-metars")
    parser.add_argument("-f_out")
    parser.add_argument("-geo_scale",type=float)
    parser.add_argument("-hour_scale",type=float)
    args = parser.parse_args()
    lvar = ["mslp","tmpf","drct","sknt","gust","wxcodes","vsby","skyl1","skyl2","skyl3","skyl4","skyc1","skyc2","skyc3","skyc4","alti","relh"]
    metars = pd.read_parquet(args.metars)[["lon","lat","valid"]+lvar]
    airports = pd.read_parquet(args.airports)
    flights = readers.read_flights(args.f_in)
    dfadded = add_weather(flights,airports,metars,lvar,args.geo_scale,args.hour_scale)
    return dfadded.to_parquet(args.f_out,index=False)
if __name__ == '__main__':
    main()
