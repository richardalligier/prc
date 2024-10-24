import warnings
import pitot.geodesy
import utils
import pandas as pd

def joincdates(flights,airports,trajs,rod=-1000,roc=1000):
    ''' Compute corrected dates and fill with original dates if cannot be corrected '''
    cdates=fillnadates(flights,compute_dates(flights,airports,trajs,rod=rod,roc=roc))
    flights =flights.join(cdates.set_index("flight_id"),on="flight_id",how="inner")
    return flights

def compute_dates(flights,airports,trajs,rod=-1000,roc=1000):
    ''' Compute corrected dates'''
    dfjoined = trajs.join(
        flights.set_index("flight_id"),
        on="flight_id",
        how="inner",
        validate="many_to_one")
    res = dfjoined[["flight_id"]].drop_duplicates()
    for apt in ["adep","ades"]:
        adf = dfjoined.join(
            airports[["icao_code","latitude_deg","longitude_deg"]].set_index("icao_code"),
            on=apt,
            how="left").reset_index()
        adf[f"distance_{apt}"] = pitot.geodesy.distance(
            adf.latitude,
            adf.longitude,
            adf.latitude_deg,
            adf.longitude_deg)/utils.NM2METER
        mindist = adf.query(f"distance_{apt}<10").groupby("flight_id")
        idxdistmin = mindist["timestamp"].idxmin() if apt == "ades" else mindist["timestamp"].idxmax()
        xtime=adf.loc[idxdistmin,["flight_id","timestamp","altitude"]]
        rocd = rod if apt =="ades" else roc
        timedelta = ((xtime.altitude/rocd*60)*1e9).round() # nanosecond rounding
        # feed timedelta as nanoseconds, otherwise,
        # random overflow error from one run to another on the exact same data with the exact same code...
        warnings.simplefilter("error")
        xtime["timestamp"]=xtime["timestamp"]-pd.to_timedelta(timedelta,unit="nanoseconds")
        warnings.simplefilter("default")
        x = xtime[["flight_id","timestamp"]].rename(columns={"timestamp":f"t_{apt}"}).set_index("flight_id")
        res = res.join(x,on="flight_id",how="left")
    return res

def fillnadates(flights,cdates):
    ''' fill with original dates if cannot be corrected '''
    df = flights[["flight_id","arrival_time"]].assign(
        t_adep=flights.actual_offblock_time + pd.to_timedelta(flights.taxiout_time,unit="minutes")
    ).rename(columns={"arrival_time":"t_ades"})
    df = df.set_index("flight_id").loc[cdates.flight_id]
    return cdates.set_index("flight_id").combine_first(df).reset_index()
