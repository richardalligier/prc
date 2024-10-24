import pandas as pd
from timezonefinder import TimezoneFinder

def add_localtime(af,df,tadep,tades):
    l = [(tadep,"adep"),
     (tades,"ades")]
    for (t,a) in l:
        df[t] = pd.to_datetime(df[t])
        df = df.join(af[["icao_code","time_zone"]].set_index("icao_code"),on=a)
        df = df.rename(columns={"time_zone":"time_zone_"+a})
        df[f"local_{t}"]=(pd.to_datetime(df.groupby(f'time_zone_{a}')[f'{t}']
                                         .transform(lambda x: x.dt.tz_convert(x.name).dt.tz_localize(tz=None))
        )
        )
    return df
