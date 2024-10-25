import pandas as pd
from timezonefinder import TimezoneFinder

def add_localtime(airports,flights,tadep,tades):
    '''adds local_time date at adep and ades computed using the @airports, the utc times @tadep @tades in @flights '''
    l = [(tadep,"adep"),
     (tades,"ades")]
    for (t,a) in l:
        flights[t] = pd.to_datetime(flights[t])
        flights = flights.join(airports[["icao_code","time_zone"]].set_index("icao_code"),on=a)
        flights = flights.rename(columns={"time_zone":"time_zone_"+a})
        flights[f"local_{t}"]=(pd.to_datetime(flights.groupby(f'time_zone_{a}')[f'{t}']
                                         .transform(lambda x: x.dt.tz_convert(x.name).dt.tz_localize(tz=None))
        )
        )
    return flights
