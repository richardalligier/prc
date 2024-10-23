import pandas as pd
from timezonefinder import TimezoneFinder
import numpy as np
import argparse
import readers

parser = argparse.ArgumentParser(
    prog='add_timezone',
    description='add_timezone',
)
parser.add_argument("-a_in")
parser.add_argument("-a_out")
parser.add_argument("-flights")
args = parser.parse_args()

af = pd.read_csv(args.a_in)
af = af.query("ident!='EGSY'") # drop shefield city heliport
af = af.astype({x:"string" for x in ['ident', 'type', 'name', 'continent', 'iso_country', 'iso_region', 'municipality', 'scheduled_service', 'gps_code', 'iata_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords']})
for x in ['latitude_deg', 'longitude_deg', 'elevation_ft']:
    af[x]=af[x].astype(np.float64)

usedairports = set()
for fname in args.flights.split():
    f = readers.read_flights(fname)
    usedairports = usedairports.union(set(f.ades.values))
    usedairports = usedairports.union(set(f.adep.values))
af = af.query("gps_code.isin(@usedairports) or ident.isin(@usedairports)")
icao_code =[i if i in usedairports else g for g,i in zip(af.gps_code.values,af.ident.values)]
af["icao_code"] = icao_code
af["icao_code"]=af["icao_code"].astype("string")
af = af.drop(columns="ident")
assert (len(usedairports) == af.shape[0])
assert (af.icao_code.nunique()== af.shape[0])



tf = TimezoneFinder()

ltz = []

for i,line in af.iterrows():
    # print(i,line)
    tz = tf.timezone_at(lng=line.longitude_deg,lat=line.latitude_deg)
    ltz.append(str(tz))
af["time_zone"] = ltz
af["time_zone"]=af["time_zone"].astype("string")
print(af["time_zone"])
print(af.dtypes)

af.to_parquet(args.a_out,index=False)

