import pandas as pd

import os
from os.path import isfile, join
import utils
import numpy as np
import readers
import argparse


def read_METARS(dirname):
    '''
    Read the metar folder and build a parquet file
    '''
    onlyfiles = [f for f in os.listdir(dirname) if isfile(join(dirname, f))]#[:1]
    dtype = {}
    for x in ["lon","lat","elevation","tmpf","dwpf","relh","drct","sknt","p01i","alti","mslp","vsby","gust","skyl1","skyl2","skyl3","skyl4","feel"]:
        dtype[x] = np.float64
    for x in ["station","skyc1","skyc2","skyc3","skyc4","wxcodes","metar"]:
        dtype[x]="string"
    df = pd.concat([pd.read_csv(join(dirname,f),comment="#",dtype=dtype).astype({"valid":"datetime64[ns, UTC]"}) for f in onlyfiles])
    return df.sort_values("valid")

def main():
    parser = argparse.ArgumentParser(
        description='build a parquet metar file from a folder of csv metars',
    )
    parser.add_argument("-metars_folder_in")
    parser.add_argument("-metars_parquet_out")
    args = parser.parse_args()
    metars = read_METARS(args.metars_folder_in)
    metars.to_parquet(args.metars_parquet_out)

if __name__ == '__main__':
    main()
