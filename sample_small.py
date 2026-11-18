import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import argparse

PATH_SOURCE = "/disk4/prc"
FRAC = 0.01
SEED=42




def main():
    parser = argparse.ArgumentParser(
                        prog='sampler of the PRC2024 dataset',
                        description='',
        )
    parser.add_argument("-path_source",default="/disk4/prc")
    parser.add_argument("-path_dest",default=None)
    parser.add_argument("-frac",default=0.01)
    parser.add_argument("-seed",default=42)
    args = parser.parse_args()
    if args.path_dest is None:
        path_dest = f"/disk4/prcsmall_{args.frac}_{args.seed}"
    else:
        path_dest = args.path_dest
    d={}
    Path(f"{path_dest}/flights").mkdir(parents=True, exist_ok=True)
    for file in os.listdir(f"{PATH_SOURCE}/flights"):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename!="test_final.csv":
            d[filename] = pd.read_csv(f"{PATH_SOURCE}/flights/{filename}")
    FL_LIST = "flight_list.csv" 
    ddest={FL_LIST:d[FL_LIST].sample(frac=FRAC,random_state=SEED).reset_index(drop=True)}
    fid = ddest[FL_LIST].flight_id
    for k in d:
        if k!=FL_LIST:
            ddest[k]=d[k].query('flight_id in @fid').reset_index(drop=True)
    for k in ddest:
        print(k,ddest[k].shape)
        print(ddest[k].groupby("aircraft_type").count())
        ddest[k].to_csv(f"{path_dest}/flights/{k}",index=False)
        #print(d[k].info())
    #raise Exception
    for file in tqdm.tqdm(os.listdir(f"{PATH_SOURCE}/rawtrajectories")):
        if file.endswith(".parquet"):
            df = pd.read_parquet(f"{PATH_SOURCE}/rawtrajectories/{file}")
            path=f"{path_dest}/rawtrajectories"
            Path(path).mkdir(parents=True, exist_ok=True)
            df.query('flight_id in @fid').reset_index(drop=True).to_parquet(f"{path}/{file}")


main()