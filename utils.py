import os
import pandas as pd
import numpy as np
from collections import namedtuple

FEET2METER = 0.3048
NM2METER = 1852.
KTS2MS = NM2METER/3600.

def read_config(fname="CONFIG"):
    ''' read config to get folders '''
    with open(fname,"r") as f:
        l = [[x.strip() for x in line.split("=")] for line in f if len(line.strip())>0 and line.strip()[0]!="#"]
        d = {k:v for k,v in l}
    for subfolder in ["trajectories","flights","flightsparquet","METARs"]:
        d[subfolder]=os.path.join(d["FOLDER_DATA"],subfolder)
    Config = namedtuple("config",list(d))
    return Config(**d)

def identifyargs(args):
    names = ["what","train","submit","variables","random_state","scaler"]
    return [str(getattr(args,x)) for x in names]

def write_submission(config,df,args):
    ''' write submission in the dedicated folder with the right version number '''
    sub_folder = args.folderout
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    fname = "-".join(identifyargs(args)) + "-predictions.csv"
    df.to_csv(os.path.join(sub_folder,fname),index=False)




if __name__ == '__main__':
    print(read_config())
