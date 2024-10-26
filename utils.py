import os
import pandas as pd
import numpy as np
from collections import namedtuple

FEET2METER = 0.3048
NM2METER = 1852.
KTS2MS = NM2METER/3600.

def read_config():
    ''' read config to get folders '''
    with open("CONFIG","r") as f:
        l = [[x.strip() for x in line.split("=")] for line in f if len(line.strip())>0 and line.strip()[0]!="#"]
        d = {k:v for k,v in l}
    for subfolder in ["trajectories","flights","flightsparquet","METARs"]:
        d[subfolder]=os.path.join(d["FOLDER_DATA"],subfolder)
    Config = namedtuple("config",list(d))
    return Config(**d)


def read_json(fname):
    ''' read json file, useful to make submission '''
    if os.path.exists(fname):
        with open(fname,'r') as f:
            return eval("\n".join(f.readlines()),{"false":False,"true":True})
    else:
        print(f"WARNING: could not find {fname}, using empty json")
        return {"team_name":"__","team_id":""}


def get_version_number(submission_folder):
    ''' get next version looking at files in the submission folder '''
    imax = -1
    for (dirpath, dirnames, filenames) in os.walk(submission_folder):
        for fname in filenames:
            v = fname.split("_")[3][1:]
            imax = max(imax,int(v))
    return imax + 1



def get_submission_name(config):
    ''' get submission filename '''
    json = read_json(config.JSON)
    team_name = json["team_name"]
    team_id = json["team_id"]
    return f"{team_name}_"+"v{version}"+f"_{team_id}.csv"


def write_submission(config,df):
    ''' write submission in the dedicated folder with the right version number '''
    sub_folder = config.SUBMISSIONS_FOLDER
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    version = get_version_number(sub_folder)
    fname = get_submission_name(config).format(version=version)#f"{team_name}_v{version}_{team_id}.csv"
    df.to_csv(os.path.join(sub_folder,fname),index=False)




if __name__ == '__main__':
    print(read_config())
