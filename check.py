import pandas as pd
import numpy as np
import os


what = "final_submission_set"
folderref = "/disk3/refprc"
folder = "/disk3/prc"

def replace(name):
    return folder+name[len(folderref):]

def compare(a,b):
    print(a)
    af=pd.read_parquet(a)
    bf=pd.read_parquet(b)
    sa = set(af)
    sb = set(bf)
    print(sa-sb,sb-sa)
    names = list(sa.intersection(sb))
    # print(af["flight_id"])
    # print(bf["flight_id"])
#    print((bf["flight_id"]==af["flight_id"]).all())
    print((af[names].isna().sum()-bf[names].isna().sum()))
#    print(af.Cruisemach_7.isna().sum(),bf.Cruisemach_7.isna().sum())
    # print(af.Cruisemach_1.isna().sum(),bf.Cruisemach_1.isna().sum())
    print((af[names].select_dtypes(include=[np.number])-af[names].select_dtypes(include=[np.number])).abs().max(axis=None))
    mask = np.logical_not(af[names].isna().values)
    print(mask.shape)
    assert((af[names].isna()==bf[names].isna()).all(axis=None))
    assert((af[names].values[mask]==bf[names].values[mask]).all())


#compare("/disk3/refprc/METARs.parquet","/disk3/refprc/METARsref.parquet")
#compare("/disk3/prc/METARs (copy).parquet","/disk3/prc/METARs.parquet")
#compare("/disk3/prc/thunder1024/final_submission_set.parquet","/disk3/prc/thunder/final_submission_set.parquet")
#compare("/disk3/prc/weather1024/final_submission_set.parquet","/disk3/prc/weather/final_submission_set.parquet")
compare("/disk3/prc/airports_tz.parquet","/home/alligier/workInProgress/githubprc/prc/toto.parquet")
#for f in ["classic__1e-2__20_cruise/final_submission_set","classic__1e-2__5_500_40_daltitude_1_-0.5_1_masses/final_submission_set","classic__1e-2_wind/final_submission_set"]:#"classic__1e-2_interpolated_trajectories"]:#"classic_filtered_trajectories"]:
#    compare(f"/disk3/refprc/{f}/2022-01-01.parquet",f"/disk3/prc/{f}/2022-01-01.parquet")

# for root, dirs, files in os.walk(folderref):
#     for name in files:
#         fname = os.path.join(root, name)
#         if "mass" in fname and "challenge_set" not in fname and fname.endswith(".parquet"):
#             compare(fname,replace(fname))
