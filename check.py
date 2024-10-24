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


compare("/disk3/refprc/METARs.parquet","/disk3/refprc/METARsref.parquet")

# for root, dirs, files in os.walk(folderref):
#     for name in files:
#         fname = os.path.join(root, name)
#         if "mass" in fname and "challenge_set" not in fname and fname.endswith(".parquet"):
#             compare(fname,replace(fname))