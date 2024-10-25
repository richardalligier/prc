import argparse
import pandas as pd
import os
import utils
import numpy as np

def average_prediction(istart,istop):
    config = utils.read_config()
    fname = os.path.join(config.SUBMISSIONS_FOLDER,utils.get_submission_name(config))
    l = [pd.read_csv(fname.format(version=i)) for i in range(istart,istop)]
    for i in range(1,istop-istart):
        assert((l[0].flight_id.values == l[i].flight_id.values).all())
    tows = np.array([li.tow.values for li in l])
    results = pd.DataFrame({"flight_id":l[0].flight_id.values,"tow":np.mean(tows,axis=0)})
    print("results.shape",results.shape)
    return results


def main():
    import readers
    parser = argparse.ArgumentParser(
                    description='compute features related to cruise phase by computing stats on slices aligned on time',
    )
    parser.add_argument("-istart",default=0,type=int)
    parser.add_argument("-istop",type=int)
    parser.add_argument("-out_csv")
    args = parser.parse_args()
    res = average_prediction(args.istart,args.istop)
    res.to_csv(args.out_csv,index=False)
if __name__ == '__main__':
    main()
