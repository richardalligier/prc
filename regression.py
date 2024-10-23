from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import linear_model
import numpy as np
import os
from os import walk
from sklearn.metrics import root_mean_squared_error
import lightgbm
from lightgbm import LGBMRegressor
import utils
import readers
from sklearnutils import GroupByTransformer,MassOewMtow,TargetIdentity,TargetStandardScaler,MassStandardScalerByAircraft,LearnMassStandardScalerByAircraft#,NoNormalizeMass
from add_localtime import add_localtime
from sklego.preprocessing import IdentityTransformer
import features
import argparse
import time
import itertools
import tqdm
from argparse import Namespace

YVAR = "tow"

class Predictor:
    def __init__(self, norm_mass,model_params):
        self.norm_mass = norm_mass
        self.model_params=model_params
        self.log = {}
    def fit(self,feat):
        df = feat.data
        print(list(df))
        print(len(list(df)))
        categorical_features, numeric_features, preprocessor = preprocessor_flights(feat)
        model = LGBMRegressor(**self.model_params)
        clf = Pipeline([
            ("preprocessor",preprocessor),
            ("model",model)
        ])
        print(model.get_params())
        vx = categorical_features + numeric_features
        X = df[vx]
        self.norm_mass.fit(df)
        Y,w = self.norm_mass.transform(df)
        assert (len(Y.shape)==1)
        start_time = time.time()
        self.clf = clf.fit(X, Y,model__sample_weight=w,model__feature_name=vx,model__categorical_feature=categorical_features)
        self.log["elapsed_time"] = time.time() - start_time
        print(self.log["elapsed_time"])
        self.log["model_params"] = self.clf.named_steps["model"].get_params()
        return self

    def predict(self,df,**kwargs):
        rawypred = self.rawpredict(df,**kwargs)
        return self.norm_mass.inverse_transform(df, rawypred)
    def rawpredict(self,df,**kwargs):
        return self.clf.predict(df[self.clf.feature_names_in_],**kwargs)
    def valid_curve(self,dftest,step=2000):
        num_iterations = self.clf.named_steps["model"].n_estimators_
        rawypred = 0
        lloss = []
        for i in range(0,num_iterations,step):
            actual_step = min(step,num_iterations-i)
            rawypred += self.rawpredict(dftest,start_iteration=i,num_iteration=actual_step)
            lloss.append((i+actual_step, root_mean_squared_error(dftest[YVAR].values,self.norm_mass.inverse_transform(dftest,rawypred))))
        return lloss
    def feature_importances(self):
        return sorted(zip(self.clf.named_steps["model"].feature_importances_,self.clf.named_steps["model"].feature_name_), reverse=True)


def assemble_sub(dfsub,ysub):
    assert(not np.isnan(ysub).any())
    dfsub["tow"]=ysub
    return dfsub[["flight_id","tow"]]


def preprocessor_flights(feat):
    numeric_features = sorted(feat.numeric_features())
    numeric_features_not_scaled = sorted(feat.numeric_features_not_scaled())
    categorical_features = sorted(feat.categorical_features())
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1,dtype=np.int32)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", IdentityTransformer()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
            ("unscalednumeric", IdentityTransformer(), numeric_features_not_scaled),
        ],
        remainder = "drop",
    )
    return categorical_features, numeric_features+numeric_features_not_scaled, preprocessor


def train_valid(random_state):
    feat = features.read_features("challenge_set")
    dftrain, dftest = train_test_split(feat.data,test_size=0.2,random_state=random_state)
    feat.data = dftrain
    return feat, lambda : dftest

def train_test(random_state):
    feat = features.read_features("challenge_set")
    return feat, lambda : features.read_features("final_submission_set").data


def valid(predictor, dftest):
    test_rmse=predictor.valid_curve(dftest)
    best_test_rmse=min(test_rmse,key= lambda x:x[1])
    print(test_rmse)
    print(best_test_rmse)
    ypred = predictor.predict(dftest)
    loss = root_mean_squared_error(dftest[YVAR].values,ypred)
    for imp,v in predictor.feature_importances():
        print(f"{v:30} {imp:30}")
    predictor.log["test_error"]=loss
    print(f"test error {loss}")
    for ac in sorted(dftest.aircraft_type.unique()):
        mask = dftest.aircraft_type.values == ac
        loss = root_mean_squared_error(dftest[YVAR].values[mask],ypred[mask])
        print(f"test error |{ac:<10}|{mask.sum():>10}|{loss:>10.0f}|")
    for month in sorted(dftest.date.dt.month.unique()):
        mask = dftest.date.dt.month == month
        loss = root_mean_squared_error(dftest[YVAR].values[mask],ypred[mask])
        print(f"test error |{month:<10}|{mask.sum():>10}|{loss:>10.0f}|")

def submit(predictor, dftest, config):
    ysub = predictor.predict(dftest)
    dfsubpred = assemble_sub(dftest, ysub)
    utils.write_submission(config, dfsubpred)

def convert(v):
    try:
        return int(v)
    except:
        try:
            return float(v)
        except:
            return v

def main():
    parser = argparse.ArgumentParser(
                    prog='predictor',
                    description='',
    )
    parser.add_argument("-what",default="valid")
    parser.add_argument("-log",default="log_file")
    parser.add_argument("-random_state",default=0,type=int)
    args, unknown = parser.parse_known_args()
    unknown = {k[1:]:convert(v) for k,v in zip(unknown[:-1:2],unknown[1::2])}
    print(unknown)
    log = {}
    log["args"]=args
    dsplitter = {
        "valid":train_valid,
        "submit":train_test,
    }
    config = utils.read_config()
    feat, fdftest = dsplitter[args.what](args.random_state)
    print(args)
    model_params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.7368753029236468, 'importance_type': 'gain', 'learning_rate': 0.04347089639866006, 'max_depth': -1, 'min_child_samples': 460, 'min_child_weight': 0.0, 'min_split_gain': 0.0, 'n_estimators': 50000, 'n_jobs': None, 'num_leaves': 14, 'objective': None, 'random_state': args.random_state, 'reg_alpha': 400, 'reg_lambda': 500, 'subsample': 0.879814641620981, 'subsample_for_bin': 400000, 'subsample_freq': 100, 'device': 'cpu', 'verbose': 1, 'max_bin': 511, 'cat_l2': 450, 'cat_smooth': 150}
       # model_params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.9983479969770903, 'importance_type': 'gain', 'learning_rate': 0.034044216343711194, 'max_depth': -1, 'min_child_samples': 360, 'min_child_weight': 0.0, 'min_split_gain': 0.0, 'n_estimators': 50000, 'n_jobs': None, 'num_leaves': 17, 'objective': None, 'random_state': 0, 'reg_alpha': 300, 'reg_lambda': 800, 'subsample': 0.9708398642260044, 'subsample_for_bin': 400000, 'subsample_freq': 100, 'device': 'cpu', 'verbose': 1, 'max_bin': 1023, 'cat_l2': 50, 'cat_smooth': 150}#, 'test_error': 1572.2932228357363}
    for k,v in unknown.items():
        if k in model_params:
            model_params[k]=v
        else:
            raise Exception(f"unknown args '-{k} {v}'")
    predictor = Predictor(MassOewMtow(YVAR,"aircraft_type"),model_params)
    predictor.fit(feat)
    ypred = predictor.predict(feat.data)
    print("training error",root_mean_squared_error(feat.data[YVAR],ypred))
    if args.what == "valid":
        valid(predictor,fdftest())
        with open(args.log,"a") as f:
            loglog = {** log, **predictor.log}
            f.write(f"{loglog},")
    elif args.what == "submit":
        submit(predictor,fdftest(),config)
    print({**log, **predictor.log})

if __name__ == '__main__':
    main()

