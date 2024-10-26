from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class TargetIdentity:
    '''
    Not used in final submission
    Does not scale target
    '''
    def __init__(self, yvar):
        self.yvar = yvar
    def fit(self, df):
        return self
    def transform(self, df):
        return df[self.yvar].values.copy(),None
    def inverse_transform(self, df, yrawpred):
        return yrawpred.copy()


class TargetStandardScaler(TargetIdentity):
    '''
    Not used in final submission
    Standardscaler on target
    '''
    def __init__(self, yvar):
        super().__init__(yvar)
        self.yscaler = StandardScaler()
    def fit(self, df):
        self.yscaler.fit(df[[self.yvar]])
        return self
    def transform(self, df):
        return self.yscaler.transform(df[[self.yvar]])[:,0],None
    def inverse_transform(self, df, yrawpred):
        return self.yscaler.inverse_transform(yrawpred[:,None])[:,0]


class TargetScaleByGroup(TargetIdentity):
    '''
    Scale target by group, typically aircraft_type
    '''
    def __init__(self, yvar, by):
        super().__init__(yvar)
        self.by = by
        self.yvar = yvar
        self.dmean = {}
        self.dscale = {}
    def transform(self, df):#aircraft_type, tow):
        res = np.empty_like(df[self.yvar].values)
        res[:]=np.nan
        w = res.copy()
        for val in df[self.by].unique():
            mask = df[self.by].values == val
            y = (df[self.yvar].values[mask]-self.dmean[val]) / self.dscale[val]
            res[mask]=y
            w[mask] = self.dscale[val]
        return res, w**2
    def inverse_transform(self, df, yrawpred):
        res = np.empty_like(yrawpred)
        res[:]=np.nan
        for val in df[self.by].unique():
            mask = df[self.by].values == val
            res[mask] = yrawpred[mask] * self.dscale[val] + self.dmean[val]
        return res

class MassOewMtow(TargetScaleByGroup):
    '''
    !!!! Used in final submission !!!
    Scale target according documented MTOW and EOW
    '''
    def __init__(self, yvar, aircraft_type):
        super().__init__(yvar,aircraft_type)
        self.dmass = pd.read_csv("aircraft_type_masses.csv")
        for _,line in self.dmass.iterrows():
            self.dmean[line[self.by]] = line.oew
            self.dscale[line[self.by]] = line.mtow - line.oew

class MassStandardScalerByAircraft(TargetScaleByGroup):
    '''
    Not used in final submission
    Scale target by group
    '''
    def fit(self, df):
        dmean = {}
        dscale = {}
        for val in df[self.by].unique():
            mask = df[self.by].values == val
            assert(mask.sum()>10)
            y = df[self.yvar].values[mask]
            dmean[val] = np.mean(y)
            dscale[val] = np.std(y)
        self.dmean = dmean
        self.dscale = dscale


class LearnMassStandardScalerByAircraft(TargetScaleByGroup):
    '''
    Not used in final submission
    Standardcaler target by group, but if not enough data, use regression model relating
    documented MTOW and EOW to the observed mean and standard deviation
    '''
    def fit(self, df):
        dmean = {}
        dscale = {}
        for val in df[self.by].unique():
            mask = df[self.by].values == val
            if mask.sum()>10:
                y = df[self.yvar].values[mask]
                dmean[val] = np.mean(y)
                dscale[val] = np.std(y)
        massdf = pd.read_csv("aircraft_type_masses.csv")[["aircraft_type","mtow","oew"]].set_index("aircraft_type")
        keys = list(dmean.keys())
        Xb = massdf.loc[keys].values
        X = Xb[:,:1]-Xb[:,1:]
        print(X.shape)
        ymean = np.array([dmean[k] for k in keys])
        yscale = np.array([dscale[k] for k in keys])
        modelmean = make_pipeline(PolynomialFeatures(2),linear_model.LinearRegression()).fit(X,ymean-Xb[:,1])
        modelscale = make_pipeline(PolynomialFeatures(2),linear_model.LinearRegression()).fit(X,yscale)
        del Xb
        del X
        for k in massdf.index:
            if k not in dmean:
                # print(k)
                Xb = massdf.loc[[k]].values
                # print(Xb.shape)
                X = Xb[:,:1]-Xb[:,1:]
                dmean[k]=(Xb[:,1]+modelmean.predict(X))[0]
                dscale[k]=modelscale.predict(X)[0]
                print(dscale[k])
                assert(dscale[k]>0)
                assert(dmean[k]>0)
        self.dmean = dmean
        self.dscale = dscale


class GroupByTransformer(BaseEstimator, TransformerMixin):
    '''
    Not used in final submission
    Standardcaler features by group, typically aircraft_type
    but if not enough data, use aircraft_type synonym
    '''
    def __init__(self, transformer,synonym, by=None):
        self.dtransformer = dict()
        self.transformer = transformer
        self.by = by
        self.synonym = synonym

    def fit(self, X, y = None):
        df = X
        self.cols = [v for v in  list(df) if v!=self.by]
        lnotdone = []
        for val in X[self.by].unique():
            print(val)
            mask = X[self.by]==val
            X_sub = X.loc[mask, self.cols]
            if X_sub.shape[0] < 10:
                lnotdone.append(val)
            self.dtransformer[val] = self.transformer().fit(X_sub)
        print(lnotdone)
        for k in lnotdone:
            assert(k in self.synonym)
        for k in self.synonym.keys():
            self.dtransformer[k] = self.dtransformer[self.synonym[k]]
        return self

    def transform(self, X, y = None):
        X = X.copy()[self.cols+[self.by]]
        for val in X[self.by].unique():
            mask = X[self.by]==val
            transformed = self.dtransformer[val].transform(X.loc[mask, self.cols])
            X.loc[mask, self.cols] = transformed
        return X.loc[:, self.cols]
    def inverse_transform(self, X, y = None):
        X = X.copy()[self.cols+[self.by]]
        for val in X[self.by].unique():
            mask = X[self.by]==val
            transformed = self.dtransformer[val].inverse_transform(X.loc[mask, self.cols])
            X.loc[mask, self.cols] = transformed
        return X.loc[:, self.cols]
    def get_feature_names_out(self,names):
        return [v for v in names if v!=self.by]
