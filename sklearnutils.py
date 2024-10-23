from sklearn.base import BaseEstimator, TransformerMixin
# import sklego
# from sklego.preprocessing import IdentityTransformer
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class TargetIdentity:
    def __init__(self, yvar):
        self.yvar = yvar
    def fit(self, df):
        return self
    def transform(self, df):
        return df[self.yvar].values.copy(),None
    def inverse_transform(self, df, yrawpred):
        return yrawpred.copy()
    # def predict_from_raw(self, df, yrawpred):
    #     return self.inverse_transform(df, yrawpred)

class TargetStandardScaler(TargetIdentity):
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
    def __init__(self, yvar, aircraft_type):
        super().__init__(yvar,aircraft_type)
        self.dmass = pd.read_csv("aircraft_type_masses.csv")
        for _,line in self.dmass.iterrows():
            self.dmean[line[self.by]] = line.oew
            self.dscale[line[self.by]] = line.mtow - line.oew

class MassStandardScalerByAircraft(TargetScaleByGroup):
    def fit(self, df):
        # dmass = pd.read_csv("aircraft_type_masses.csv")
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
        # modelscale = linear_model.LinearRegression().fit(poly.fit_transform(X),yscale)
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
        # import matplotlib.pyplot as plt
        # plt.scatter(X[:,0],ymean)
        # plt.scatter(X[:,0],Xb[:,1]+modelmean.predict(X))
        # plt.show()
        # plt.scatter(X[:,0],yscale)
        # plt.scatter(X[:,0],modelscale.predict(X))
        # plt.show()
        # plt.scatter(ymean,y-modelscale.predict(X))
        # # plt.scatter(X[:,0],yscale)
        # plt.show()
        # self.meanmodel =
        # self.meanmodel =
        self.dmean = dmean
        self.dscale = dscale

# class MassStandardScalerByAircraft(TargetScaleByGroup):
#     def fit(self, df):
#         # dmass = pd.read_csv("aircraft_type_masses.csv")
#         dmean = {}
#         dscale = {}
#         for val in df[self.by].unique():
#             mask = df[self.by].values == val
#             assert(mask.sum()>10)
#             y = df[self.yvar].values[mask]
#             dmean[val] = np.mean(y)
#             dscale[val] = np.std(y)
#         self.dmean = dmean
#         self.dscale = dscale


    # def get_mtow(self,aircraft_type):
    #     ac = self.dmass.query(f"{self.aircraft_type}==@aircraft_type")
    #     assert ac.shape[0]==1
    #     mtow = ac.mtow.values[0]
    #     oew = ac.oew.values[0]
    #     return (oew,mtow)
    # def transform(self, df):#aircraft_type, tow):
    #     res = np.empty_like(df[self.yvar].values)
    #     res[:]=np.nan
    #     w = res.copy()
    #     self.dtrans = {}
    #     for val in df[self.aircraft_type].unique():
    #         # print(val)
    #         mask = df[self.aircraft_type].values == val
    #         oew, mtow = self.get_mtow(val)
    #         y = (df[self.yvar].values[mask]-oew) / (mtow-oew)
    #         res[mask]=y
    #         w[mask] = (mtow-oew)
    #         # self.dtrans[val]=QuantileTransformer().fit(y[:,None])
    #         # res[mask] = self.dtrans[val].transform(y[:,None])[:,0]
    #     return res, w**2
    # def inverse_transform(self, df, yrawpred):
    #     res = np.empty_like(yrawpred)
    #     res[:]=np.nan
    #     for val in df[self.aircraft_type].unique():
    #         mask = df[self.aircraft_type].values == val
    #         oew, mtow = self.get_mtow(val)
    #         res[mask] = yrawpred[mask] * (mtow-oew) + oew
    #         # res[mask] = self.dtrans[val].inverse_transform(normalized_tow[mask][:,None])[:,0] * self.get_mtow(val)
    #     return res

class GroupByTransformer(BaseEstimator, TransformerMixin):
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
        # df = X
        # for ac in sorted(df.aircraft_type.unique()):
        #     print(ac,df.query("aircraft_type==@ac").shape[0])
        for val in X[self.by].unique():
            mask = X[self.by]==val
            transformed = self.dtransformer[val].transform(X.loc[mask, self.cols])
            X.loc[mask, self.cols] = transformed
        # df = X
        # for ac in sorted(df.aircraft_type.unique()):
        #     print(ac,df.query("aircraft_type==@ac").shape[0])
        # raise Exception
        return X.loc[:, self.cols]
    def inverse_transform(self, X, y = None):
        X = X.copy()[self.cols+[self.by]]
        # df = X
        # for ac in sorted(df.aircraft_type.unique()):
        #     print(ac,df.query("aircraft_type==@ac").shape[0])
        for val in X[self.by].unique():
            mask = X[self.by]==val
            transformed = self.dtransformer[val].inverse_transform(X.loc[mask, self.cols])
            X.loc[mask, self.cols] = transformed
        return X.loc[:, self.cols]
    def get_feature_names_out(self,names):
        return [v for v in names if v!=self.by]

def main():
    import readers
    import utils
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    config = utils.read_config()
    df = readers.read_features("challenge")
    for val in [ 'A310', 'B752', 'B773', 'C56X', 'E290']:
        df = df.query("aircraft_type!=@val")
    categorical_features = ["adep","ades","airline","aircraft_type","wtc","callsign","country_code_ades","country_code_adep","dayofweek","weekofyear"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1,dtype=np.int32)),
        ]
    )
    scale_by_group = "aircraft_type"
    numeric_features = ["flight_duration","taxiout_time","flown_distance",] + [f"{v}{i}" for i in range(7) for v in ["mass_","masscount_"]]
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", GroupByTransformer(StandardScaler,by=scale_by_group)),#
        ]
    )
    numeric_features_not_scaled = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features+[scale_by_group]),
            # ("numeric", StandardScaler(), numeric_features),
            # ("unscalednumeric", StandardScaler(), numeric_features_not_scaled),
        ],
        remainder = "drop",
    )
    preprocessor.fit(df)
    scaled = preprocessor.transform(df)
    ynames = preprocessor.get_feature_names_out()
    print(scaled.shape)
    print(ynames)
    print(len(ynames))
    transformed = pd.DataFrame(data=scaled,columns=ynames)
    # raise Exception
    # print(pd.DataFrame(data=preprocessor.transform(df.query("aircraft_type=='A310'")),columns=ynames)[["numeric__mass_0"]])
    # print(df.query("aircraft_type=='A310'")[["mass_0"]])
    print(readers.read_features("challenge").groupby("aircraft_type").aircraft_type.count())
    print(readers.read_features("submission").groupby("aircraft_type").aircraft_type.count())
    # raise Exception
    normass = NormalizeMass()
    # print(df.query("aircraft_type=='C56X'")[["tow"]])
    # raise Exception
    df = df.query("aircraft_type!='C56X'")
    for val in df.aircraft_type.unique():
        trajs = df.query("aircraft_type==@val")
        normed = normass.transform(trajs.aircraft_type,trajs.tow)
        print(val,normed.min(),normed.max(),trajs.tow.mean(),trajs.tow.max())
        plt.hist(normed,bins=30,density=True)
        plt.show()
        # plt.hist(trajs.tow.values,bins=30)#,density=True)

    # plt.scatter(transformed.numeric__flight_duration,transformed.numeric__mass_0)
    # plt.show()
if __name__ == '__main__':
    main()
