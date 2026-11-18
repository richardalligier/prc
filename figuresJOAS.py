import pandas as pd
import correct_date
import utils
import matplotlib.pyplot as plt
from traffic.core import Traffic
from filterclassic import FilterCstPosition, FilterCstSpeed, MyFilterDerivative, FilterCstLatLon,FilterIsolated
import filter_trajs
import matplotlib.dates as mdates
import numpy as np
import features
FIGURESOUT = "/home/alligier/papiers/PRC2024/figuresJOAS"
config = utils.read_config()
FOLDER_DATA = config.FOLDER_DATA

DNAMES ={
    "t" : "{METAR\\\\ data}",
    "c" : "cruise",
    "o" : "others",
    "m" : "masses",
    "e" : "{specific\\\\ energy}",
}
CLMB_THRESHOLD_DT="{Points dt}"
CLMB_PERIODS="{periods}"
MIN_ALT_SLICE="{min alt slice [ft]}"
STEP_ALT_SLICE="{vertical slice \\\\ height [ft]}"
MAX_ALT_SLICE="{max alt slice [ft]}"
LK = "{Subsections\\\\ {\\ref{sub:repeated}}-{\\ref{sub:erroneous}}}"
NF = "\\texttt{.filter('default')}"
AG = "\\texttt{.filter('aggressive')}"
NT = "no filter"
NS = "no interpolation"
CS = "{Subsection\\\\ {\\ref{sub:smoothed}}}"
RS = "\\texttt{.resample('1s')}"
MTOW = "mtow"
NOSCALE = "identity"
STANDARD = "standard"
RMSE = "RMSE [kg]"
WIND_CORRECTION="{wind\\\\ correction}"
TEMP_CORRECTION="{temperature\\\\ correction}"
DFILTER = {
    LK : "likablejelly",
    NF : "default",
    AG : "aggressive",
    NT : "nothing",
}
DALTITUDE ={
    CS : "daltitude",
    RS : "vertical_rate",
    NS : "vertical_rate",
}
DSMOOTH = {
    CS : "1e-2",
    RS : "resample",
    NS : "nosmooth",
}
DSCALING = {
    MTOW : "mtow",
    NOSCALE : "identity",
    STANDARD : "standard",
}
DYESNO = {
    "yes" : "yes",
    "no":"no"
}
KEEPTOGETHER=[list(DNAMES.values()),[CLMB_THRESHOLD_DT,CLMB_PERIODS,MAX_ALT_SLICE,STEP_ALT_SLICE,MIN_ALT_SLICE],["filter","interpolation"],[WIND_CORRECTION,TEMP_CORRECTION]]

def testtogether(k,d):
    s = set()
    for g in KEEPTOGETHER:
        for kd in d:
            if kd in g:
                s.add(tuple(g))
    for gi in s:
        if k in gi:
            return True
    return False

def load_data():
    fname = "2022-07-14.parquet"
    df = pd.read_parquet(f"{FOLDER_DATA}/rawtrajectories/{fname}")
    airports = pd.read_parquet(f"{FOLDER_DATA}/airports_tz.parquet")
    flights = pd.read_parquet(f"{FOLDER_DATA}/flights/challenge_set.parquet")
    clean = pd.read_parquet(f"{FOLDER_DATA}/classic__1e-2_interpolated_trajectories/{fname}")
    dates = correct_date.compute_dates(flights,airports,clean)
    flights["t_adep"]=flights["actual_offblock_time"]+pd.to_timedelta(flights["taxiout_time"],unit="minutes")
    flights["t_ades"]=flights["arrival_time"]
    selected=dates.set_index("flight_id").loc[(dates.set_index("flight_id").t_adep-flights.set_index("flight_id").t_adep).abs()> pd.Timedelta("10m")]
    return df, flights, clean, selected, dates

def plot_trajs(df,flights,clean,dates):
    fid=253437355
    traj = df.query("flight_id==@fid")
    filter = FilterCstLatLon()|FilterCstPosition()|FilterCstSpeed()
    tran = Traffic(traj).filter(filter=filter,strategy=filter_trajs.nointerpolate).eval(max_workers=1).data
    trac = clean.query("flight_id==@fid")
    ymin=0
    ymax=35000
    cvline = "black"
    ptextadep = 1.01
    ptextades = 1.01
    xmin = flights.query("flight_id==@fid").iloc[0].t_adep-pd.Timedelta(3,unit="minutes")
    xmax = traj.timestamp.max()+pd.Timedelta(5,unit="minutes")

    def plot(fid,df,flights,fname,textadep,textades):
        plt.scatter(df.timestamp,df.altitude,s=1)
        plt.vlines(flights.query("flight_id==@fid").iloc[0].t_adep,ymin=ymin,ymax=ymax,colors=cvline)
        plt.text(flights.query("flight_id==@fid").iloc[0].t_adep,(ymin*(1-ptextadep)+ymax*ptextadep),textadep)
        plt.text(flights.query("flight_id==@fid").iloc[0].t_ades,(ymin*(1-ptextades)+ymax*ptextades),textades,ha="right")
        plt.vlines(flights.query("flight_id==@fid").iloc[0].t_ades,ymin=ymin,ymax=ymax,colors=cvline)
        plt.ylabel("altitude [ft]")
        plt.xlabel("timestamp [hours:minutes]")
        plt.xlim(xmin, xmax) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        f=1.5
        plt.gcf().set_size_inches(4.8*f, 2.5*f)
        plt.tight_layout(pad=0)
        
        plt.savefig(fname,pad_inches=0,dpi=300)
        plt.clf()

    plot(fid,traj,flights,f"{FIGURESOUT}/traj1.png","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    plot(fid,tran,flights,f"{FIGURESOUT}/traj2.png","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    plot(fid,trac,flights,f"{FIGURESOUT}/traj3.png","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    plot(fid,trac,dates,f"{FIGURESOUT}/traj4.png","t_adep","t_ades")
    print(flights.query("flight_id==@fid")[["t_adep","arrival_time"]])#)
    print(dates.query("flight_id==@fid"))

def plot_climbs(df,flights,clean,dates):
    fids = list(dates.flight_id.unique())[70:75]#40:45
    #traj = df.query("fligh "vertical_rate",t_id==@fid")
    ymin=-500
    ymax=47000
    cvline = "black"
    ptextadep = 1.01
    ptextades = 1.01
    cdates = correct_date.fillnadates(flights,dates)#compute_dates(flights,airports,clean)
    def plot(fids,flights,fname,textadep,textades,slices=False,filter=True):
        scale=False
        for fid in fids:
            trac = clean.query("flight_id==@fid")
            if filter:
                trac = trac.query("daltitude>500")
            print(flights.query("flight_id==@fid"))
            tadep = flights.query("flight_id==@fid").iloc[0].t_adep
            tades = flights.query("flitesttogetherght_id==@fid").reset_index().iloc[0].t_ades
            t = (trac.timestamp - tadep)/(tades-tadep) if scale else (trac.timestamp - tadep).dt.total_seconds()
            print(t)
            plt.scatter(t,trac.altitude,s=1)
        plt.ylabel("altitude [ft]")
        plt.vlines(0,ymin=ymin,ymax=ymax,colors=cvline)
        plt.text(0,(ymin*(1-ptextadep)+ymax*ptextadep),textadep)
        plt.xlabel(r"$timestamp-t\_adep$ [s]")
        xs = np.arange(-500,48000,1000)
        plt.hlines(xs[-1],xmin=0,xmax=9000,colors="white",linewidth=0.5)
        plt.hlines(xs[0],xmin=0,xmax=9000,colors="white",linewidth=0.5)
        plt.vlines(14500,ymin=0,ymax=9000,colors="white",linewidth=0.5)
        if slices:
            for xi in zip(xs):
                plt.hlines(xi,xmin=0,xmax=14500,colors=cvline,linewidth=0.5)
        f=1.5
        plt.gcf().set_size_inches(4.8*f, 1.8*f)
        plt.tight_layout(pad=0)
        
        plt.savefig(fname,pad_inches=0,dpi=300)
        plt.clf()

    plot(fids,cdates,f"{FIGURESOUT}/climb0.png","t_adep","t_ades",filter=False)
    plot(fids,cdates,f"{FIGURESOUT}/climb1.png","t_adep","t_ades")
    plot(fids,cdates,f"{FIGURESOUT}/climb2.png","t_adep","t_ades",slices=True)
    #plot(tran,flights,"figures/traj2.pdf","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    #plot(trac,flights,"figures/traj3.pdf","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    #plot(trac,dates,"figures/traj4.pdf","t_adep","t_ades")
    #print(flights.query("flight_id==@fid")[["t_adep","arrival_time"]])#)
    #dates.query("flight_id==@fid")

def plot_cruises(df,flights,clean,dates):
    fids = list(dates.flight_id.unique())[70:75]#40:45
    #traj = df.query("flight_id==@fid")
    ymin=0
    ymax=40000
    cvline = "black"
    ptextadep = 1.01
    ptextades = 1.01
    cdates = correct_date.fillnadates(flights,dates)#compute_dates(flights,airports,clean)
    def plot(fids,flights,fname,textadep,textades,scale=True,slices=False):
        for fid in fids:
            trac = clean.query("flight_id==@fid")
            print(flights.query("flight_id==@fid"))
            tadep = flights.query("flight_id==@fid").iloc[0].t_adep
            tades = flights.query("flight_id==@fid").reset_index().iloc[0].t_ades
            t = (trac.timestamp - tadep)/(tades-tadep) if scale else (trac.timestamp - tadep).dt.total_seconds()
            print(t)
            plt.scatter(t,trac.altitude,s=1)
        plt.ylabel("altitude [ft]")
        plt.vlines(0,ymin=ymin,ymax=ymax,colors=cvline)
        plt.text(0,(ymin*(1-ptextadep)+ymax*ptextadep),textadep)
        #plt.ylim(0, 38000)
        if scale:
            plt.text(1,(ymin*(1-ptextades)+ymax*ptextades),textades,ha="right")
            plt.vlines(1,ymin=ymin,ymax=ymax,colors=cvline)
            plt.xlabel(r"$\frac{timestamp-t\_adep}{t\_ades-t\_adep}$ [-]")
            plt.xlim(-0.1, 1.1)
        else:
            plt.xlabel(r"$timestamp-t\_adep$ [s]")
        if slices:
            xs = np.linspace(0,1,21)
            for xi in zip(xs[1:-1]):
                plt.vlines(xi,ymin=ymin,ymax=40000,colors=cvline,linewidth=0.5)
        f=1.5
        plt.gcf().set_size_inches(4.8*f, 2.2*f)
        plt.tight_layout(pad=0)
        
        plt.savefig(fname,pad_inches=0,dpi=300)
        plt.clf()
    plot(fids,cdates,f"{FIGURESOUT}/cruise0.png","t_adep","t_ades",scale=False)
    plot(fids,cdates,f"{FIGURESOUT}/cruise1.png","t_adep","t_ades")
    plot(fids,cdates,f"{FIGURESOUT}/cruise2.png","t_adep","t_ades",slices=True)
    #plot(tran,flights,"figures/traj2.pdf","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    #plot(trac,flights,"figures/traj3.pdf","t_adep=actual_offblock_time+taxiout_time","t_ades=arrival_time")
    #plot(trac,dates,"figures/traj4.pdf","t_adep","t_ades")
    #print(flights.query("flight_id==@fid")[["t_adep","arrival_time"]])#)
    #dates.query("flight_id==@fid")

def main_filters():
    df, flights, clean, selected, dates = load_data()
    plot_trajs(df,flights,clean,dates)
    plot_cruises(df,flights,clean,dates)
    plot_climbs(df,flights,clean,dates)

def rmse(df,tf):
    return np.sqrt(np.mean((df.set_index("flight_id").tow-tf.set_index("flight_id").tow)**2))
collect = []
def read_pred(filt,smooth,dalt,vars,scaling,i,CLMB_ALT_START,CLMB_ALT_STEP,CLMB_ALT_STOP,CRUISE_NSPLIT,WIND_ERATE,TEMP_ERATE,CLMB_THRESHOLD_DT,CLMB_PERIODS):
    setup = f"{filt}__{smooth}_{filt}__{smooth}__{CLMB_PERIODS}_500_{CLMB_THRESHOLD_DT}_{dalt}_1_{CLMB_ALT_START}_{CLMB_ALT_STOP}_{CLMB_ALT_STEP}_{WIND_ERATE}_{TEMP_ERATE}_{filt}__{smooth}__{CRUISE_NSPLIT}_{filt}__{smooth}"
    try:
        params = f"PREFIX_FILTER={filt} CLMB_PERIODS={CLMB_PERIODS} CLMB_THRESHOLD_DT={CLMB_THRESHOLD_DT} INTERPOL_SMOOTH={smooth} CLMB_VRATE_VAR={dalt} VARIABLES={vars} SCALER={scaling} CLMB_ALT_STEP={CLMB_ALT_STEP} CRUISE_NSPLIT={CRUISE_NSPLIT} WIND_ERATE={WIND_ERATE} TEMP_ERATE={TEMP_ERATE} CLMB_ALT_STOP={CLMB_ALT_STOP} CLMB_ALT_START={CLMB_ALT_START}"
        features = f"make -j6 features {params};"
        #print(features)
        tobeadded=f"make submissions {params}" #{features} ; 
        #tobeadded=f"make submissions {params}"
        if collect==[] or collect[-1]!=tobeadded:
            collect.append(tobeadded)
        #df =pd.read_parquet(f"{FOLDER_DATA}/flights/ranking_test_final.parquet")[['flight_id', 'tow']]
        #return df
        return pd.read_csv(f"{FOLDER_DATA}/submissions/{setup}/submit-train-ranking_test_final-{vars}-{i}-{scaling}-predictions.csv")#
    except Exception as e:
        print(" ; ".join(collect))
        df =pd.read_parquet(f"{FOLDER_DATA}/flights/ranking_test_final.parquet")[['flight_id', 'tow']]
        return df
        #raise e
def read_data(filt,smooth,dalt,vars,scaling,iteri,CLMB_ALT_START,CLMB_ALT_STEP,CLMB_ALT_STOP,CRUISE_NSPLIT,WIND_ERATE,TEMP_ERATE,CLMB_THRESHOLD_DT,CLMB_PERIODS):
    #periods=5
    CLMB_THRESHOLD_VR=500
#    CLMB_THRESHOLD_DT=40
    CLMB_CTHRUST=1
    prefix_interpol = f"{filt}__{smooth}"
    #setup = f"{filt}__{smooth}_{filt}__{smooth}__{periods}_{CLMB_THRESHOLD_VR}_{CLMB_THRESHOLD_DT}_{dalt}_{CLMB_CTHRUST}_{CLMB_ALT_START}_{CLMB_ALT_STOP}_{CLMB_ALT_STEP}_{WIND_ERATE}_{TEMP_ERATE}_{filt}__{smooth}__{CRUISE_NSPLIT}_{filt}__{smooth}"
    trainfile = "train"
    prefixes = {
        "mass":f"{prefix_interpol}__{CLMB_PERIODS}_{CLMB_THRESHOLD_VR}_{CLMB_THRESHOLD_DT}_{dalt}_{CLMB_CTHRUST}_{CLMB_ALT_START}_{CLMB_ALT_STOP}_{CLMB_ALT_STEP}_{WIND_ERATE}_{TEMP_ERATE}",
        "cruise":f"{prefix_interpol}__{CRUISE_NSPLIT}",
        "wind":f"{prefix_interpol}",
    }
    variables = "oefmntwc"
    feat = features.read_features(trainfile,prefixes,variables) #train_test(args.random_state,args.train,args.submit,prefixes,args.variables)
    return feat.data

def read_mean_pred(**kwargs):
#def read_mean_pred(filt,smooth,dalt,vars,scaling,iteri,INCLUDE_TAKEOFF,CLMB_ALT_STEP,CLMB_ALT_STOP,CRUISE_NSPLIT,WIND_ERATE,TEMP_ERATE):
    l = []
    params=kwargs.copy()
    iteri=params["iteri"]
    del params["iteri"]
    for i in iteri:
        params["i"]=i
        l.append(read_pred(**params))
    df = pd.concat(l)
    return df.groupby("flight_id").mean().reset_index()

def tabularize(s):
    s = s.replace("tabular","tblr")
    for k in ["midrule","toprule","bottomrule"]:
        s = s.replace(k,"hline")
    return s

def write_dict_as_table(d,dual=False):
    print(RMSE)
    def stripaccolade(s):
        return s[1:-1] if s[0]=="{" else s
    def unique(l):
        s=[]
        for x in l:
            if x not in s:
                s.append(x)
        return s

    def aux(d,dual=False,limit=100000):
        print(RMSE)
        if dual:
            print(d)
            assert(len(d)==3)
            dvalues = {k:unique(d[k]) for k in d if k!=RMSE}
            #print(dvalues)
            #raise Exception
            cmax = max(d,key=lambda x:len(set(x)))
            cmin = list(set(dvalues)-set([cmax]))[0]
            # print(cmin)
            # print(cmax)
            # raise Exception
            newd={("\\diagbox{"+stripaccolade(cmin)+"}{"+stripaccolade(cmax)+"}"): dvalues[cmin]}
            dd = pd.DataFrame(d)
            dcross = {}
            for _,line in dd.iterrows():
                dcross[(line[cmin],line[cmax])]=line[RMSE]
            #print(dcross)
            #print(dd)
            #print(RMSE)
            #print(dvalues)
            for k in dvalues[cmax]:
                newd[k]=[dcross[(y,k)] for y in dvalues[cmin]]
            d = newd
            #print(d)
            #print(pd.DataFrame(d))
            #print(dvalues)
            #raise Exception
            lnames = DNAMES.values()
            c ="".join([ "X[l]" if name in lnames else "l"  for name in d])
            c =c[:1]+"|"+c[1:]
            c = "\ncolspec = {"+c+"},\ncolsep = 3pt,\n"
            return tabularize(pd.DataFrame(d).to_latex(index=False,float_format="{:.0f}".format,column_format=c))
            #return tabularize(pd.DataFrame(d).to_latex(index=False,float_format="{:.0f}".format,column_format=c))
        else:
            lnames = DNAMES.values()
            c ="".join([ "c" if name in lnames else "l"  for name in d])
            c = "\ncolspec = {"+c+"},\ncolsep = 3pt,\n"
            if len(d)<=limit:
                return tabularize(pd.DataFrame(d).to_latex(index=False,float_format="{:.0f}".format,column_format=c))
            else:
                dlimit = {}
                dother = {}
                for i,(k,v) in enumerate(d.items()):
                    if testtogether(k,dlimit):
                        dlimit[k]=v
                    elif i<limit:
                        dlimit[k]=v
                    else:
                        dother[k]=v
                s = tabularize(pd.DataFrame(dlimit).to_latex(index=False,float_format="{:.0f}".format,column_format=c))
                print(dlimit)
                print(dother)
                if len(dother)==0:
                    return s
                else:
                    return s + aux(dother,dual,limit)
                
    dconst = {k:v[:1] for k,v in d.items() if len(set(v))==1}
    dvar = {k:v for k,v in d.items() if len(set(v))>1}
    print((dvar))
    print((dconst))
    print(len(dvar),len(dconst))
    if len(dconst)>0 and len(dvar)>0:
        h="\\subfloat[Constant parameters]%\n{\n\\vbox{"
        f = "}\n}\n\\vspace*{0.5cm}\\hfill\n\\subfloat[Parameters that changes and the associated RMSE (in~kg)]%\n{"
        e= "}"
        return h+aux(dconst,limit=4) + f + aux(dvar,dual=dual) +e
    elif len(dvar)==0:
        return aux(dconst,dual=False,limit=8)
    else:
        return aux(dvar)

def add_measure(d,k,v):
        if k not in d:
            d[k]=[v]
        else:
            d[k].append(v)

def make_tables(iterators,score):
    d = {}
    for params in iterators:
        #print(params)
        add_measure(d, "filter", params["filt"])
        add_measure(d, "interpolation", params["smooth"])
        add_measure(d, MIN_ALT_SLICE, int(params["CLMB_ALT_START"]*1000))
        add_measure(d, STEP_ALT_SLICE, int(params["CLMB_ALT_STEP"]*1000))
        last_alt= np.arange(
            int(params["CLMB_ALT_START"]*1000),
            int(params["CLMB_ALT_STOP"]*1000),
            int(params["CLMB_ALT_STEP"]*1000),
            )[-1]
        add_measure(d, MAX_ALT_SLICE, last_alt)
        add_measure(d, WIND_CORRECTION, params["WIND_ERATE"])
        add_measure(d, TEMP_CORRECTION, params["TEMP_ERATE"])
        add_measure(d, CLMB_PERIODS, params["CLMB_PERIODS"])
        add_measure(d, CLMB_THRESHOLD_DT, params["CLMB_THRESHOLD_DT"])
        add_measure(d, "{number of\\\\ temporal slices}", params["CRUISE_NSPLIT"])
        add_measure(d, "scaling", params["scaling"])
        for k,vname in DNAMES.items():
            add_measure(d,vname, r"\Checkmark" if k in params["vars"] else r"\XSolidBrush")
        add_measure(d,RMSE , score(params))
    return d
def main_tables():
    NMODEL = 10
    ref = pd.read_parquet(f"{FOLDER_DATA}/flights/ranking_test_final.parquet")
    d = {}
    params_default = {
        "filt":LK,
        "smooth":CS,
        "dalt":CS,
        "vars":"oefmntwc",
        "scaling":MTOW,
        "iteri":list(range(NMODEL)),
        "CLMB_ALT_STEP":1,
        "CRUISE_NSPLIT":20,
        "WIND_ERATE":"yes",
        "TEMP_ERATE":"yes",
#        "INCLUDE_TAKEOFF":"yes",
        "CLMB_ALT_START":-0.5,
        "CLMB_ALT_STOP":48,
        "CLMB_THRESHOLD_DT":40,
        "CLMB_PERIODS":5,
        }
    def convert(params):
        params = params.copy()
        for k,v in [("filt",DFILTER),("smooth",DSMOOTH),("dalt",DALTITUDE),("scaling",DSCALING)]:
            params[k]=v[params[k]]
        return params
    def score_rmse(params):
        #print(params)
        params = convert(params.copy())
        #print(params)
        l = []
        for i in params["iteri"]:
            params["iteri"]=range(i,i+1)
            try:
                l.append(rmse(read_mean_pred(**params),ref))
            except Exception as e:
                raise e
                l.append(np.nan)
        return np.mean(np.array(l))#+np.random.randn()
    def count_energy(params):
        params = convert(params.copy())
        l = []
        for i in params["iteri"]:
            params["iteri"]=range(i,i+1)
            try:
                df=read_data(**params)
                v = [x for x in list(df) if "climbe_rate" in x]
                l.append(df[v].isna().mean(axis=None)*1000000)
            except Exception as e:
                #raise e
                l.append(np.nan)
        return np.mean(np.array(l))#+np.random.randn()        
    params = params_default.copy()
    d = make_tables([params],score_rmse)
    del d[RMSE]
    with open(f"{FIGURESOUT}/averaging.tex", "w") as f:
        f.write(write_dict_as_table(d,dual=False))
    #raise Exception
    
    #for filt,smooth in [(NT,NS),(NT,RS),(NF,RS),(AG,CS),(NF,CS),(LK,RS),(LK,CS)]:
    lparams = []
    for v in ["oefmntwc",]:
        for filt in [NT,NF,LK]:#for filt in [NT,NF,AG,LK]:
            for smooth in [NS,RS,CS]:
                #for thresh in [400,40]:
                    for periods in [1,3,5]:
                        params = params_default.copy()
                 #       params["CLMB_THRESHOLD_DT"]=thresh
                        params["CLMB_PERIODS"]=periods
                        params["filt"]=filt
                        params["smooth"]=smooth
                        params["dalt"]=smooth
                        params["vars"]=v
                        params["iteri"]=[0]
                        lparams.append(params)
                        score_rmse(params)
    d = make_tables(lparams,score_rmse) 
    print(pd.DataFrame(d))
    with open(f"{FIGURESOUT}/preproc.tex", "w") as f:
        f.write(write_dict_as_table(d,dual=False))
    raise Exception(" ; ".join(collect))
    for p in lparams:
        p["iteri"]=[0]
    d = make_tables(lparams,count_energy)
    print(pd.DataFrame(d))
    with open(f"{FIGURESOUT}/preproccount.tex", "w") as f:
        f.write(write_dict_as_table(d,dual=False))
    raise Exception
    lparams = []
    for filt,smooth,scaling,v in [(LK,CS,MTOW,"otwc"),(NF,RS,NOSCALE,"otwc")]:
            params = params_default.copy()
            params["filt"]=filt
            params["smooth"]=smooth
            params["dalt"]=smooth
            params["scaling"]=scaling
            params["vars"]=v
            lparams.append(params)
    d = make_tables(lparams,score_rmse)     
    print(pd.DataFrame(d))
    with open(f"{FIGURESOUT}/vs.tex", "w") as f:
        f.write(write_dict_as_table(d))
    lparams = []
    for filt,smooth,scaling,v,crse_nsplit,alt_start,alt_stop,alt_step in [
        (NF,RS,NOSCALE,"otwcef",10,-0.5,48,1),
        (NF,RS,NOSCALE,"otwcef",3,0,48,5),
        (NF,RS,NOSCALE,"otwcef",3,0,25,5),
        #(NF,RS,NOSCALE,"otwcef",3,-0.5,48,5),
        #(NF,RS,NOSCALE,"otwcef",3,-0.5,23,5)
        ]:
    #[(NF,RS,NOSCALE,"otwcef",10,-0.5,48,1),(NF,RS,NOSCALE,"otwcef",3,0,48,5),(NF,RS,NOSCALE,"otwcef",3,0,25,5),(NF,RS,NOSCALE,"otwcef",3,-0.5,48,5),(NF,RS,NOSCALE,"otwcef",3,-0.5,23,5)]:
        params = params_default.copy()
        params["CLMB_ALT_START"]=alt_start
        params["CLMB_ALT_STOP"]=alt_stop
        params["CLMB_ALT_STEP"]=alt_step
        params["WIND_ERATE"]="no"
        params["TEMP_ERATE"]="no"
        params["filt"]=filt
        params["smooth"]=smooth
        params["dalt"]=smooth
        params["scaling"]=scaling
        params["CRUISE_NSPLIT"]=crse_nsplit
        params["vars"]=v
        lparams.append(params)
    d = make_tables(lparams,score_rmse)
    print(pd.DataFrame(d))
    with open(f"{FIGURESOUT}/vs10.tex", "w") as f:
        f.write(write_dict_as_table(d))#f.write(tabularize(pd.DataFrame(d).to_latex(index=False,float_format="{:.0f}".format,column_format="l" * 5 + COLUMNTICKS*(len(d)-6) +"l" )))
    #return 
    lparams = []
    for clmb_alt in [0.5,1,2,5]:#,10]:#(10,10),(0.5,20),(1,20),(2,20),(1,10),(1,30)]:
        for crse_nsplit in [3,5,10,20,30,40]:
            params = params_default.copy()
            params["CLMB_ALT_STEP"]=clmb_alt
            params["CRUISE_NSPLIT"]=crse_nsplit
            lparams.append(params)
    d = make_tables(lparams,score_rmse)
    del d[MAX_ALT_SLICE]
    print(pd.DataFrame(d))
    #return
    with open(f"{FIGURESOUT}/vsbetter.tex", "w") as f:
        f.write(write_dict_as_table(d,dual=True))      
    lparams = []
    for v in ["oefmntwc",]:
        for filt,smooth in [(LK,CS)]:
            for scaling in [NOSCALE,MTOW]:
                params = params_default.copy()
                params["filt"]=filt
                params["smooth"]=smooth
                params["dalt"]=smooth
                params["scaling"]=scaling
                params["vars"]=v
                lparams.append(params)
    d = make_tables(lparams,score_rmse) 
    print(pd.DataFrame(d))
    with open(f"{FIGURESOUT}/scaler.tex", "w") as f:
        f.write(write_dict_as_table(d))#f.write(tabularize(pd.DataFrame(d).to_latex(index=False,float_format="{:.0f}".format)))



    tnames ={
        "t" : "Table~\\ref{tb:features_weather} and \\ref{tb:features_sigweather}",
        "c" : "Table~\\ref{tb:features_profile}",
        "o" : "Table~\\ref{tb:features_climb}",
        "m" : "Table~\\ref{tb:features_climb}",
        "e" : "Table~\\ref{tb:features_climb}",
    }
    lparams = []
    for v in ["tw","c","o","mn","ef","oefmntwc",]:
        params = params_default.copy()
        params["vars"]=v
        lparams.append(params)
    d = make_tables(lparams,score_rmse) 
    '''  
arrays=[[tnames[v] for v in DNAMES],[DNAMES[v] for v in DNAMES]]
arrays[0].append("")
arrays[1].append("RMSE [kg]")
columns = pd.MultiIndex.from_arrays(arrays)
a = np.array(list(d.values())).T
print(a)
print(f"{a.shape=}")
df=pd.DataFrame(a,columns=columns)
print(df)
'''
    #print(pd.DataFrame(np.array(d.values()).T, columns=columns))
    with open(f"{FIGURESOUT}/vars.tex", "w") as f:
        f.write(write_dict_as_table(d))#f.write(tabularize(df.to_latex(index=False,float_format="{:.0f}".format,multicolumn=True, multirow=True,multicolumn_format="|c|",column_format="|".join(["c"]*a.shape[0]))))
    
    lparams = []
    for wind_erate in ["yes","no"]:#(10,10),(0.5,20),(1,20),(2,20),(1,10),(1,30)]:
        for temp_erate in ["yes","no"]:
            params = params_default.copy()
            params["WIND_ERATE"]=wind_erate
            params["TEMP_ERATE"]=temp_erate
            lparams.append(params)
    d = make_tables(lparams,score_rmse)     
    print(pd.DataFrame(d))
    #return
    with open(f"{FIGURESOUT}/windtemperate.tex", "w") as f:
        f.write(write_dict_as_table(d,dual=True))

    nb = 10
    lparams =[]
    l=[]
    np.random.seed(0)
    for i in range(1,nb+1):
        params = params_default.copy()
        lrepeat=[]
        for _ in range(100):
            li = np.array(range(nb))
            np.random.shuffle(li)
            params["iteri"]=li[:i]
            lrepeat.append(rmse(read_mean_pred(**convert(params)),ref))
        l.append(np.array(lrepeat).mean())
    #l = [rmse(read_mean_pred(DFILTER[LK],DSMOOTH[CS],DALTITUDE[CS],"oemtwc",DSCALING[MTOW],range(i)),ref) for i in range(1,nb+1)]

    print(l)
    plt.scatter(range(1,nb+1),l)
    f=1.5
    plt.gcf().set_size_inches(4.8*f, 2.2*f)
    plt.xlabel("number of models averaged [-]")
    plt.ylabel("RMSE on the Take-Off Weight [kg]")
    plt.savefig(f"{FIGURESOUT}/averaging.pdf",bbox_inches="tight",pad_inches=0,dpi=300)
    plt.clf()
    
    

if __name__ == '__main__':
    main_tables()
    print(" ; ".join(collect))
