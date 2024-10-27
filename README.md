# PRC Data Challenge
The objective of this challenge was to build an **open** Machine Learning model predicting the Take Off Weight (TOW) of an aircraft from its flight and trajectory information, more info at [PRC Challenge](https://ansperformance.eu/study/data-challenge/). Please bear in mind that **all** this repository (documentation included) was done during a competition and hence was done in a limited time.

# Overview of our method
Our predictions are obtained by averaging the results of several gradient-boosted tree models, trained with different random seeds on the same data.

In order to predict the take-off weight, our model takes as input a number of basic variables such as the departure and arrival airports, the airline, aircraft type, wake turbulence category, day of week, the flight duration, taxi-out time, flown distance, etc, and additionnal variables extracted from the ADS-B trajectories, and also weather data obtained from METAR. Prior to this feature extraction, the trajectories are filtered (see technical details below) and smoothed using cubic splines.

The features extracted from ADS-B trajectories include:
- several statistics on the rate of climb or descent (ROCD), the energy rate and an estimated "equivalent mass" obtained using an OpenAP, an open point-mass aircraft performance model, for a number of altitude "slices" of each trajectory,
- the median Mach number, median altitude and altitude difference between the first and last point, for a number of temporal "slices" of the trajectory, capturing cruise informations,
- the average wind along the trajectory.

The features derived from METAR include the presence of thunderstorms or fog at the arrival airport or in an area around this airport, around the time of arrival.

The target variable TOW and some explanatory variables were rescaled so as to ease training. More details on the additionnal features and training process are given in the "Technical Aspects" section below.

# The Results
On a ThreadRipper
1920X, one model takes ~65 minutes to train. Each model is a LightGBM
model with 50,000 trees. As the training process involves
randomness, averaging different models (hence different draws of the model)
improves the results[^note1]. In the final part of the challenge, we just trained models in the limited
time we had left, and averaged them.

| number of model(s) averaged | RMSE on the final\_submission\_set [kg] | seed(s) | submission version |
|----------------------------:|:---------------------------------------:|:---------|---------------------:|
|                           1 | 1,612                                 |    0     |          20          |
|                    10 | 1,564                                     |     0 to 9    |         21            |

# To Reproduce the Results
First setup your environment using the `environment.yml` file and edit the variable `FOLDER_DATA` in the `CONFIG` file, then just run the command below. Please be aware that it might take some time. To reduce this time depending on your computer you might want to use the option `-j` of `make`. For instance, `make -j4 cleantrajectories`, will launch 4 processes in parallel. In this whole project, each process takes no more than approximately 20GB of RAM. The only process that takes more is the training but this training phase does not use the `-j` of the `make` to run in parallel.
```
make download && make cleantrajectories && make features && make submissions
```
# Technical Aspects
In this Section you will find technical aspects and motivations for the code in this repository.
## Filtering and Smoothing Trajectories
Informations about the aircraft weight are somehow "hidden" in the kinematics of the aircraft. In order to have a good fine-grain vision of these kinematics, we need to filter and smooth the raw trajectories first.
### Filtering Trajectory (`filter_trajs.py`)
First, we try to remove measurements that are repeated from one time step to the next when no updated measurements are available. As we don't know which ones are repeated, we compute a proxy criteria by considering which variables are said to be updated together in [ADS-B](https://mode-s.org/decode/content/quickstart.html). If these variables have the same values from one measurement to the next, they are most likely repeated information. To remove erroneous measurements, we coded our own version of the `FilterDerivative` found in the [Traffic](https://github.com/xoolive/traffic) library. We took care of not interpolating data at all. In fact we even discarded "isolated" points, points which are spaced more than 20 seconds from any other point.

### Smoothing Trajectory (`interpolate.py`)
Smoothing is done using cubic smoothing splines of the [csaps](https://csaps.readthedocs.io/en/latest/) library. We avoid interpolating where there are not enough points. More specifically, we will not interpolate between two points spaced by more than 20 seconds.
## Building Features
### Correcting DateTimes (`correct_date.py`)
When looking at trajectories, the `arrival_time` variable is not
always correct. It is the same for the departure. To solve this issue,
we look at points of the trajectory in a 10 NM radius of the departure
airport (resp. arrival). Then we consider the point of maximum (resp. minimal)
timestamp among these points. Lastly, using the altitude of this
point, we add to the point's timestamp a buffer time
corresponding to a climb (or a descent) of 1000 ft/min, to compute an estimate of the departure (resp. arrival) time. Choosing a radius lower than 10 NM leaves a lot of trajectories without a
corrected date/time, so we opted for 10 NM. In cases where no points are found
inside the 10 NM radius, we keep the departure and arrival date/times
provided in the flight data.
### Features derived from METARs at departure and arrival airports (`feature_weather_from_metars.py`)
At departure and arrival airports, METARs are not always available in our files. To solve this issue, we build a `sklearn.neighbors.NearestNeighbors` "model" to query the METAR closest (in the space-time domain) to the wanted one.
### Features for Thunder/Fog at arrival airports (`feature_thunder_from_metars.py`)
We expect thunder and fog to be related to holding patterns. If such bad weather predictions are available to the crew before departure, we can expect that they might take some extra fuel on-board, thus increasing the mass. However, taking just the METAR arrival airport at the arrival time might be too fine-grained. So, using a `sklearn.neighbors.NearestNeighbors` "model", we take all the METARs in a given space-time radius around the arrival airport at arrival time, and we summarize it. Not knowing, a priori, the good space-time radius, we did that for several radius values, letting the training algorithm select the relevant ones (hopefully).
### Features Extracted from the trajectories
One difficulty to extract features from trajectories is that a given
feature should represent the same concept from one trajectory to another, a concept hopefully related to the TOW.

#### Features for the climbing Phase (`feature_climbing.py`,`polynomial.py`)
Here, trajectories are "aligned" by considering altitude slices: for instance we compute a feature for the observed ROCD between 10500ft and 11500ft, hopefully capturing the concept "climbing performance at 10000ft". Of course, this climbing segment might take place sooner or later during the climb, depending on the flight (and of the aircraft's weight). We hope to capture this by adding a temporal feature for the slice. These features are computed for 48 slices, each of height 1000ft, starting from [-500ft,500ft] to [46500ft,47500ft]. In detail, considering points inside a given slice, we compute our features by doing statistics on points in this slice:
- $\mathrm{Cardinal}\left(\mathrm{slice}\right)$ - number of points in the slice
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{ROCD}_i$
- $max_j ROCD_j - min_i ROCD_i$
<!-- - $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{ROCD}_i$ -->
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{TrueAirSpeed}_i$
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\Delta T_i$
- $\mathrm{min}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{energyrate}_i$
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{energyrate}_i$
- $\mathrm{max}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{energyrate}_i$
- $\mathrm{min}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{timestamp}_i - t\_{adep}$
- $t\_{ades} - \mathrm{max}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{timestamp}_i $
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{mass}_i $

We do that for all the 48 slices, creating hundreds of features. The last feature $\mathrm{mass}_i$ is an estimated equivalent mass computed by solving a polynomial equation of degree 2, obtained from a physical model (see [publication on this matter](https://enac.hal.science/hal-00911686/file/Alligier_ATMSeminar2013.pdf)). To do this efficiently, we use the open physical model [OpenAP](https://github.com/TUDelft-CNS-ATM/openap). A module `polynomial.py` was written, modeling a set of polynomials as numpy arrays, leveraging numpy's broadcasting. The mass feature for a slice is computed by aggregating the masses estimated at each point in the slice with a median operator. Maybe it would be better to take all the points together and solve a 4th degree polynomial as in the publication linked above. This path was not followed because, in hindsight, these mass variables might not be that useful, considering that the energy rate variable contains very similar information. The energyrate variable is the variation of the energy divided by the mass. It is computed using only kinematic information as illustrated in the formula in this 2018's [code](https://github.com/richardalligier/trc2018/blob/master/OCaml/csvaddenergyrate.ml).

One can proceed similarly on the descent phase but it did not provide much improvement in our tests, so we did not use it in the end. The reason for this is unclear, maybe the variability is greater in the descent phase than in the climb phase, maybe the descent phase is at the end of the flight, so the relation with the TOW is looser.

#### Features for the Whole Flight Temporal Profile (`feature_cruise_infos.py`)
Here, trajectories are "aligned" by considering scaled temporal slices computed using $\left({\mathrm{timestamp}_i-t\_{adep}}\right)/\left({t\_{ades}-t\_{adep}}\right)$ which can be seen as the completion percentage, the slice [5%,10%] is at the begining of the flight whereas [90%,95%] is at the end of the flight. Due to the scaling, the first part and the last part are not well "aligned": considering two flights, for instance a 1-hour flight and a 5-hour flight, the statistics computed on the [5%-10%] slice does not represent the same concept as they are likely not in the same flight phase. However, statistics in the [45%;50%] slice represent somewhat the same concept, they are both in the same flight phase and mid-flight. These "mis-alignements" on the first slices might not be too critical as we already have features dealing with the climbing phase.
Considering points inside a given slice, our features are obtained by computing the following statistics on the points within the slice:
- ${t\_{ades}-t\_{adep}}$ - the scaling factor and flight duration
- $\mathrm{Cardinal}\left(\mathrm{slice}\right)$ - number of points in the slice
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{Mach}_i$
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{altitude}_i$
- $\mathrm{altitude}_{last(slice)}-\mathrm{altitude}\_{first(slice)}$

We do that for 20 slices starting from the slice [0,5%] to the slice [95%,100%]. These features are mostly designed to capture the information in the cruise phase.

#### Features for the Wind Effect (`feature_wind_effect.py`)
We compute the average value of the wind projected onto the ground speed
$\mathrm{dot}\left(\vec{wind},\vec{groundspeed}\right)/ \lVert \vec{groundspeed} \rVert$
along the trajectory. It quantifies if there is a tail wind, "helping" along the
flight path, on average, of if there is a head wind.

## Training the Model (`regression.py`,`features.py`,`sklearnutils.py`,`optimparam.py`)
The model was trained using
[LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html). Hyperparameters
were obtained by doing a random search using a validation set. It is implemented in `optimparam.py`.

The variable TOW to be predicted was scaled using
(TOW-EOW)/(MTOW-EOW). This way, the range of the variable to be
predicted is roughly the same and mostly inside [0.2,1]. This scaling
avoids the training process to consume a lot of splits of `aircraft_type` and `wtc` just to get the
mass range right. However a 10% relative error on an
A320 or a A343 does not produce the same absolute mass error. To still
optimize the root mean square error on the "absolute" mass, the LightGBM
was trained with a vector of weights equal to the squared scaling term: (MTOW-EOW)**2.


In "vanilla"[^bignote] tree models like the ones used in LightGBM, any
strictly monotonic transformation of the variables does not change the
split choices and hence the predictions. However, applying different
scalings on different groups of examples does change the
split choices. The estimated mass explanatory variables were scaled according to their
aircraft type using the formula: (mass-EOW)/(MTOW-EOW). Regardless of
the aircraft type, a value of 1
is associated with an heavy aircraft. Furthermore this aligns
well with the predicted variable's scaling.

<!-- This "scaling by group" principle should also -->
<!-- be useful on other variables like the distance_flown for instance: a 1,500km trip is -->
<!-- a short or long flight depending on the aircraft type, so scaling the -->
<!-- distance flown according the aircraft type should make sense. This was -->
<!-- tested with the ScaleByGroup class. Despite not being actually used -->
<!-- this class was not deleted from the final code We left this in the final code, maybe -->

<!-- As the training process makes -->
<!-- random choices to build the model, the model can be seen as a random -->
<!-- model. One can decompose Running the same algorithm with a different seed -->
<!-- produces a different model. Bagging is a -->
<!-- special case of this. -->

# Main External Softwares
This work builds on some external software:
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [sklego](https://github.com/koaning/scikit-lego)
- [csaps](https://csaps.readthedocs.io/en/latest/)
- [Traffic](https://github.com/xoolive/traffic)
- [OpenAP](https://github.com/TUDelft-CNS-ATM/openap)
- [Pitot](https://github.com/open-aviation/pitot)
# External Data
This work uses external data in the public domain, namely:
- airports.csv file from [ourairports.com](https://ourairports.com)
- metar files from [https://mesonet.agron.iastate.edu/ASOS/](https://mesonet.agron.iastate.edu/ASOS/)

And Wikipedia pages for the MTOW and EOW not inside OpenAP.



[^note1]: Think of the bias-variance decomposition and its
usage in bagging. This decomposition is valid with randomness in
general not just randomness introduced by bagging.

[^bignote]: "Vanilla" here refers to tree with simple split condition where one variable is
compared to a constant threshold: $X_i<Threshold$

