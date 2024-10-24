# PRC Data Challenge
This challenge aimed at estimating the Take Off Weight (TOW) of an aircraft from its flight and trajectory information, more info at [PRC Challenge](https://ansperformance.eu/study/data-challenge/). Please bear in mind that **all** this repository (documentation included) was done during a competition and hence was done in a limited time.


# The Results

| Model   |     Training time      |  RMSE on the final\_submission\_set |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |

# To Reproduce the Results
First edit the variable `FOLDER_DATA` in the `CONFIG` file, then just run the command below. Please be aware that it might take some time. To reduce this time depending on your computer you might want to use the option `-j` of `make`. For instance, `make -j4 cleantrajectories`, will launch 4 processes in parallel. In this whole project, each process takes no more than approximately 20GB of RAM. The only process that takes more is the training but this training phase does not use the `-j` of the `make` to run in parallel.
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
corresponding to a climb (or a descent) of 1000 ft/min, to compute an estimate of the departure (resp. arrival) time. A lower radius than 10 NM leaves a lot of trajectories without a
corrected date. In cases where no points are found
inside the 10 NM radius, we keep the departure and arrival date
provided in the flight data.
### Features derived from METARs at departure and arrival airports (`correct_dates.py`)
At departure and arrival airports, METARs are not always available in our files. To solve this issue, we build a `sklearn.neighbors.NearestNeighbors` "model" to query the METAR closest (in the space-time domain) to the wanted one.
### Features for Thunder/Fog at arrival airports (`correct_dates.py`)
We expect thunder and fog to be related to holding patterns. If such bad weather predictions are available to the crew before departure, we can expect that they might take some extra fuel on-board, thus increasing the mass. However, taking just the METAR arrival airport at the arrival time might be too fine-grained. So, using a `sklearn.neighbors.NearestNeighbors` "model", we take all the METARs in a given space-time radius around the arrival airport at arrival time, and we summarize it. Not knowing, a priori, the good space-time radius, we did that for several radius values, letting the training algorithm select the relevant ones (hopefully).
### Features Extracted from the trajectories
One difficulty to extract features from trajectories is that a given
feature should represent the same concept from one trajectory to another, a concept hopefully related to the TOW.

#### Features for the climbing Phase (`correct_dates.py`)
Here, trajectories are "aligned" by considering altitude slices: for instance we compute a feature for the observed ROCD between 10500ft and 11500ft, hopefully capturing the concept "climbing performance at 10000ft". Of course, this climbing segment might take place sooner or later during the climb, depending on the flight (and of the aircraft's weight). We hope to capture this by adding a temporal feature for the slice. These features are computed for 48 slices, each of height 1000ft, starting from [-500ft,500ft] to [46500ft,47500ft]. In detail, considering points inside a given slice, we compute our features by doing statistics on points in this slice:
- $\mathrm{Cardinal}\left(\mathrm{slice}\right)$ - number of points in the slice
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{ROCD}_i$
- $max_j ROCD_j - min_i ROCD_i$
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{ROCD}_i$
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

#### Features for the Whole Flight Temporal Profile (`correct_dates.py`)
Here, trajectories are "aligned" by considering scaled temporal slices computed using $\left({\mathrm{timestamp}_i-t\_{adep}}\right)/\left({t\_{ades}-t\_{adep}}\right)$ which can be seen as the completion percentage, the slice [5%,10%] is at the begining of the flight whereas [90%,95%] is at the end of the flight. Due to the scaling, the first part and the last part are not well "aligned": considering two flights, for instance a 1-hour flight and a 5-hour flight, the statistics computed on the [5%-10%] slice does not represent the same concept as they are likely not in the same flight phase. However, statistics in the [45%;50%] slice represent somewhat the same concept, they are both in the same flight phase and mid-flight. These "mis-alignements" on the firsts slices might not be too critical as we already have features dealing with the climbing phase.
Considering points inside a given slice, we compute our features whihch would simply statistics on points in this slice:
- ${t\_{ades}-t\_{adep}}$ - the scaling factor and flight duration
- $\mathrm{Cardinal}\left(\mathrm{slice}\right)$ - number of points in the slice
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{Mach}_i$
- $\mathrm{median}_{\mathrm{i}\in\mathrm{slice}}~\mathrm{altitude}_i$
- $\mathrm{altitude}_{last(slice)}-\mathrm{altitude}\_{first(slice)}$

We do that for 20 slices starting from the slice [0,5%] to the slice [95%,100%]. These features are mostly designed to capture the information in the cruise phase.

#### Features for the Wind Effect (`feature_wind_effect.py`)
We computed the average value of
$\mathrm{dot}\left(\vect{Wind},\vect{Groundspeed}\right)/\vect{Groundspeed}$
along the trajectory.

## Training the Model
The model was trained using [LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html).MTOW-EOW Scaling. Hyperparameters were obtained by doing a random search implemented in `optimparam.py`. Using these hyperparameters, we built 10 models using 10 different random seeds. The final model is the average of these 10 models with 50000 trees each.

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
- Wikipedia pages for the MTOW and EOW not inside OpenAP
