# PRC Data Challenge
This challenge aimed at estimating the Take Off Weight (TOW) of an aircraft from its flight and trajectory information, more info at [PRC Challenge](https://ansperformance.eu/study/data-challenge/). Please bear in mind that **all** this repository (documentation included) was done during a competition and hence was done in a limited time.

# To Reproduce the Results
First edit the variable `FOLDER_DATA` in the `CONFIG` file, then just run the command below. Please be aware that it might take some time. To reduce this time depending on your computer you might want to use the option `-j` of `make`. For instance, `make -j4 cleantrajectories`, will launch 4 processes in parallel. In this whole project, each process takes no more than approximately 20GB of RAM. The only process that takes more is the training but this training phase does not use the `-j` of the `make` to run in parallel.
```
make download && make cleantrajectories && make features && make submissions
```
# Technical Aspects
In this Section you will find technical aspects and motivations for the code in this repository.
## Filtering and Smoothing Trajectories
Informations on the weight hides inside the kinematic of the aircraft. Thus, we want to have a good fine grain vision of its kinematic, leading us to filter and smooth the trajectory.
### Filtering Trajectory (`filter_trajs.py`)
### Smoothing Trajectory (`interpolate.py`)
The smoothing is done using smoothing splines, and we are careful about not interpolating in area where there are not points. More specifically, we will not interpolated between two points spaced by more than 20 seconds.
## Building Features
### Correcting dateTimes (`correct_dates.py`)
When looking at trajectories, the `arrival_time` variable is not always correct, it is the same for the departure.
### Features derived from METARs at departure and arrival airports (`correct_dates.py`)
At departure and arrival airports, METARs are not always available in our files. To solve this issue, we build a `sklearn.neighbors.NearestNeighbors` "model" to query the METAR closest (in the space-time domain) to the wanted one.
### Features for Thunder/Fog at arrival airports (`correct_dates.py`)
We expect that thunder and fog to be related to holding patterns. If the crew can expect such weather maybe they take more fuel. However, taking just the METAR arrival airport at the arrival time might be too fine grained. So, using a `sklearn.neighbors.NearestNeighbors` "model", we took all the METARs in a given space-time radius around the arrival airport at arrival time, and we summarize it. Not knowing, a priori, the good space-time radius, we did that for serveral radius value, the training algorithm will select the relevant ones, hopefully.
### Features Extracted from the trajectories
One difficulty to extract features from trajectories is that a given feature should represent the same concept from one trajectory to an other, a concept hopefully related to the TOW.
#### Features for the climbing Phase (`correct_dates.py`)
Here, trajectories are "aligned" by considering altitude slices: we expect that the ROCD taken between 10500ft and 11500ft from one flight, capturing the concept "climbing performance at 10000ft", regardless when it happens during the flight. We do this for 48 slices of 1000ft height, starting from [-500ft,500ft] to [46500ft,47500ft]. Considering points inside a given slice, we compute our features whihch would simply statistics on points in this slice: 
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

We do that for all the 48 slices, creating us hundreds of features. The last feature $\mathrm{mass}_i$ is computed by solving a polynomial equations of degree 2 built using a physical model (see [publication on this matter](https://enac.hal.science/hal-00911686/file/Alligier_ATMSeminar2013.pdf)). To do this efficiently, we used the open physical model [OpenAP](https://github.com/TUDelft-CNS-ATM/openap) and a module `polynomials.py` was written, modeling a set of polynomials as numpy arrays, leveraging numpy's broadcasting. The mass feature is computed by aggregating masses with a median operator. Maybe it would be better to take all the points together and solve a 4th degree polynomials as in the publication linked above. This path was not followed because in hindsight these masses variables might not be that useful as the energyrate contains a very similar information. The energyrate variable is the variation of the energy over the mass, one can compute it using only kinematic information as illustrated in the formula in this 2018's [code](https://github.com/richardalligier/trc2018/blob/master/OCaml/csvaddenergyrate.ml).

One can proceed similarly on the descent phase but it did not provide much improvement in our test, so we did not used it in the end. The reason for this is unclear, maybe the variability is greater in the descent phase than in the climb phase, maybe the descent phase is at the end of the flight, so the relation with the TOW is looser.
#### Features for the Whole Flight Temporal Profile (`correct_dates.py`)
Here, trajectories are "aligned" by considering scaled temporal slices computed using $\left({\mathrm{timestamp}_i-t\_{adep}}\right)/\left({t\_{ades}-t\_{adep}}\right)$ which can be seen as the completion percentage, the slice [5%,10%] is at the begining of the flight whereas [90%,95%] is at the end of the flight. Due to the scaling, the first part and the last part are not well "aligned": considering two flights, for instance a 1-hour flight and a 5-hour flight, the statistics computed on the [5%-10%] slice, does not represent the same concept as they are likely not in the same flight phase. However, statistics in the [45%;50%] slice represent somewhat the same concept, they are both in the same flight phase and mid-flight. These "mis-alignements" on the firsts slices might not be too critical as we already have features dealing with the climbing phase.
Considering points inside a given slice, we compute our features whihch would simply statistics on points in this slice: 

## Training the Model
The model was trained using [LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html)

MTOW-EOW Scaling

Random Search

10 random seed

# Main External Softwares
This work builds on some external software:
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenAP](https://github.com/TUDelft-CNS-ATM/openap)
- [Pitot](https://github.com/open-aviation/pitot)
# External Data
This work uses external data in the public domain, namely:
- airports.csv file from [ourairports.com](https://ourairports.com)
- metar files from [https://mesonet.agron.iastate.edu/ASOS/](https://mesonet.agron.iastate.edu/ASOS/)
- Wikipedia pages for the MTOW and EOW not inside OpenAP
