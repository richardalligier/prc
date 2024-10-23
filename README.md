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
### Features derived from METARs at departure and arrival airports (`correct_dates.py`)
At departure and arrival airports, METARs are not always available in our files. To solve this issue, we build a `sklearn.neighbors.NearestNeighbors` "model" to query the METAR closest (in the space-time domain) to the wanted one.
### Features for Thunder/Fog at arrival airports (`correct_dates.py`)
We expect that thunder and fog to be related to holding patterns. If the crew can expect such weather maybe they take more fuel. However, taking just the METAR arrival airport at the arrival time might be too fine grained. So, using a `sklearn.neighbors.NearestNeighbors` "model", we took all the METARs in a given space-time radius around the arrival airport at arrival time, and we summarize it. Not knowing, a priori, the good space-time radius, we did that for serveral radius value, the training algorithm will select the relevant ones, hopefully.
### Features Extracted from the trajectories
One difficulty to extract features from trajectories is that a given feature should represent the same concept from one trajectory to an other, a concept hopefully related to the TOW.
#### Features for the climbing Phase (`correct_dates.py`)
Here, trajectories are "aligned" by considering altitude slices: we expect that the ROCD taken between 1500ft and 2500ft to
#### Features for the Whole Flight Temporal Profile (`correct_dates.py`)
Here, trajectories are "aligned" by considering scaled temporal slices: time
## Training the Model
The model was trained using [LightGBM](https://lightgbm.readthedocs.io/en/stable/index.html)
# External Data
This work uses external data in the public domain, namely:
- airports.csv file from [ourairports.com](https://ourairports.com)
- metar files from [https://mesonet.agron.iastate.edu/ASOS/](https://mesonet.agron.iastate.edu/ASOS/)
