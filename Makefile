
.PHONY: download cleantrajectories features submissions
.SECONDARY:

include CONFIG

#### which filter to use
PREFIX_FILTER = likablejelly

#### SMOOTHING SPLINE PARAMETER
INTERPOL_SMOOTH = 1e-2

#### Climbing slicing parameter
# wind power in energy rate
WIND_ERATE = yes
# temperature correction in energy rate
TEMP_ERATE = yes

#CLMB_INCLUDETAKEOFF = yes
# energyrate(i+CLMB_PERIODS)-energyrate(i)
CLMB_PERIODS = 5
# keep points with vertical_rate > CLMB_THRESHOLD_VR
CLMB_THRESHOLD_VR = 500
# set thrust command
CLMB_CTHRUST = 1
# variable used as vertical_rate
CLMB_VRATE_VAR = daltitude
# if timestamp(i+CLMB_PERIODS)-timestamp(i) > CLMB_THRESHOLD_DT, then thowaway
CLMB_THRESHOLD_DT = 40
# start of the slice -0.5=-500ft
CLMB_ALT_START = -0.5
# step for slicing 1=1000ft
CLMB_ALT_STEP = 1
# step for slicing 1=1000ft
CLMB_ALT_STOP = 48

#### Number of slices for the cruise features
CRUISE_NSPLIT = 20

VARIABLES = oefmntwc


PREFIX_INTERPOL = $(PREFIX_FILTER)__$(INTERPOL_SMOOTH)
PREFIX_MASS = $(PREFIX_INTERPOL)__$(CLMB_PERIODS)_$(CLMB_THRESHOLD_VR)_$(CLMB_THRESHOLD_DT)_$(CLMB_VRATE_VAR)_$(CLMB_CTHRUST)_$(CLMB_ALT_START)_$(CLMB_ALT_STOP)_$(CLMB_ALT_STEP)_$(WIND_ERATE)_$(TEMP_ERATE)
PREFIX_CRUISE = $(PREFIX_INTERPOL)__$(CRUISE_NSPLIT)
PREFIX_WIND = $(PREFIX_INTERPOL)
PREFIX_SUB = $(PREFIX_INTERPOL)_$(PREFIX_MASS)_$(PREFIX_CRUISE)_$(PREFIX_WIND)

FOLDER_RAW = $(FOLDER_DATA)/rawtrajectories
FOLDER_FLGT = $(FOLDER_DATA)/flights
FOLDER_WEATHER = $(FOLDER_DATA)/weather
FOLDER_THUNDER = $(FOLDER_DATA)/thunder
FOLDER_WIND = $(FOLDER_DATA)/$(PREFIX_WIND)_wind
FOLDER_CRUISE = $(FOLDER_DATA)/$(PREFIX_CRUISE)_cruise
FOLDER_FILT = $(FOLDER_DATA)/$(PREFIX_FILTER)_filtered_trajectories
FOLDER_INT = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_interpolated_trajectories
FOLDER_MASS = $(FOLDER_DATA)/$(PREFIX_MASS)_masses

FLIGHT_FILES = train ranking_test_final
#flight_list train



TRAJS_SRC = $(shell ls $(FOLDER_RAW))


TRAJS = $(foreach f,$(TRAJS_SRC),$(FOLDER_INT)/$(f))

FLIGHTS = $(foreach f,$(FLIGHT_FILES),$(FOLDER_FLGT)/$(f).parquet)


AIRPORTS = $(FOLDER_DATA)/airports_tz.parquet
METARS = $(FOLDER_DATA)/METARs.parquet
MASSES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_MASS)/$(flight)/$(f)))
WINDS = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_WIND)/$(flight)/$(f)))
CRUISES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_CRUISE)/$(flight)/$(f)))
WEATHERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_WEATHER)/$(flight).parquet)
THUNDERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_THUNDER)/$(flight).parquet)

$(FOLDER_DATA)/ro-crate-metadata.json:
	curl $(DATAREPO) > $(FOLDER_DATA)/ro-crate-metadata.json

download: $(foreach f,$(shell python3 generate_dates.py),$(FOLDER_RAW)/$(f)) $(METARS) $(FLIGHTS) 

flights: $(FLIGHTS)
cleantrajectories: $(TRAJS)

features: $(CRUISES) $(MASSES) $(WINDS) $(WEATHERS) $(THUNDERS)
#$(WEATHERS) $(THUNDERS) $(WINDS)
RANDOMS = $(shell seq 0 9 | tr '\n' ' ')
SCALER = mtow
submissions: $(foreach random,$(RANDOMS),$(SUBMISSIONS_FOLDER)/$(PREFIX_SUB)/submit-train-ranking_test_final-$(VARIABLES)-$(random)-$(SCALER)-predictions.csv)
#	mkdir -p $(SUBMISSIONS_FOLDER)
#	for number in $(shell seq 0 19); do \
#		python3 regression.py -what submit -random_state $$number; \
#	done;
#	python3 average_prediction.py -istop 10 -out_csv $(SUBMISSIONS_FOLDER)/averaged_10.csv
#	python3 average_prediction.py -istop 20 -out_csv $(SUBMISSIONS_FOLDER)/averaged_20.csv


$(SUBMISSIONS_FOLDER)/$(PREFIX_SUB)/%predictions.csv: #$(CRUISES) $(MASSES) $(WINDS) $(WEATHERS) $(THUNDERS)
	@mkdir -p $(@D)
	$(eval BASE := $*)
	@echo "what $(BASE)"
	$(eval PARTS := $(subst -, ,$(BASE)))
	$(eval WHAT:=$(word 1,$(PARTS)))
	$(eval TRAIN:=$(word 2,$(PARTS)))
	$(eval TEST:=$(word 3,$(PARTS)))
	$(eval VARS:=$(word 4,$(PARTS)))
	$(eval RANDOM_STATE:=$(word 5,$(PARTS)))
	$(eval SCALE:=$(word 6,$(PARTS)))
	python3 regression.py -what $(WHAT) -train $(TRAIN) -submit $(TEST) -variables $(VARS) -scaler $(SCALE) -random_state $(RANDOM_STATE) -prefix_mass $(PREFIX_MASS) -prefix_cruise $(PREFIX_CRUISE) -prefix_wind $(PREFIX_WIND) -folderout $(@D)


define feature_climbing
	python3 feature_climbing.py -is_climb -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_MASS)/%/$(@F),%,$@).parquet -f_out $@ -periods $(CLMB_PERIODS) -thresh_dt $(CLMB_THRESHOLD_DT) -threshold_vr $(CLMB_THRESHOLD_VR) -cthrust $(CLMB_CTHRUST) -vrate_var $(CLMB_VRATE_VAR) -altstep $(CLMB_ALT_STEP)  -altstart $(CLMB_ALT_START) -altstop $(CLMB_ALT_STOP) -airports $(AIRPORTS) -wind_effect $(WIND_ERATE) -temperature_effect $(TEMP_ERATE)
endef

define feature_cruise
	python3 feature_cruise_infos.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_CRUISE)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(AIRPORTS) -nsplit $(CRUISE_NSPLIT)
endef

define feature_wind
	python3 feature_wind_effect.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_WIND)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(AIRPORTS)
endef


$(AIRPORTS): $(FLIGHTS)
#	curl -o $(FOLDER_DATA)/airports.csv https://github.com/davidmegginson/ourairports-data/blob/main/airports.csv
	python3 airports_to_parquet.py -a_in ourairports2024-10-21.csv -a_out $@  -flights "$(FLIGHTS)"


$(METARS): #$(AIRPORTS)
	# This link has been deleted the 23 June 2025
	# gdown 'https://drive.google.com/uc?export=download&id=1udmsuT317LECvr1JJNEmhdp0bM2OGq9W' -O $@
	# This link replaces the link above
	gdown 'https://drive.google.com/uc?export=download&id=1XlK1NHySQ9Z9FksG1GhyPiVpFOMrA4ue' -O $@
	# uncomment below to generate it from scratch
	# result might be different as mesonet's files might have been updated
	# I've experienced one station's location update in a 2 weeks timespan
	# mkdir -p $(FOLDER_DATA)/METARs
	# python3 download_metars.py
	# python3 metars_folder_to_parquet.py -metars_folder_in $(FOLDER_DATA)/METARs -metars_parquet_out $@

$(FOLDER_WEATHER)/%.parquet: $(FOLDER_FLGT)/%.parquet $(AIRPORTS) $(METARS)
	@mkdir -p $(@D)
	python3 feature_weather_from_metars.py -f_in $< -airports $(AIRPORTS) -metars $(METARS) -f_out $@ -geo_scale 1 -hour_scale 1


$(FOLDER_THUNDER)/%.parquet: $(FOLDER_FLGT)/%.parquet $(AIRPORTS) $(METARS)
	@mkdir -p $(@D)
	python3 feature_thunder_from_metars.py -f_in $< -airports $(AIRPORTS) -metars $(METARS) -f_out $@ -geo_scale 1 -hour_scale 1

$(FOLDER_FLGT)/%.parquet: $(FOLDER_FLGT)/%.csv
	mkdir -p $(@D)
	python3 flights_to_parquet.py -f_in $(@:.parquet=.csv) -f_out $@

$(FOLDER_RAW)/%.parquet: $(FOLDER_DATA)/ro-crate-metadata.json
	mkdir -p $(@D)
	python3 download_competition_data.py -json $^ -filename $(@F) -out $@

$(FOLDER_FILT)/%.parquet: $(FOLDER_RAW)/%.parquet
	@mkdir -p $(@D)
	python3 filter_trajs.py -t_in $< -t_out $@ -strategy $(PREFIX_FILTER)

$(FOLDER_INT)/%.parquet: $(FOLDER_FILT)/%.parquet
	@mkdir -p $(@D)
	python3 interpolate.py -t_in $< -t_out $@ -smooth $(INTERPOL_SMOOTH)

$(FOLDER_MASS)/flight_list/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_climbing)

$(FOLDER_MASS)/ranking_test_final/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_climbing)

$(FOLDER_MASS)/train/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_climbing)


$(FOLDER_CRUISE)/flight_list/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_cruise)

$(FOLDER_CRUISE)/ranking_test_final/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_cruise)

$(FOLDER_CRUISE)/train/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_cruise)

$(FOLDER_WIND)/flight_list/%.parquet: $(FOLDER_INT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_wind)

$(FOLDER_WIND)/ranking_test_final/%.parquet: $(FOLDER_INT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_wind)

$(FOLDER_WIND)/train/%.parquet: $(FOLDER_INT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	$(call feature_wind)
