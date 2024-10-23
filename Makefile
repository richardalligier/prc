
.PHONY: download cleantrajectories features submission
.SECONDARY:

include CONFIG

#### SMOOTHING SPLINE PARAMETER
INTERPOL_SMOOTH = 1e-2

#### Climbing slicing parameter
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

#### Number of slices for the cruise features
CRUISE_NSPLIT = 20

#### use the MyFilterDerivative
PREFIX_FILTER = classic

PREFIX_INTERPOL = $(PREFIX_FILTER)__$(INTERPOL_SMOOTH)
PREFIX_MASS = $(PREFIX_INTERPOL)__$(CLMB_PERIODS)_$(CLMB_THRESHOLD_VR)_$(CLMB_THRESHOLD_DT)_$(CLMB_VRATE_VAR)_$(CLMB_CTHRUST)_$(CLMB_ALT_START)_$(CLMB_ALT_STEP)
PREFIX_CRUISE = $(PREFIX_INTERPOL)__$(CRUISE_NSPLIT)

FOLDER_RAW = $(FOLDER_DATA)/rawtrajectories
FOLDER_FLGT = $(FOLDER_DATA)/flights
FOLDER_WEATHER = $(FOLDER_DATA)/weather
FOLDER_THUNDER = $(FOLDER_DATA)/thunder
FOLDER_WIND = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_wind
FOLDER_CRUISE = $(FOLDER_DATA)/$(PREFIX_CRUISE)_cruise
FOLDER_FILT = $(FOLDER_DATA)/$(PREFIX_FILTER)_filtered_trajectories
FOLDER_INT = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_interpolated_trajectories
FOLDER_MASS = $(FOLDER_DATA)/$(PREFIX_MASS)_masses



FLIGHT_FILES = challenge_set final_submission_set



TRAJS_SRC = $(shell ls $(FOLDER_RAW) )
#2022-11-04.parquet


TRAJS = $(foreach f,$(TRAJS_SRC),$(FOLDER_INT)/$(f))

FLIGHTS = $(foreach f,$(FLIGHT_FILES),$(FOLDER_FLGT)/$(f).parquet)


AIRPORTS = $(FOLDER_DATA)/airports_tz.parquet
MASSES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_MASS)/$(flight)/$(f)))
WINDS = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_WIND)/$(flight)/$(f)))
CRUISES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_CRUISE)/$(flight)/$(f)))
WEATHERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_WEATHER)/$(flight).parquet)
THUNDERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_THUNDER)/$(flight).parquet)


download: $(FLIGHTS) $(foreach f,$(shell mc ls dc24/competition-data  | rev | cut -d' ' -f1 | rev | grep "parquet" | grep "2022-11-04"),$(FOLDER_RAW)/$(f))
	mkdir -p $(FOLDER_DATA)/METARs
	python3 download_METARs.py

cleantrajectories: $(TRAJS)

features:  $(CRUISES)
#$(CRUISES)
#$(WEATHERS) $(WINDS) $(THUNDERS) $(MASSES)



define add_mass
	python3 add_mass.py -is_climb -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_MASS)/%/$(@F),%,$@).parquet -f_out $@ -periods $(CLMB_PERIODS) -thresh_dt $(CLMB_THRESHOLD_DT) -threshold_vr $(CLMB_THRESHOLD_VR) -cthrust $(CLMB_CTHRUST) -vrate_var $(CLMB_VRATE_VAR) -altstep $(CLMB_ALT_STEP)  -altstart $(CLMB_ALT_START) -airports $(AIRPORTS)
endef

define add_cruise
	python3 feature_cruise_infos.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_CRUISE)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(AIRPORTS) -nsplit $(CRUISE_NSPLIT)
endef

define add_wind
	python3 add_wind_effect.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_WIND)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(AIRPORTS)
endef


$(AIRPORTS):
#	curl -o $(FOLDER_DATA)/airports.csv https://github.com/davidmegginson/ourairports-data/blob/main/airports.csv
	python3 add_timezone.py -a_in airports.csv -a_out $@  -flights "$(FLIGHTS)"


$(FOLDER_DATA)/METARs.parquet: $(AIRPORTS)
	python metars_folder_to_parquet.py to_parquet -airports $(AIRPORTS) -metars_folder_in $(FOLDER_DATA)/METARs -metars_parquet_out $@

$(FOLDER_WEATHER)/%.parquet: $(FOLDER_FLGT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	python3 add_weather_from_metars.py -f_in $^ -airports $(AIRPORTS) -metars $(FOLDER_DATA)/METARs.parquet -f_out $@ -geo_scale 1 -hour_scale 1


$(FOLDER_THUNDER)/%.parquet: $(FOLDER_FLGT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	python3 add_thunder_from_metars.py -f_in $^ -airports $(AIRPORTS) -metars $(FOLDER_DATA)/METARs.parquet -f_out $@ -geo_scale 1 -hour_scale 1

$(FOLDER_FLGT)/%.parquet:
	mkdir -p $(@D)
	mc cp  dc24/competition-data/$(basename $(@F)).csv $(FOLDER_FLGT)/$(basename $(@F)).csv
	python3 flights_to_parquet.py -f_in $(@:.parquet=.csv) -f_out $@

$(FOLDER_RAW)/%.parquet:
	mkdir -p $(@D)
	mc cp  dc24/competition-data/$(@F) $@

$(FOLDER_FILT)/%.parquet: $(FOLDER_RAW)/%.parquet
	@mkdir -p $(@D)
	python3 filter_trajs.py -t_in $^ -t_out $@ -strategy classic

$(FOLDER_INT)/%.parquet: $(FOLDER_FILT)/%.parquet
	@mkdir -p $(@D)
	python3 interpolate.py -t_in $^ -t_out $@ -smooth $(INTERPOL_SMOOTH)

$(FOLDER_MASS)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_mass)

$(FOLDER_MASS)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_mass)


$(FOLDER_CRUISE)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_cruise)

$(FOLDER_CRUISE)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet  $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_cruise)


$(FOLDER_WIND)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_wind)

$(FOLDER_WIND)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet $(AIRPORTS)
	@mkdir -p $(@D)
	$(call add_wind)
