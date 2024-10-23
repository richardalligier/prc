
.PHONY: download cleantrajectories features submission
.SECONDARY:

include CONFIG

# SMOOTHING SPLINE PARAMETER
INTERPOL_SMOOTH = 1e-2

# SMOOTHING SPLINE PARAMETER
CLMB_PERIODS = 5
CLMB_THRESHOLD_VR = 500
CLMB_CTHRUST = 1
CLMB_VRATE_VAR = daltitude
CLMB_THRESHOLD_DT = 40
CLMB_ALT_START = -0.5
CLMB_ALT_STEP = 1

DESC_PERIODS = 5
DESC_THRESHOLD_VR = 1500
DESC_CTHRUST = 0.3
DESC_VRATE_VAR = daltitude
DESC_THRESHOLD_DT = 40
DESC_ALT_START = -0.5
DESC_ALT_STEP = 1

VECT_PERIODS = 5
VECT_THRESHOLD_VR = 500
VECT_THRESHOLD_DT = 40
VECT_VRATE_VAR = daltitude
VECT_ALT_START = -0.5
VECT_ALT_STEP = 1

CRUISE_NSPLIT = 20

PREFIX_FILTER = classic
PREFIX_INTERPOL = $(PREFIX_FILTER)__$(INTERPOL_SMOOTH)
PREFIX_MASS = $(PREFIX_INTERPOL)__$(CLMB_PERIODS)_$(CLMB_THRESHOLD_VR)_$(CLMB_THRESHOLD_DT)_$(CLMB_VRATE_VAR)_$(CLMB_CTHRUST)_$(CLMB_ALT_START)_$(CLMB_ALT_STEP)
PREFIX_MASSDESC = $(PREFIX_INTERPOL)__$(DESC_PERIODS)_$(DESC_THRESHOLD_VR)_$(DESC_THRESHOLD_DT)_$(DESC_VRATE_VAR)_$(DESC_CTHRUST)_$(DESC_ALT_START)_$(DESC_ALT_STEP)
PREFIX_VECT = $(PREFIX_INTERPOL)__$(VECT_PERIODS)_$(VECT_THRESHOLD_VR)_$(VECT_THRESHOLD_DT)_$(VECT_VRATE_VAR)_$(VECT_ALT_START)_$(VECT_ALT_STEP)
PREFIX_CRUISE = $(PREFIX_INTERPOL)__$(CRUISE_NSPLIT)

FOLDER_RAW = $(FOLDER_DATA)/rawtrajectories
FOLDER_ICAOS = $(FOLDER_DATA)/icaos
FOLDER_FLGT = $(FOLDER_DATA)/flights
FOLDER_WEATHER = $(FOLDER_DATA)/weather
FOLDER_THUNDER = $(FOLDER_DATA)/thunder
FOLDER_DATES = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_corrected_dates
FOLDER_WIND = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_wind
FOLDER_CRUISE = $(FOLDER_DATA)/$(PREFIX_CRUISE)_cruise
FOLDER_FILT = $(FOLDER_DATA)/$(PREFIX_FILTER)_filtered_trajectories
FOLDER_INT = $(FOLDER_DATA)/$(PREFIX_INTERPOL)_interpolated_trajectories
FOLDER_MASS = $(FOLDER_DATA)/$(PREFIX_MASS)_masses
FOLDER_MASSDESC = $(FOLDER_DATA)/$(PREFIX_MASSDESC)_massesdesc
FOLDER_VECTORIZED = $(FOLDER_DATA)/$(PREFIX_VECT)_vectorized_trajectories



FLIGHT_FILES = challenge_set final_submission_set



TRAJS_SRC = $(shell ls $(FOLDER_RAW) )
#2022-11-04.parquet
#2022-01-01.parquet
#$(shell ls $(FOLDER_RAW))
#$(shell mc ls dc24/competition-data  | rev | cut -d' ' -f1 | rev | grep "parquet" )#| head -n 1)
#2022-11-04.parquet
#$(shell mc ls dc24/competition-data  | rev | cut -d' ' -f1 | rev | grep "parquet" )#| head -n 1)
#TRAJS_SRC = $(shell ls $(FOLDER_RAW)  | rev | cut -d' ' -f1 | rev | grep "parquet") # | sort -r -n)

TRAJS = $(foreach f,$(TRAJS_SRC),$(FOLDER_INT)/$(f))

FLIGHTS = $(foreach f,$(FLIGHT_FILES),$(FOLDER_FLGT)/$(f).parquet)


MASSES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_MASS)/$(flight)/$(f)))
MASSESDESC = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_MASSDESC)/$(flight)/$(f)))


VECTORIZED = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_VECTORIZED)/$(flight)/$(f)))

ICAOS = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_ICAOS)/$(flight)/$(f)))
DATES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_DATES)/$(flight)/$(f)))
WINDS = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_WIND)/$(flight)/$(f)))
CRUISES = $(foreach flight,$(FLIGHT_FILES), $(foreach f,$(TRAJS_SRC),$(FOLDER_CRUISE)/$(flight)/$(f)))
WEATHERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_WEATHER)/$(flight).parquet)
THUNDERS = $(foreach flight,$(FLIGHT_FILES), $(FOLDER_THUNDER)/$(flight).parquet)

AIRPORTS = $(FOLDER_DATA)/airports_tz.parquet
#$(FOLDER_DATA)/METARs.parquet
# $(FOLDER_DATA)/METARs.parquet $(FOLDER_DATA)/metars_ident_station.parquet

download: $(FLIGHTS) $(foreach f,$(shell mc ls dc24/competition-data  | rev | cut -d' ' -f1 | rev | grep "parquet" | grep "2022-11-04"),$(FOLDER_RAW)/$(f))
	mkdir -p $(FOLDER_DATA)/METARs
	python3 download_METARs.py

cleantrajectories: $(TRAJS)

features:  $(MASSES)
#$(CRUISES)
#$(WEATHERS) $(WINDS) $(THUNDERS) $(MASSES)



define add_mass
	python3 add_mass.py -is_climb -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_MASS)/%/$(@F),%,$@).parquet -f_out $@ -periods $(CLMB_PERIODS) -thresh_dt $(CLMB_THRESHOLD_DT) -threshold_vr $(CLMB_THRESHOLD_VR) -cthrust $(CLMB_CTHRUST) -vrate_var $(CLMB_VRATE_VAR) -altstep $(CLMB_ALT_STEP)  -altstart $(CLMB_ALT_START) -airports $(FOLDER_DATA)/airports_tz.parquet
endef

define add_mass_descent
	python3 add_mass.py -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_MASSDESC)/%/$(@F),%,$@).parquet -f_out $@ -periods $(DESC_PERIODS) -thresh_dt $(DESC_THRESHOLD_DT) -threshold_vr "-$(DESC_THRESHOLD_VR)" -prefix "desc" -cthrust $(DESC_CTHRUST) -vrate_var $(DESC_VRATE_VAR) -altstep $(DESC_ALT_STEP)  -altstart $(DESC_ALT_START) -airports $(FOLDER_DATA)/airports_tz.parquet
endef

define vectorize
	python3 add_summary.py -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_VECTORIZED)/%/$(@F),%,$@).parquet -f_out $@ -periods $(VECT_PERIODS) -thresh_dt $(VECT_THRESHOLD_DT) -threshold_vr $(VECT_THRESHOLD_VR)  -vrate_var $(VECT_VRATE_VAR) -altstep $(VECT_ALT_STEP)  -altstart $(VECT_ALT_START) -airports $(FOLDER_DATA)/airports_tz.parquet
endef

define correct_date
	python3 add_correct_date.py -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_DATES)/%/$(@F),%,$@).parquet -airports $(FOLDER_DATA)/airports_tz.parquet -f_out $@
endef


define add_cruise
	python3 add_cruise_infos.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_CRUISE)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(FOLDER_DATA)/airports_tz.parquet -nsplit $(CRUISE_NSPLIT)
endef

define add_wind
	python3 add_wind_effect.py -t_in $< -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_WIND)/%/$(@F),%,$@).parquet  -f_out $@ -airports $(FOLDER_DATA)/airports_tz.parquet
endef
#-corrected_dates $(patsubst $(FOLDER_CRUISE)/%,$(FOLDER_DATES)/%,$@)



$(FOLDER_DATA)/airports_tz.parquet:
#	curl -o $(FOLDER_DATA)/airports.csv https://github.com/davidmegginson/ourairports-data/blob/main/airports.csv
	python3 add_timezone.py -a_in airports.csv -a_out $@  -flights "$(FLIGHTS)"


$(FOLDER_DATA)/METARs.parquet: $(FOLDER_DATA)/airports_tz.parquet
	python metars_folder_to_parquet.py to_parquet -airports $(FOLDER_DATA)/airports_tz.parquet -metars_folder_in $(FOLDER_DATA)/METARs -metars_parquet_out $@

$(FOLDER_WEATHER)/%.parquet: $(FOLDER_FLGT)/%.parquet #$(FOLDER_DATA)/METARs.parquet $(FOLDER_DATA)/airports_tz.parquet
	@mkdir -p $(@D)
	python3 add_weather_from_metars.py -f_in $^ -airports $(FOLDER_DATA)/airports_tz.parquet -metars $(FOLDER_DATA)/METARs.parquet -f_out $@ -geo_scale 1 -hour_scale 1


$(FOLDER_THUNDER)/%.parquet: $(FOLDER_FLGT)/%.parquet #$(FOLDER_DATA)/METARs.parquet $(FOLDER_DATA)/airports_tz.parquet
	@mkdir -p $(@D)
	python3 add_thunder_from_metars.py -f_in $^ -airports $(FOLDER_DATA)/airports_tz.parquet -metars $(FOLDER_DATA)/METARs.parquet -f_out $@ -geo_scale 1 -hour_scale 1

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

$(FOLDER_MASS)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet  $(FOLDER_DATA)/airports_tz.parquet
	@mkdir -p $(@D)
	$(call add_mass)

$(FOLDER_MASS)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet  $(FOLDER_DATA)/airports_tz.parquet
	@mkdir -p $(@D)
	$(call add_mass)


$(FOLDER_VECTORIZED)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call vectorize)


$(FOLDER_VECTORIZED)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call vectorize)


$(FOLDER_MASSDESC)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call add_mass_descent)


$(FOLDER_MASSDESC)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call add_mass_descent)

$(FOLDER_DATES)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call correct_date)

$(FOLDER_DATES)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet
	@mkdir -p $(@D)
	$(call correct_date)




$(FOLDER_CRUISE)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet  $(FOLDER_DATA)/airports_tz.parquet
	@mkdir -p $(@D)
	$(call add_cruise)

$(FOLDER_CRUISE)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet  $(FOLDER_DATA)/airports_tz.parquet #$(FOLDER_DATES)/challenge_set/%.parquet
	@mkdir -p $(@D)
	$(call add_cruise)


$(FOLDER_WIND)/final_submission_set/%.parquet: $(FOLDER_INT)/%.parquet #$(FOLDER_DATES)/submission_set/%.parquet
	@mkdir -p $(@D)
	$(call add_wind)

$(FOLDER_WIND)/challenge_set/%.parquet: $(FOLDER_INT)/%.parquet #$(FOLDER_DATES)/challenge_set/%.parquet
	@mkdir -p $(@D)
	$(call add_wind)




# $(FOLDER_MASS)/challenge_set/%.parquet $(FOLDER_MASS)/submission_set/%.parquet: $(FOLDER_INT)/%.parquet
# 	@mkdir -p $(@D)
# 	python3 add_mass.py -t_in $^ -f_in $(FOLDER_FLGT)/$(patsubst $(FOLDER_MASS)/%/$(@F),%,$@).parquet -f_out $@
