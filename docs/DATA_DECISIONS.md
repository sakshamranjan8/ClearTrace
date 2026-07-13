# Data Decisions

## Station selection
- OpenAQ initially returned 99 candidate Delhi-area monitoring locations.
- After removing stale records, NCR stations, duplicate station versions, and weak pollutant coverage stations, 38 Delhi stations were retained.
- Stations were not required to have all 6 pollutants, because that would reduce hyperlocal coverage for the user's input location.

## Selected pollutants
- Selected AQI pollutants: PM2.5, PM10, NO2, CO, SO2, O3.
- Stations were retained if they had enough pollutant coverage and at least one particulate pollutant.

## Missing values
- Missing pollutant values are preserved as NaN.
- Pollutants are not blindly filled before AQI calculation.

## Unit handling
- PM2.5, PM10, and O3 are used as µg/m³.
- NO2 is converted from ppb to µg/m³.
- SO2 is converted from ppb to µg/m³.
- CO is labelled as ppb in OpenAQ, but observed values are consistent with mg/m³-scale readings, so CO is treated as mg/m³ for AQI calculation.

## AQI calculation
- PM2.5, PM10, NO2, and SO2 use 24-hour rolling averages.
- CO and O3 use 8-hour rolling averages.
- AQI is calculated as the maximum available pollutant sub-index.
- AQI is marked valid only when at least 3 pollutant sub-indices are available and at least one particulate sub-index exists.

## City-wide outage handling:
- A timestamp-level missingness audit detected 70 outage hours where at least four AQI pollutants were missing for 90% or more of stations. 
- These outage hours were removed instead of imputed, dropping 2,660 station-hour rows.
-  After removal, remaining pollutant missingness was below 0.6% for every pollutant, so residual NaNs were preserved in master_clean.csv rather than forcing nearest-station proxy imputation.

### Pollutant-specific proxy filling
- Remaining gaps are filled only from reliable nearby stations for the same pollutant.
- Reliability threshold: at least **80% valid coverage** for that pollutant.
- Maximum proxy distance: **15 km**.
- Candidate stations are tried nearest-first at the same timestamp.
- A frozen pre-proxy snapshot is used so newly filled values can never become sources for other fills.
- Proxy use is recorded in pollutant-specific `*_proxy_imputed` flags.

Remaining pollutant missing values in `master_clean.csv`:
- PM2.5: 7
- PM10: 7
- NO2: 13
- CO: 13
- SO2: 91
- O3: 7

Approximate proxy-imputation shares:
- PM2.5: 6.1%
- PM10: 7.7%
- NO2: 6.0%
- CO: 6.4%
- SO2: 23.1%
- O3: 7.8%

SO2 is treated conservatively in feature engineering because its proxy share is materially higher than the other pollutants.

##  Forecasting target definition
The MVP predicts an hourly AQI trajectory for the next 24 hours at a Delhi location.

At station level, the target columns are:
- `target_aqi_1h`
- `target_aqi_2h`
- ...
- `target_aqi_24h`

Each target is created by exact station-and-timestamp matching. A target is NaN when the exact future hour is absent or its observed AQI is invalid. Targets are never imputed.

A 48-hour forecast is deferred until the 24-hour system is validated.

## Feature-engineering decisions

### Calendar features
- `hour`
- `day_of_week` using Pandas convention: Monday = 0, Sunday = 6
- `is_weekend`
- `month`
- cyclical encodings: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`

### AQI lag features
Created by exact timestamp matching:
- 1 hour
- 6 hours
- 12 hours
- 24 hours

### Pollutant lag features
Created by exact timestamp matching with a controlled configuration:
- PM2.5: 1h, 6h, 12h, 24h
- PM10: 1h, 6h, 12h, 24h
- NO2: 1h, 6h, 24h
- CO: 1h, 6h, 24h
- O3: 1h, 6h, 24h
- SO2: 1h, 6h

The reduced SO2 lag set limits propagation of its higher proxy-imputation uncertainty.

### Rolling pollutant features
Rolling means are timestamp-aware and do not treat a fixed number of rows as a fixed number of clock hours:
- PM2.5: 6h, 12h, 24h
- PM10: 6h, 12h, 24h
- NO2: 6h, 12h
- CO: 6h, 12h
- O3: 6h, 12h
- SO2: 6h

Rolling windows include the current hour because current measurements are available at forecast time.

### Weather-derived features
- `wind_speed_10m_ms`
- `wind_u_ms`
- `wind_v_ms`
- `is_raining`

Meteorological wind direction is converted to vector components using:
- positive `u`: eastward flow
- positive `v`: northward flow

## 9. Deferred feature blocks

### NASA FIRMS
NASA FIRMS active-fire integration is excluded from the MVP because:
- it detects thermal anomalies rather than smoke concentration;
- meaningful use requires transport distance, time lag, wind alignment, confidence, and fire-radiative-power logic;
- the current 90-day period may contain too few relevant events;
- an unvalidated fire feature risks adding mostly sparse noise.

It remains a future seasonal enhancement subject to ablation testing.

### Emission-source database
Static and wind-aware emission-source features are being developed separately. They will be added to a new artifact, recommended as:
- `features_with_emissions_v2.csv`

Module 2 should begin with the core v1 dataset and later compare:
- core model
- core model + emission-source features

##  Modelling constraints
- Never use any `target_aqi_*` column as an input feature.
- Use chronological, purged train/validation/test splits; never use a random row split.
- Fit preprocessing only on the training partition.
- Do not impute target values.
- Train each forecast horizon only on rows where that horizon's target exists.
- Preserve interpolation and proxy-imputation flags as model inputs or diagnostics.
- Report performance by forecast horizon and by station, not only as one global score.

## Versioning rule
- `features_core_v1.csv`: current core forecasting handoff.
- `features_with_emissions_v2.csv`: future augmented handoff.
- Do not overwrite v1 when the emission-source block is added.