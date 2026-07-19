# ClearTrace Data Decisions

## Station selection

- OpenAQ initially returned 99 candidate Delhi-area monitoring locations.
- Stale locations, NCR stations, duplicate station versions and stations with insufficient useful coverage were removed.
- The final dataset retains 38 Delhi monitoring stations.
- Stations were not required to measure all six pollutants because doing so would significantly reduce geographic coverage.

## Selected pollutants

The project uses the six pollutants required for CPCB AQI calculation:

- PM2.5
- PM10
- NO2
- CO
- SO2
- O3

## Unit handling

- PM2.5, PM10 and O3 are represented in µg/m³.
- NO2 and SO2 are converted from ppb to µg/m³.
- CO is labelled as ppb in the source data, but its observed value range is consistent with mg/m³-scale readings. It is therefore interpreted as mg/m³ for AQI calculation.
- Wind speed is converted from km/h to m/s before creating wind-vector features.

Unit-handling decisions must remain identical during training and live inference.

## Observation provenance

For every pollutant, `has_<pollutant>` records whether the station supplied an original valid observation at that timestamp.

These availability flags are the source of truth.

The original cleaning pipeline created:

- `*_interpolated`
- `*_proxy_imputed`

These flags preserve the provenance of filled values. However, interpolation and proxy estimates are not treated as original observations in the final v2 feature contract.

Notebook 04 reconstructs each observed-only pollutant column using:

`<pollutant>_observed = pollutant value where has_<pollutant> is true`

This removes both interpolation and proxy estimates from the local observed-pollutant history.

## Missing values and synchronized outages

The initial EDA contained approximately 82,004 station-hour rows from 38 stations.

A timestamp-level missingness audit identified 70 hours where at least four pollutants were missing for at least 90% of stations. These periods were treated as synchronized network or provider outages rather than ordinary station-level missingness.

The outage timestamps were removed instead of being spatially imputed.

After all cleaning and feature-preparation steps, the final v2 dataset contains 78,204 station-hour rows.

## Intermediate interpolation and proxy processing

The intermediate `master_clean.csv` and `features_core_v1.csv` retain short-gap interpolation and neighbouring-station proxy estimates for auditability.

The original proxy strategy used:

- pollutant-specific donor stations;
- an 80% availability threshold;
- a maximum distance of 15 km;
- same-timestamp donor values;
- nearest-first donor selection;
- a frozen pre-proxy snapshot so newly filled values were not reused as donors.

The later feature audit found that proxy estimates had also created pollutant histories at stations that had never originally reported those pollutants. The largest structural issue occurred for SO2, but PM10 and O3 were also affected.

Rather than treating the resulting values as local observations, v2 reconstructs original observed values using `has_<pollutant>` and excludes the legacy filled pollutant columns from the model contract.

Therefore:

- `master_clean.csv` is an intermediate cleaning artifact.
- `features_core_v1.csv` is the historical baseline.
- `features_core_v2.csv` is the authoritative modelling dataset.

## AQI calculation

- PM2.5, PM10, NO2 and SO2 use 24-hour rolling inputs.
- CO and O3 use 8-hour rolling inputs.
- The hourly AQI is the maximum available pollutant subindex.
- AQI is valid only when at least three pollutant subindices are available and at least one particulate subindex is present.
- AQI is calculated before pollutant interpolation and proxy processing.
- AQI is never recalculated from proxy-imputed pollutant values.

## Forecasting targets

ClearTrace predicts an hourly AQI trajectory for the next 24 hours.

The target columns are:

- `target_aqi_1h`
- `target_aqi_2h`
- ...
- `target_aqi_24h`

Targets are created using exact station-and-future-timestamp matching rather than row-based shifting.

A target remains NaN when:

- the exact future timestamp is absent; or
- the future observed AQI is invalid.

Target values are never interpolated or imputed.

A 48-hour forecast is deferred until the 24-hour system is fully validated.

## Calendar features

The v2 dataset includes:

- `hour`
- `day_of_week`
- `is_weekend`
- `month`
- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`

Pandas weekday convention is used: Monday = 0 and Sunday = 6.

Cyclical sine and cosine encodings prevent artificial discontinuities between adjacent cyclic values such as 23:00 and 00:00.

## AQI context features

The model receives:

- `current_aqi`
- `aqi_calculation_valid`
- `aqi_lag_1h`
- `aqi_lag_6h`
- `aqi_lag_12h`
- `aqi_lag_24h`

AQI lags are created through exact station-and-timestamp matching.

## Observed pollutant features

For every pollutant, v2 contains:

`<pollutant>_observed`

These columns contain only original station observations. Interpolated and proxy-imputed values are represented as missing in the observed-only columns.

## Causal missingness features

Full-period station-availability summaries reveal whether a pollutant appears later in the dataset and are therefore unsafe for live inference.

The final v2 model instead uses:

- `station_seen_<pollutant>_so_far`
- `<pollutant>_temporary_missing_causal`
- `<pollutant>_not_yet_observed_causal`

These features use station-wise cumulative history. At any timestamp, they depend only on the current and previous timestamps.

## Observed pollutant lag features

All pollutant lags are rebuilt from `<pollutant>_observed` only.

Configured lags:

- PM2.5: 1h, 6h, 12h, 24h
- PM10: 1h, 6h, 12h, 24h
- NO2: 1h, 6h, 24h
- CO: 1h, 6h, 24h
- O3: 1h, 6h, 24h
- SO2: 1h, 6h

No interpolation or proxy estimate is allowed to enter these lag features.

## Observed rolling features

Rolling means are timestamp-aware and are calculated from original observations only.

Configured windows:

- PM2.5: 6h, 12h, 24h
- PM10: 6h, 12h, 24h
- NO2: 6h, 12h
- CO: 6h, 12h
- O3: 6h, 12h
- SO2: 6h

Rolling windows include the current timestamp because current measurements are available at the forecast origin.

Every rolling mean has a corresponding observation-count feature. This allows the model to distinguish a well-supported rolling mean from one calculated using only a small number of observations.

## Weather-derived features

The final weather feature set includes:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `surface_pressure`
- `is_raining`
- `wind_speed_10m_ms`
- `wind_u_ms`
- `wind_v_ms`

Meteorological wind-from direction is converted into flow components:

- positive `u`: eastward flow;
- positive `v`: northward flow.

## Neighbouring-station spatial features

Neighbouring-station measurements remain separate spatial context features. They are never inserted into the local observed-pollutant columns.

For every pollutant, v2 contains:

- `<pollutant>_neighbor_idw_v2`
- `<pollutant>_neighbor_count_v2`
- `<pollutant>_neighbor_nearest_km_v2`

Spatial-feature rules:

- only original observations are eligible donors;
- the target station cannot donate to itself;
- donors must report at the same timestamp;
- up to three nearest available donors are used;
- estimates use inverse-distance weighting with power 2;
- the neighbour count and nearest distance are retained as uncertainty context.

Candidate radii were selected using a chronological validation period. The final chronological test period was evaluated only after radius selection.

Selected pollutant-specific radii:

- PM10: 12.5 km
- CO: 12.5 km
- PM2.5: 15 km
- NO2: 15 km
- SO2: 15 km
- O3: 15 km

MAE, RMSE and coverage measure the usefulness of neighbouring estimates. They do not establish causal source attribution.

## Final v2 feature contract

`features_core_v2.csv` contains:

- 78,204 station-hour rows;
- 145 total columns;
- 24 future AQI targets;
- 117 numeric or Boolean model features;
- `station_name` as an additional categorical predictor;
- 118 total Module 2 model inputs;
- 18 final spatial features;
- zero duplicate station-hour rows;
- zero infinite feature values.

Identifiers such as `location_id`, `provider_name` and `timestamp_hour` are retained for alignment and auditing but are not ordinary numeric model predictors.

The following legacy columns are excluded from the v2 model predictors:

- raw mixed-provenance pollutant columns;
- `*_interpolated`;
- `*_proxy_imputed`;
- `station_observed_*`;
- `*_temporary_missing`;
- `*_structural_missing`;
- `*_neighbor_estimate_v1`;
- `station_reliability`;
- all `target_aqi_*` columns.

## Emission-source context

The emission-source inventory and station–source links are maintained as separate context artifacts:

- `source_inventory_v1.csv`
- `station_source_links_v1.csv`

They support explanation and RAG by retrieving plausible nearby source context for a station or user location.

They are not inputs to the CatBoost models trained on `features_core_v2.csv`.

A nearby source is contextual evidence, not proof that it caused an AQI event. Source explanations should consider:

- source confidence and provenance;
- distance from the station or user;
- current activity evidence;
- source–pollutant compatibility;
- wind alignment;
- possible transport lag.

Adding emission-source features directly to forecasting would require a new feature version and complete model retraining.

## Deferred NASA FIRMS integration

NASA FIRMS active-fire data is excluded from the current forecasting model because:

- it detects thermal anomalies rather than smoke concentration;
- meaningful use requires transport distance, time lag, wind alignment, confidence and fire-radiative-power logic;
- the current dataset may contain too few relevant fire events;
- an unvalidated fire feature risks adding sparse noise.

It remains a possible seasonal enhancement subject to separate validation and ablation testing.

## Modelling constraints

- Never use a `target_aqi_*` column as an input feature.
- Use chronological, horizon-purged train, validation and test splits.
- Never use a random row split for time-series forecasting.
- Fit model and preprocessing state using training data only.
- Do not impute target values.
- Train each horizon only on rows with a valid corresponding target.
- Keep interpolation and legacy proxy values out of the v2 model inputs.
- Preserve ordered feature names with the serialized models.
- Evaluate performance by horizon, station and AQI severity.
- Compare model performance against a persistence baseline.
- Treat the final test period as untouched until all feature and model choices are fixed.

## Versioning rule

- `features_core_v1.csv` is the historical baseline produced by the original feature-engineering notebook.
- `features_core_v2.csv` is the current authoritative forecasting handoff used by Module 2.
- Emission-source inventory and station–source links remain separate explanation artifacts.
- Any future change to the model feature contract must create a new dataset version.
- Any changed feature contract requires model retraining.