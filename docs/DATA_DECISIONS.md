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