import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from m3mp1.config.core import config
from m3mp1.processing.features import WeathersitImputer, WeekdayImputer, WeekdayOneHotEncoder
from m3mp1.processing.features import Mapper
from m3mp1.processing.features import OutlierHandler

bike_rental_pipe = Pipeline([

    #('weekday_imputation', WeekdayImputer(variable='weekday')),
    ('weathersit_imputation', WeathersitImputer(variable=config.model_config.weathersit_var)),

    ##==========Mapper======##
    ('map_year', Mapper(config.model_config.year_var, config.model_config.year_mapping)),
    ('map_mnth', Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping)),
    ('map_season', Mapper(config.model_config.season_var, config.model_config.season_mapping)),
    ('map_weather', Mapper(config.model_config.weathersit_var, config.model_config.weather_mapping)),
    ('map_holiday', Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping)),
    ('map_workingday', Mapper(config.model_config.workingday_var, config.model_config.workingday_mapping)),
    ('map_hr', Mapper(config.model_config.hr_var, config.model_config.hr_mapping)),

    # Outlier handling
    #('dteday_outlier', OutlierHandler(variable='dteday')),
    #('temp_outlier', OutlierHandler(variable='temp')),
    #('atemp_outlier', OutlierHandler(variable='atemp')),
    #('hum_outlier', OutlierHandler(variable='hum')),
    #('windspeed_outlier', OutlierHandler(variable='windspeed')),
    #('casual_outlier', OutlierHandler(variable='casual')),
    #('registered_outlier', OutlierHandler(variable='registered')),
    #('cnt_outlier', OutlierHandler(variable='cnt')),
    #('year_outlier', OutlierHandler(variable='year')),

    #WeekDay One hot encoding
    #('weekday_onehot', WeekdayOneHotEncoder(variables ='weekday')),

    #Drop Features
    #("columnDropper", columnDropperTransformer(['dteday', 'casual', 'registered','weekday','day'])),
    #('drop_features', DropFeatures(features_to_drop=['dteday', 'casual', 'registered','weekday','day'])),

    # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, 
                                       max_depth=config.model_config.max_depth,
                                       random_state=config.model_config.random_state))
])