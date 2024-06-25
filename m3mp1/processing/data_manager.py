import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from m3mp1 import __version__ as _version
from m3mp1.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from m3mp1.processing.features import OutlierHandler, WeekdayImputer, WeekdayOneHotEncoder

##  Pre-Pipeline Preparation

# 1. Extracts the title (Mr, Ms, etc) from the name variable
def get_year_month(dataframe):
    X = dataframe.copy()
    X['dteday'] = pd.to_datetime(X['dteday'], format='%Y-%m-%d')
    X['year'] = X['dteday'].dt.year
    X['month'] = X['dteday'].dt.month_name()
    X['day'] = X['dteday'].dt.day_name().str[:3]
    return X

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # Convert dteday to time stamp
    data_frame = get_year_month(data_frame)
    
    pipeline_list = [
        ('weekday_imputation', WeekdayImputer(variable=config.model_config.weekday_var)),
        ('weekday_onehot', WeekdayOneHotEncoder(variables =config.model_config.weekday_var))
    ]
    
    for i in config.model_config.feature_outlier:
        outHandler = OutlierHandler(variable=i)
        pipeline_list.append(("{0}_outlier".format(i), outHandler))
        
    print(pipeline_list)
    
    preprocess_pipe = Pipeline(pipeline_list)
    
    print("---- Before Pre Processing -----")
    print(data_frame.head())
    
    preprocess_pipe.fit(data_frame)
    data_frame = preprocess_pipe.transform(data_frame)

    print("---- Post Pre Processing -----")
    print(data_frame.head())
    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    print("---- Post column drop -----")
    print(data_frame.head())
    
    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
