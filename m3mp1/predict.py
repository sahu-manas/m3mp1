import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from m3mp1 import __version__ as _version
from m3mp1.config.core import config
from m3mp1.pipeline import bike_rental_pipe
from m3mp1.processing.data_manager import load_pipeline
from m3mp1.processing.data_manager import pre_pipeline_preparation
from m3mp1.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_rental_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    #validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    preprocessed_data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    #validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version}
    
    predictions = bike_rental_pipe.predict(preprocessed_data)

    results = {"predictions": predictions,"version": _version}
    print(results)

    predictions = bike_rental_pipe.predict(preprocessed_data)
    results = {"predictions": predictions,"version": _version}
    print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday':['2012-11-05'],'season':['winter'],'hr':['7am'],'holiday':['No'],'weekday':['Mon'],
                'workingday':['Yes'],'weathersit':['Mist'],'temp':[7.1],'atemp':[3.0014000000000003],
                'hum':[50.0],'windspeed':[19.0012], 
                'casual':[29], 'registered':[140]}
    
    make_prediction(input_data=data_in)
