from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables must be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.idx = X[X[self.variable].isnull() == True].index
        #self.day_name = X.loc[idx, 'day']
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X.loc[self.idx, self.variable] = X.loc[self.idx, 'day']
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables must be a string")
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill_value = X[self.variable].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill_value)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mapping: dict):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables must be a string")
        self.variable = variable
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mapping).astype(int)
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables must be a string")
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        for i in X.index:
            if X.loc[i,self.variable] > self.upper_bound:
                X.loc[i,self.variable] = self.upper_bound
            if X.loc[i,self.variable] < self.lower_bound:
                X.loc[i,self.variable] = self.lower_bound
        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables):
      if not isinstance(variables, str):
        raise ValueError('variables should be a str')
      self.variables = variables
      self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y=None):
      self.encoder.fit(X[[self.variables]])
      #print(self.encoder.get_feature_names_out(['weekday']))
      return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       X = X.copy()
       enc_wkday_features = self.encoder.get_feature_names_out([self.variables])
       #print(enc_wkday_features)
       X[enc_wkday_features] = self.encoder.transform(X[[self.variables]])
       return X
