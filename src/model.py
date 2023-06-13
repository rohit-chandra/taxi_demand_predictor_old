import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """Adds 1 column with the avergae rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    
    Args:
        X (pd.DataFrame): _description_
    
    Returns:
        pd.DataFrame: _description_
    """
    
    X["average_rides_last_4_weeks"] = 0.25 * (
        X[f"rides_previous_{7 * 24}_hour"]  + \
        X[f"rides_previous_{2 * 7 * 24}_hour"] + \
        X[f"rides_previous_{3 * 7 * 24}_hour"] + \
        X[f"rides_previous_{4 * 7 * 24}_hour"]
    )
    
    return X


class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    """
    
    def fit(self, X, y = None):
        return self
    
    
    def transform(self, X, y = None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        
        # remove pickup_hour column since it's a datetime column and lightgbm only accepts numeric columns
        return X_.drop(columns=["pickup_hour"])



def get_pipeline(**hyperparams) -> Pipeline:
    """Returns a pipeline with the following steps:
    - average_rides_last_4_weeks
    - TemporalFeatureEngineering
    - lgb.LGBMRegressor
    
    Args:
        **hyperparams: _description_
    
    Returns:
        _type_: _description_
    """
    
    # sklearn Transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
    )
    
    # sklearn Transform
    add_temporal_features = TemporalFeatureEngineering()
    
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )