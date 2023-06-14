from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from src.paths import RAW_DATA_DIR

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm



def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Downloads Parquet file with historical taxi rides for the given "year" and "month"

    Args:
        year (int): current year
        month (int): current month

    Returns:
        Path: parquet file path
    """
    
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)
    
    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path, 'wb').write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")
    
    

def validate_raw_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Validate raw data: Removes rows with pickup_datetimes outside their valid range
    
    Returns:
        Path: cleaned pandas DataFrame
    """
    
    this_month_start = f"{year}-{month:02d}-01"
    next_month_start = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
    # validate pickup_datetime
    rides = rides[rides["pickup_datetime"] >= this_month_start]
    rides = rides[rides["pickup_datetime"] < next_month_start]
    
    return rides


def load_raw_data(year: int, month_lst: Optional[List[int]] = None) -> pd.DataFrame:
    """Loads raw data from local storage or downloads it from the NYC website, and
    then loads it into a Pandas DataFrame
    
    Args:
        year (int): given year
        month (Optional[List[int]], optional): given month to download data. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_location_id: ID of the pickup location
    """
    
    rides = pd.DataFrame()
    
    if month_lst is None:
        # download data for the entire year (all months)
        month_lst = list(range(1, 13))
    elif isinstance(month_lst, int):
        # download data only for the months specified in the int "month_lst" argument
        month_lst = [month_lst]
    
    # download data for the specified months
    for month in month_lst:
        # caching the file. This is not downloading the file again if it is already downloaded
        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        if not local_file.exists():
            try:
                # download the file form the NYC website
                print(f"Downloading file for {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"{year}-{month:02d} file is not available")
                continue
        else:
            print(f"File for {year}-{month:02d} already exists in local directory")
            
        # load the file into a pandas DataFrame
        rides_one_month = pd.read_parquet(local_file)
        
        # rename the columns to be consistent across months
        rides_one_month = rides_one_month[["tpep_pickup_datetime", "PULocationID"]]
        rides_one_month.rename(columns={
            "tpep_pickup_datetime": "pickup_datetime", 
            "PULocationID": "pickup_location_id"
        }, inplace = True)
        
        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)
        
        # append to existing DataFrame
        rides = pd.concat([rides, rides_one_month])
    
    if rides.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides



def add_missing_slots(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'rides' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """
    # location_ids = rides["pickup_location_id"].unique()
    location_ids = range(1, rides['pickup_location_id'].max() + 1)
    full_range = pd.date_range(rides["pickup_hour"].min(), rides["pickup_hour"].max(), freq="H")
    
    output = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only rides for this "location_id"
        rides_i = rides.loc[rides["pickup_location_id"] == location_id, ["pickup_hour", "rides_count"]]
        
        #quick way to add missing dates with 0 in a Series
        # taken from https://stackoverflow.com/a/19324591
        rides_i.set_index("pickup_hour", inplace=True)
        rides_i.index = pd.DatetimeIndex(rides_i.index)
        rides_i = rides_i.reindex(full_range, fill_value=0)
        
        # add back "location_id" column
        rides_i["pickup_location_id"] = location_id
        
        output = pd.concat([output, rides_i])
        
    
    # move the purchase_day from the index to dataframe column
    output = output.reset_index().rename(columns = {"index": "pickup_hour"})
    
    return output



def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw data into time series data

    Args:
        rides (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: time series data
    """
    # sum rides per location and pickup_hour
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("H")
    agg_rides = rides.groupby(["pickup_hour", "pickup_location_id"]).size().reset_index()
    agg_rides.rename(columns={0: "rides_count"}, inplace=True)
    
    
    # add rows for locations, pickup_hours with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)
    
    return agg_rides_all_slots




def get_cutoff_indices(data: pd.DataFrame, n_features: int, step_size:int) -> list:
    stop_position = len(data) - 1
    
    # start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices_lst = []
    
    while subseq_last_idx <= stop_position:
        indices_lst.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
        
    return indices_lst


def tranform_ts_data_into_features_and_target(ts_data: pd.DataFrame, input_seq_len: int, step_size: int) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target) format
    that can use to train supervised ML models
    """
    assert set(ts_data.columns) == {"pickup_hour", "rides_count", "pickup_location_id"}
    
    locations_ids = ts_data["pickup_location_id"].unique()
    features = pd.DataFrame()
    target = pd.DataFrame()
    
    
    for location_id in tqdm(locations_ids):
        
        # keep only ts data for this location_id
        ts_data_one_location = ts_data.loc[ts_data["pickup_location_id"] == location_id, ["pickup_hour", "rides_count"]]
        
        # pre-compute cutoff indices to split datafrane rows
        indices = get_cutoff_indices(ts_data_one_location, input_seq_len, step_size)
        
        # slice and transpose data into numpy arrays for features and target
        n_examples = len(indices)
        x = np.ndarray(shape = (n_examples, input_seq_len), dtype = np.float32)
        y = np.ndarray(shape = (n_examples), dtype = np.float32)
        pickup_hours_lst = []
        
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]["rides_count"].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]["rides_count"].values
            pickup_hours_lst.append(ts_data_one_location.iloc[idx[1]]["pickup_hour"])
            
        # convert numpy arrays into pandas dataframes for both features and target
        features_one_location = pd.DataFrame(x, columns = [f"rides_previous_{i + 1}_hour" for i in reversed(range(input_seq_len))])
        features_one_location["pickup_hour"] = pickup_hours_lst
        features_one_location["pickup_location_id"] = location_id
        
        # target
        targets_one_location = pd.DataFrame(y, columns = [f"target_rides_next_hour"])
        
        # concatenate features and target into 2 dataframes
        features = pd.concat([features, features_one_location])
        target = pd.concat([target, targets_one_location])
        
    features.reset_index(drop=True, inplace=True)
    target.reset_index(drop=True, inplace=True)
    
    return features, target["target_rides_next_hour"]