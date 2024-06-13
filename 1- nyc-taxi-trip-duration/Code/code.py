import pandas as pd
import numpy as np
import os
import math
import time          #to get the system time

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
import pickle

pd.options.display.max_columns = None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def manhattan_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    delta_lat = np.abs(lat2_rad - lat1_rad)
    delta_lon = np.abs(lon2_rad - lon1_rad)
    
    a_lat = R * delta_lat
    a_lon = R * delta_lon * np.cos((lat1_rad + lat2_rad) / 2)
    
    distance = a_lat + a_lon
    return distance


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def Rush_Hour_Indicator(hour):
    if (hour >= 6 and hour <= 9) or (hour >= 16 and hour <= 18):
        return 1
    else:
        return 0

def Time_of_Day_Segment(hour):
    if hour >= 6 and hour < 12:
        return 0  # Morning
    elif hour >= 12 and hour < 18:
        return 1  # Afternoon
    elif hour >= 18 and hour < 22:
        return 2  # Evening
    else:
        return 3  # Night

def remove_outliers(train):
    train = train.loc[(train.pickup_latitude > 40.6) & (train.pickup_latitude < 40.9)]
    train = train.loc[(train.dropoff_latitude>40.6) & (train.dropoff_latitude < 40.9)]
    train = train.loc[(train.pickup_longitude > -74.05) & (train.pickup_longitude < -73.7)]
    train = train.loc[(train.dropoff_longitude > -74.05) & (train.dropoff_longitude < -73.7)]
    
    return train


def cluster_features(train, val, n=10):
    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))
    
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=n, batch_size=10000, random_state=42).fit(coords[sample_ind])
    
    train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
    val.loc[:, 'pickup_cluster'] = kmeans.predict(val[['pickup_latitude', 'pickup_longitude']])
    val.loc[:, 'dropoff_cluster'] = kmeans.predict(val[['dropoff_latitude', 'dropoff_longitude']])
    
    return  train, val


def prepare_data(train):
    # drop 'id'
    train.drop(columns=['id'], inplace=True)   
    
    # convert Target variable to log
    train['log_trip_duration'] = np.log1p(train.trip_duration)
    
    # drop 'trip_duration'
    train.drop(columns=['trip_duration'], inplace=True) 
    
    # convert 'pickup_datetime' to >> 'dayofweek', 'month', 'hour', 'dayofyear'
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['month'] = train.pickup_datetime.dt.month
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear

    # drop 'pickup_datetime'
    train.drop(columns=['pickup_datetime'], inplace=True)   
    
    # Note, it will map the available 2 classes to {0, 1}
    train = pd.get_dummies(train, columns=['store_and_fwd_flag'])
    
    # calculating haversine distance 
    train['haversine_distance'] = haversine_distance(train['pickup_latitude'], train['pickup_longitude'], train['dropoff_latitude'], train['dropoff_longitude'])
    
    # calculating manhattan distance 
    train['manhattan_distance'] = manhattan_distance(train['pickup_latitude'], train['pickup_longitude'], train['dropoff_latitude'], train['dropoff_longitude'])
    
    # Bearing: The direction of the trip.
    train['Bearing'] = bearing_array(train['pickup_latitude'], train['pickup_longitude'], train['dropoff_latitude'], train['dropoff_longitude'])
    
    # drop 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'
   # train.drop(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'], inplace=True)
    
    # Apply Rush_Hour_Indicator function to create the 'rush_hour' column
    train['rush_hour'] = train['hour'].apply(Rush_Hour_Indicator)
    
    # Create a new column 'weekend' with binary indicator for weekend (1) or weekday (0)
    train['weekend'] = (train['dayofweek'] >= 5).astype(int)
    
    # Apply Time_of_Day_Segment function to create the 'day_part' column
    train['day_part'] = train['hour'].apply(Time_of_Day_Segment)
    
    """
    Remove rows where pickup or dropoff latitude is outside the range 40.6 to 40.9,
    or where pickup or dropoff longitude is outside the range -74.05 to -73.7.
    """
    train = remove_outliers(train)
    
    return train

def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} R2 = {r2:.4f}")

def approach1(train, val):
    
    numeric_features = ['haversine_distance', 'manhattan_distance', 'Bearing']
    categorical_features = ['vendor_id', 'month', 'hour', 'dayofyear','passenger_count','dayofweek',
                            'pickup_cluster','dropoff_cluster']
    # rush_hour, day_part, dayofweek, 'store_and_fwd_flag_Y','store_and_fwd_flag_N'
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('poly', PolynomialFeatures(degree=2)),
        ('regression', Ridge(alpha=1, random_state = 42))
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, val, train_features, "val")

if __name__ == '__main__':
    start = time.time()
    
    root_dir = r'E:\Machine Learining\Dr Mostafa Saad\ML\my work\projectes\project-nyc-taxi-trip-duration'
    
    train = pd.read_csv(os.path.join(root_dir, 'split/train.csv'))
    val = pd.read_csv(os.path.join(root_dir, 'split/val.csv'))
    
    # Clustering
    trani , val = cluster_features(train, val, n=100)
    
    train = prepare_data(train)
    val = prepare_data(val)
    

    approach1(train, val)
    
    end = time.time()
    elapsed_time = end - start
    minutes, seconds = divmod(elapsed_time, 60)
    print( f"Time taken is {int(minutes)} minutes and {seconds:.2f} seconds." )
    