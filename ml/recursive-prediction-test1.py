import pandas as pd
import numpy as np
from geopy.distance import geodesic  # for calculating distances
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Load datasets
print("Step 0: Loading datasets...")
vessel_data = pd.read_csv('../ais_train.csv', sep='|')
schedule_data = pd.read_csv('../schedules_to_may_2024.csv', sep='|')
ports_data = pd.read_csv('../ports.csv', sep='|')
vessels_info = pd.read_csv('../vessels.csv', sep='|')
print("Datasets loaded successfully.\n")

# Step 1: Convert 'time' column to datetime in vessel_data
print("Step 1: Converting 'time' column to datetime in vessel_data...")
vessel_data['time'] = pd.to_datetime(vessel_data['time'])
print("Converted 'time' column to datetime.\n")

# Step 2: Sort the data by vesselId and time to ensure proper lag feature calculation
print("Step 2: Sorting vessel_data by vesselId and time...")
vessel_data = vessel_data.sort_values(by=['vesselId', 'time'])
print("Sorted vessel_data by vesselId and time.\n")

# Step 3: Create lag features for past positions (latitude, longitude)
print("Step 3: Creating lag features for past positions (latitude, longitude)...")
vessel_data['prev_latitude'] = vessel_data.groupby('vesselId')['latitude'].shift(1)
vessel_data['prev_longitude'] = vessel_data.groupby('vesselId')['longitude'].shift(1)
vessel_data['prev_time'] = vessel_data.groupby('vesselId')['time'].shift(1)
print("Lag features created.\n")

# Step 4: Calculate speed (in km/h) using geodesic distance and time difference
print("Step 4: Calculating speed using geodesic distance and time difference...")
vessel_data['time_diff'] = (vessel_data['time'] - vessel_data['prev_time']).dt.total_seconds() / 3600  # Time diff in hours
vessel_data['distance_traveled'] = vessel_data.apply(
    lambda row: geodesic((row['prev_latitude'], row['prev_longitude']),
                         (row['latitude'], row['longitude'])).kilometers if pd.notnull(row['prev_latitude']) else np.nan,
    axis=1)
vessel_data['speed'] = vessel_data['distance_traveled'] / vessel_data['time_diff']
print("Speed and distance traveled calculated.\n")

# Step 5: Calculate direction (bearing) of vessel movement
print("Step 5: Calculating direction (bearing) of vessel movement...")
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two points"""
    if pd.isnull(lat1) or pd.isnull(lat2) or pd.isnull(lon1) or pd.isnull(lon2):
        return np.nan
    delta_lon = lon2 - lon1
    x = np.sin(np.radians(delta_lon)) * np.cos(np.radians(lat2))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(delta_lon))
    bearing = np.degrees(np.arctan2(x, y))
    return bearing

vessel_data['direction'] = vessel_data.apply(
    lambda row: calculate_bearing(row['prev_latitude'], row['prev_longitude'], row['latitude'], row['longitude']), axis=1)
print("Direction calculated.\n")

# Step 6: Merge vessel data with vessel characteristics (from vessels_info)
print("Step 6: Merging vessel_data with vessel characteristics...")
vessel_data = pd.merge(vessel_data, vessels_info[['vesselId', 'CEU', 'DWT', 'GT', 'length', 'breadth', 'enginePower']],
                       on='vesselId', how='left')
print("Merged vessel_data with vessel characteristics.\n")

# Step 7: Ensure uniqueness in ports_data (keep the first occurrence of each portId)
print("Step 7: Ensuring uniqueness in ports_data...")
ports_data_unique = ports_data.drop_duplicates(subset='portId')
print("Ensured uniqueness in ports_data.\n")

# Ensure that 'portId' is still present after dropping duplicates
assert 'portId' in ports_data_unique.columns, "portId is missing from ports_data_unique"
print("Verified that 'portId' is present in ports_data_unique.\n")

# Step 8: Ensure a one-to-one relationship between vesselId and portId in schedule_data
print("Step 8: Ensuring a one-to-one relationship between vesselId and portId in schedule_data...")
schedule_data['arrivalDate'] = pd.to_datetime(schedule_data['arrivalDate'])
schedule_data = schedule_data.sort_values(by='arrivalDate')  # Sort by date
print("Converted 'arrivalDate' to datetime and sorted schedule_data.")

# Keep only the most recent entry per vesselId (using drop_duplicates)
schedule_data_recent = schedule_data.drop_duplicates(subset='vesselId', keep='last')
print("Kept the most recent entry per vesselId in schedule_data.\n")

# Step 9: Merge vessel_data with the recent schedule data
print("Step 9: Merging vessel_data with recent schedule data...")
vessel_data = pd.merge(vessel_data, schedule_data_recent[['vesselId', 'portId']], on='vesselId', how='left', suffixes=('', '_schedule'))
print("Merged vessel_data with recent schedule data.\n")

# Debugging output before checking for nulls
print("Columns in vessel_data before null check:", vessel_data.columns)
print("First few rows of vessel_data before null check:")
print(vessel_data.head())

# Check if the merge introduced missing 'portId' values
if 'portId' in vessel_data.columns:
    if vessel_data['portId'].isnull().any():
        print("Warning: Some rows in vessel_data have missing portId after the merge with schedule_data_recent")
        print(vessel_data[vessel_data['portId'].isnull()])  # Display rows with missing portId
else:
    print("Error: 'portId' column is missing from vessel_data.")

# Step 10: Now merge vessel_data with the cleaned ports_data on portId (one-to-one or one-to-many)
print("Step 10: Merging vessel_data with ports_data on portId...")
if 'portId' not in vessel_data.columns:
    raise KeyError("portId column is missing from vessel_data before merging with ports_data_unique")
else:
    vessel_data = pd.merge(vessel_data, ports_data_unique[['portId', 'latitude', 'longitude', 'countryName']],
                           left_on='portId', right_on='portId', how='left', suffixes=('', '_port'))
    print("Merged vessel_data with ports_data.\n")

# Step 11: Calculate distance to the nearest port (if known)
print("Step 11: Calculating distance to the nearest port...")
vessel_data['distance_to_port'] = vessel_data.apply(
    lambda row: geodesic((row['latitude'], row['longitude']),
                         (row['latitude_port'], row['longitude_port'])).kilometers if pd.notnull(row['latitude_port']) else np.nan,
    axis=1)
print("Distance to the nearest port calculated.\n")

# Step 12: Handle missing values (fill NaNs)
print("Step 12: Handling missing values...")
vessel_data['speed'].fillna(0, inplace=True)
vessel_data['direction'].fillna(0, inplace=True)
vessel_data['distance_to_port'].fillna(vessel_data['distance_to_port'].mean(), inplace=True)
print("Handled missing values in vessel_data.\n")

# Step 13: Create time-based features (hour, day_of_week, month) from 'time'
print("Step 13: Creating time-based features from 'time'...")
vessel_data['hour'] = vessel_data['time'].dt.hour
vessel_data['day_of_week'] = vessel_data['time'].dt.dayofweek
vessel_data['month'] = vessel_data['time'].dt.month
print("Time-based features created in vessel_data.\n")

# -------- Feature prediction for the test set -------- #
# Load test data
print("Step 14: Loading test data...")
ais_test = pd.read_csv('../ais_test.csv')
ais_test['time'] = pd.to_datetime(ais_test['time'])
print("Loaded test data and converted 'time' column to datetime.\n")

# Merge test data with vessels_info to get missing features
print("Step 15: Merging test data with vessel characteristics...")
ais_test = pd.merge(ais_test, vessels_info[['vesselId', 'CEU', 'DWT', 'GT', 'length', 'breadth', 'enginePower']],
                     on='vesselId', how='left')
print("Merged test data with vessel characteristics.\n")

# Create time-based features for the test set
ais_test['hour'] = ais_test['time'].dt.hour
ais_test['day_of_week'] = ais_test['time'].dt.dayofweek
ais_test['month'] = ais_test['time'].dt.month
print("Time-based features created for the test set.\n")


# We will now train models to predict missing features (speed, direction, etc.) using only time-based features
print("Step 16: Preparing training data for predicting missing features...")
X_train = vessel_data[['hour', 'day_of_week', 'month']]
y_speed = vessel_data['speed']
y_direction = vessel_data['direction']
y_distance_to_port = vessel_data['distance_to_port']
print("Training data prepared.\n")

# Train models to predict speed, direction, and distance_to_port using time features
print("Step 17: Training models to predict speed, direction, and distance to port...")
model_speed = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_speed.fit(X_train, y_speed)
print("Model for speed prediction trained.")

model_direction = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_direction.fit(X_train, y_direction)
print("Model for direction prediction trained.")

model_distance_to_port = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_distance_to_port.fit(X_train, y_distance_to_port)
print("Model for distance to port prediction trained.\n")

# Predict the missing features in the test set
print("Step 18: Predicting missing features in the test set...")
X_test_time_features = ais_test[['hour', 'day_of_week', 'month']]
ais_test['speed'] = model_speed.predict(X_test_time_features)
ais_test['direction'] = model_direction.predict(X_test_time_features)
ais_test['distance_to_port'] = model_distance_to_port.predict(X_test_time_features)
print("Predicted missing features in the test set.\n")

# Step 19: Prepare the final features for model training and prediction
print("Step 19: Preparing final features for model training and prediction...")
features = ['hour', 'day_of_week', 'month', 'speed', 'direction', 'distance_to_port', 'CEU', 'DWT', 'GT', 'length', 'breadth', 'enginePower']
X_train = vessel_data[features]
y_latitude = vessel_data['latitude']
y_longitude = vessel_data['longitude']
print("Final features prepared.\n")

# Step 20: Split the training data into training and test sets
print("Step 20: Splitting the training data into training and test sets...")
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X_train, y_latitude, test_size=0.2, random_state=42)
X_train_lon, X_test_lon, y_train_lon, y_test_lon = train_test_split(X_train, y_longitude, test_size=0.2, random_state=42)
print("Training data split into training and test sets.\n")

# Step 21: Train XGBoost models
print("Step 21: Training XGBoost models for latitude and longitude predictions...")
# Model for latitude prediction
model_latitude = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_latitude.fit(X_train_lat, y_train_lat)
print("Model for latitude prediction trained.")

# Model for longitude prediction
model_longitude = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_longitude.fit(X_train_lon, y_train_lon)
print("Model for longitude prediction trained.\n")

# Step 22: Make predictions for the test set
print("Step 22: Making predictions for latitude and longitude on the test set...")
ais_test_final_features = ais_test[features]  # Use the test set with predicted missing features
latitude_predictions = model_latitude.predict(ais_test_final_features)
longitude_predictions = model_longitude.predict(ais_test_final_features)
print("Predictions for latitude and longitude completed.\n")

# Step 23: Load the sample submission file and save the predictions
print("Step 23: Loading the sample submission file and saving predictions...")
sample_submission = pd.read_csv('../ais_sample_submission.csv')
sample_submission['latitude_predicted'] = latitude_predictions
sample_submission['longitude_predicted'] = longitude_predictions
print("Added predictions to the sample submission file.\n")

# Save the final submission file
sample_submission.to_csv('recursive-prediction1.csv', index=False)
print("Saved the final submission file as 'recursive-prediction1.csv'.\n")

# Display the first few rows of the submission file to verify
print("First few rows of the submission file:")
print(sample_submission.head())
