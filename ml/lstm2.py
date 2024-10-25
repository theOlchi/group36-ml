import pandas as pd
import numpy as np
from geopy.distance import geodesic  # for calculating distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from keras import Sequential, layers
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Step 0: Load datasets
print("Step 0: Loading datasets...")
vessel_data = pd.read_csv('../ais_train.csv', sep='|')
schedule_data = pd.read_csv('../schedules_to_may_2024.csv', sep='|')
ports_data = pd.read_csv('../ports.csv', sep='|')
vessels_info = pd.read_csv('../vessels.csv', sep='|')
print("Datasets loaded successfully.\n")

# Step 1: Convert 'time' column to datetime in vessel_data
print("Step 1: Converting 'time' column to datetime in vessel_data...")
vessel_data['time'] = pd.to_datetime(vessel_data['time'])

# Step 2: Sort the data by vesselId and time to ensure proper lag feature calculation
print("Step 2: Sorting vessel_data by vesselId and time...")
vessel_data = vessel_data.sort_values(by=['vesselId', 'time'])

# Step 3: Create lag features for past positions (latitude, longitude)
print("Step 3: Creating lag features for past positions (latitude, longitude)...")
vessel_data['prev_latitude'] = vessel_data.groupby('vesselId')['latitude'].shift(1)
vessel_data['prev_longitude'] = vessel_data.groupby('vesselId')['longitude'].shift(1)
vessel_data['prev_time'] = vessel_data.groupby('vesselId')['time'].shift(1)

# Step 4: Calculate speed (in km/h) using geodesic distance and time difference
print("Step 4: Calculating speed using geodesic distance and time difference...")
vessel_data['time_diff'] = (vessel_data['time'] - vessel_data['prev_time']).dt.total_seconds() / 3600  # Time diff in hours
vessel_data['distance_traveled'] = vessel_data.apply(
    lambda row: geodesic((row['prev_latitude'], row['prev_longitude']),
                         (row['latitude'], row['longitude'])).kilometers if pd.notnull(row['prev_latitude']) else np.nan,
    axis=1)
vessel_data['speed'] = vessel_data['distance_traveled'] / vessel_data['time_diff']

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

# Step 6: Merge vessel data with vessel characteristics (from vessels_info)
print("Step 6: Merging vessel_data with vessel characteristics...")
vessel_data = pd.merge(vessel_data, vessels_info[['vesselId', 'CEU', 'DWT', 'GT', 'length', 'breadth', 'enginePower']],
                       on='vesselId', how='left')

# Step 7: Ensure uniqueness in ports_data (keep the first occurrence of each portId)
print("Step 7: Ensuring uniqueness in ports_data...")
ports_data_unique = ports_data.drop_duplicates(subset='portId')

# Ensure that 'portId' is still present after dropping duplicates
assert 'portId' in ports_data_unique.columns, "portId is missing from ports_data_unique"

# Step 8: Ensure a one-to-one relationship between vesselId and portId in schedule_data
print("Step 8: Ensuring a one-to-one relationship between vesselId and portId in schedule_data...")
schedule_data['arrivalDate'] = pd.to_datetime(schedule_data['arrivalDate'])
schedule_data = schedule_data.sort_values(by='arrivalDate')

# Keep only the most recent entry per vesselId (using drop_duplicates)
schedule_data_recent = schedule_data.drop_duplicates(subset='vesselId', keep='last')

# Step 9: Merge vessel_data with the recent schedule data
print("Step 9: Merging vessel_data with recent schedule data...")
vessel_data = pd.merge(vessel_data, schedule_data_recent[['vesselId', 'portId']], on='vesselId', how='left', suffixes=('', '_schedule'))

# Step 10: Now merge vessel_data with the cleaned ports_data on portId (one-to-one or one-to-many)
print("Step 10: Merging vessel_data with ports_data on portId...")
vessel_data = pd.merge(vessel_data, ports_data_unique[['portId', 'latitude', 'longitude', 'countryName']],
                       left_on='portId', right_on='portId', how='left', suffixes=('', '_port'))

# Step 11: Calculate distance to the nearest port (if known)
print("Step 11: Calculating distance to the nearest port...")
vessel_data['distance_to_port'] = vessel_data.apply(
    lambda row: geodesic((row['latitude'], row['longitude']),
                         (row['latitude_port'], row['longitude_port'])).kilometers if pd.notnull(row['latitude_port']) else np.nan,
    axis=1)

# Step 12: Handle missing values (fill NaNs)
print("Step 12: Handling missing values...")
vessel_data['speed'].fillna(0, inplace=True)
vessel_data['direction'].fillna(0, inplace=True)
vessel_data['distance_to_port'].fillna(vessel_data['distance_to_port'].mean(), inplace=True)

# Step 13: Create time-based features (hour, day_of_week, month) from 'time'
print("Step 13: Creating time-based features from 'time'...")
vessel_data['hour'] = vessel_data['time'].dt.hour
vessel_data['day_of_week'] = vessel_data['time'].dt.dayofweek
vessel_data['month'] = vessel_data['time'].dt.month

# Step 14: Normalizing the data
print("Step 14: Normalizing the data...")
features = ['hour', 'day_of_week', 'month', 'speed', 'direction', 'distance_to_port']
scaler = StandardScaler()
vessel_data[features] = scaler.fit_transform(vessel_data[features])

# Step 15: Predicting missing features in the test set (Speed, Direction, Distance to Port)
print("Step 15: Training models to predict missing features (speed, direction, distance_to_port)...")
# Train separate models for each missing feature

X_train = vessel_data[['hour', 'day_of_week', 'month']]  # Only time-based features for prediction
y_speed = vessel_data['speed']
y_direction = vessel_data['direction']
y_distance_to_port = vessel_data['distance_to_port']

# XGBoost model for speed prediction
model_speed = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_speed.fit(X_train, y_speed)

# XGBoost model for direction prediction
model_direction = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_direction.fit(X_train, y_direction)

# XGBoost model for distance to port prediction
model_distance_to_port = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model_distance_to_port.fit(X_train, y_distance_to_port)

# Step 16: Predict missing features in the test set
print("Step 16: Predicting missing features for the test set...")
ais_test = pd.read_csv('../ais_test.csv')
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_test['hour'] = ais_test['time'].dt.hour
ais_test['day_of_week'] = ais_test['time'].dt.dayofweek
ais_test['month'] = ais_test['time'].dt.month

# Use only time-based features for test data
X_test = ais_test[['hour', 'day_of_week', 'month']]

# Predict missing features for test data
ais_test['speed'] = model_speed.predict(X_test)
ais_test['direction'] = model_direction.predict(X_test)
ais_test['distance_to_port'] = model_distance_to_port.predict(X_test)

# Step 17: Prepare LSTM model data
print("Step 17: Preparing LSTM model data...")
# Use the full set of features in training
sequence_length = 10
def create_sequences(data, target, sequence_length=10):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length].values
        label = target.iloc[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X_lat, y_lat = create_sequences(vessel_data[features], vessel_data['latitude'], sequence_length)
X_lon, y_lon = create_sequences(vessel_data[features], vessel_data['longitude'], sequence_length)

# Split into training and test sets
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X_lat, y_lat, test_size=0.2, random_state=42)
X_train_lon, X_test_lon, y_train_lon, y_test_lon = train_test_split(X_lon, y_lon, test_size=0.2, random_state=42)

# Step 18: Define and train LSTM models for latitude and longitude
print("Step 18: Defining and training LSTM models...")
model_latitude = Sequential()
model_latitude.add(layers.Input(shape=(sequence_length, len(features))))
model_latitude.add(layers.LSTM(units=50, return_sequences=False))
model_latitude.add(layers.Dense(1))
model_latitude.compile(optimizer='adam', loss='mean_squared_error')
model_latitude.fit(X_train_lat, y_train_lat, epochs=10, batch_size=32, validation_data=(X_test_lat, y_test_lat))

model_longitude = Sequential()
model_longitude.add(layers.Input(shape=(sequence_length, len(features))))
model_longitude.add(layers.LSTM(units=50, return_sequences=False))
model_longitude.add(layers.Dense(1))
model_longitude.compile(optimizer='adam', loss='mean_squared_error')
model_longitude.fit(X_train_lon, y_train_lon, epochs=10, batch_size=32, validation_data=(X_test_lon, y_test_lon))

# Step 19: Make predictions and evaluate
print("Step 19: Making predictions and evaluating models...")
y_pred_latitude = model_latitude.predict(X_test_lat)
y_pred_longitude = model_longitude.predict(X_test_lon)

# Evaluate the models using Mean Squared Error and R-squared
mse_latitude = mean_squared_error(y_test_lat, y_pred_latitude)
r2_latitude = r2_score(y_test_lat, y_pred_latitude)
mse_longitude = mean_squared_error(y_test_lon, y_pred_longitude)
r2_longitude = r2_score(y_test_lon, y_pred_longitude)

print(f'Mean Squared Error (Latitude): {mse_latitude}')
print(f'R-squared (Latitude): {r2_latitude}')
print(f'Mean Squared Error (Longitude): {mse_longitude}')
print(f'R-squared (Longitude): {r2_longitude}')

# Step 20: Predict for test set and create submission
print("Step 20: Predicting for test set and creating submission file...")
def create_sequences_with_padding(data, sequence_length=10):
    sequences = []
    for i in range(len(data)):
        start_idx = max(0, i - sequence_length + 1)
        seq = data.iloc[start_idx:i + 1].values
        if len(seq) < sequence_length:
            seq = np.pad(seq, ((sequence_length - len(seq), 0), (0, 0)), mode='constant')
        sequences.append(seq)
    return np.array(sequences)

X_test_sequences = create_sequences_with_padding(ais_test[features], sequence_length)

# Predict latitude and longitude using the LSTM models
latitude_predictions = model_latitude.predict(X_test_sequences)
longitude_predictions = model_longitude.predict(X_test_sequences)

# Load the sample submission file and save the predictions
sample_submission = pd.read_csv('../ais_sample_submission.csv')
sample_submission['latitude_predicted'] = latitude_predictions.flatten()
sample_submission['longitude_predicted'] = longitude_predictions.flatten()

# Save the final submission file
sample_submission.to_csv('submission_lstm_combined1.csv', index=False)
print("Final submission file saved as 'submission_lstm_combined1.csv'.\n")

# Display the first few rows of the submission file
print(sample_submission.head())
