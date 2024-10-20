import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Load datasets
vessel_data = pd.read_csv('../ais_train.csv', sep='|')
schedule_data = pd.read_csv('../schedules_to_may_2024.csv', sep='|')
ports_data = pd.read_csv('../ports.csv', sep='|')
vessels_info = pd.read_csv('../vessels.csv', sep='|')

# Step 1: Convert 'time' column to datetime in vessel_data
vessel_data['time'] = pd.to_datetime(vessel_data['time'])

# Step 2: Get the latest vessel data for each vessel (optional step)
latest_vessel_data = vessel_data.loc[vessel_data.groupby('vesselId')['time'].idxmax()]

# Feature extraction: Create time-based features (hour, day_of_week, month) from the 'time' column
vessel_data['hour'] = vessel_data['time'].dt.hour
vessel_data['day_of_week'] = vessel_data['time'].dt.dayofweek
vessel_data['month'] = vessel_data['time'].dt.month

###################### new part

from geopy.distance import geodesic

# Calculate sailing velocity (distance between positions divided by time difference)
vessel_data['prev_latitude'] = vessel_data['latitude'].shift(1)
vessel_data['prev_longitude'] = vessel_data['longitude'].shift(1)
vessel_data['prev_time'] = vessel_data['time'].shift(1)

# Compute time difference in hours
vessel_data['time_diff'] = (vessel_data['time'] - vessel_data['prev_time']).dt.total_seconds() / 3600

# Calculate the distance traveled (in kilometers)
vessel_data['distance_traveled'] = vessel_data.apply(lambda row: geodesic(
    (row['prev_latitude'], row['prev_longitude']), (row['latitude'], row['longitude'])
).kilometers if pd.notnull(row['prev_latitude']) else np.nan, axis=1)

# Calculate sailing velocity (km/h)
vessel_data['sailing_velocity'] = vessel_data['distance_traveled'] / vessel_data['time_diff']

# Merge vessel info into the main dataset
vessel_data = pd.merge(vessel_data, vessels_info[['vesselId', 'CEU', 'length', 'maxSpeed']], on='vesselId', how='left')

# Merge ports data to get port locations
vessel_data = pd.merge(vessel_data, ports_data[['portId', 'latitude', 'longitude']], left_on='portId', right_on='portId', how='left')

# Calculate the distance to the nearest port (or the destination port)
vessel_data['distance_to_port'] = vessel_data.apply(lambda row: geodesic(
    (row['latitude'], row['longitude']), (row['latitude_y'], row['longitude_y'])
).kilometers if pd.notnull(row['latitude_y']) else np.nan, axis=1)

# Merge schedule data to include planned arrival information
vessel_data = pd.merge(vessel_data, schedule_data[['vesselId', 'arrivalDate']], on='vesselId', how='left')

# Create time until scheduled arrival feature
vessel_data['time_until_arrival'] = (vessel_data['arrivalDate'] - vessel_data['time']).dt.total_seconds() / 3600  # Time in hours


#########################

# Prepare the dataset for training: Only use columns that will be available in the test set
# X = vessel_data[['hour', 'day_of_week', 'month']]
# Prepare the dataset for training: Adding more features
X = vessel_data[['hour', 'day_of_week', 'month', 'sailing_velocity', 'CEU', 'length', 'maxSpeed', 'distance_to_port', 'time_until_arrival']]

y_latitude = vessel_data['latitude']  # Target for latitude
y_longitude = vessel_data['longitude']  # Target for longitude

# Split the data into training and test sets
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X, y_latitude, test_size=0.2, random_state=42)
X_train_lon, X_test_lon, y_train_lon, y_test_lon = train_test_split(X, y_longitude, test_size=0.2, random_state=42)

# Train the model for latitude prediction
model_latitude = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_latitude.fit(X_train_lat, y_train_lat)

# Make predictions on the test set for latitude
y_pred_latitude = model_latitude.predict(X_test_lat)

# Evaluate the latitude model
mse_latitude = mean_squared_error(y_test_lat, y_pred_latitude)
r2_latitude = r2_score(y_test_lat, y_pred_latitude)

print(f'Mean Squared Error (Latitude): {mse_latitude}')
print(f'R-squared (Latitude): {r2_latitude}')

# Train the model for longitude prediction
model_longitude = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_longitude.fit(X_train_lon, y_train_lon)

# Make predictions on the test set for longitude
y_pred_longitude = model_longitude.predict(X_test_lon)

# Evaluate the longitude model
mse_longitude = mean_squared_error(y_test_lon, y_pred_longitude)
r2_longitude = r2_score(y_test_lon, y_pred_longitude)

print(f'Mean Squared Error (Longitude): {mse_longitude}')
print(f'R-squared (Longitude): {r2_longitude}')

# Load the test data and make sure to only extract the available features (hour, day_of_week, month)
ais_test = pd.read_csv('../ais_test.csv', sep=',')

# Convert 'time' column to datetime in test data
ais_test['time'] = pd.to_datetime(ais_test['time'])

# Create time-based features for test data
ais_test['hour'] = ais_test['time'].dt.hour
ais_test['day_of_week'] = ais_test['time'].dt.dayofweek
ais_test['month'] = ais_test['time'].dt.month

# Select features for prediction
X_test_limited = ais_test[['hour', 'day_of_week', 'month']]

# Make predictions for latitude and longitude using the limited features model
latitude_predictions = model_latitude.predict(X_test_limited)
longitude_predictions = model_longitude.predict(X_test_limited)

# Load the sample submission file
sample_submission = pd.read_csv('../ais_sample_submission.csv')

# Fill in the predicted latitude and longitude
sample_submission['latitude_predicted'] = latitude_predictions
sample_submission['longitude_predicted'] = longitude_predictions

# Save the filled submission file
sample_submission.to_csv('submission02.csv', index=False)

# Display the first few rows of the submission file to verify
print(sample_submission.head())
