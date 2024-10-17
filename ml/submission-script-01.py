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

# Prepare the dataset for training: Only use columns that will be available in the test set
X = vessel_data[['hour', 'day_of_week', 'month']]
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
sample_submission.to_csv('submission01.csv', index=False)

# Display the first few rows of the submission file to verify
print(sample_submission.head())
