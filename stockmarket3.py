# import pandas as pd
# import numpy as np
# import os
# import glob
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load all datasets
# data_path = "D:/sih/archive/mammals"  # Update this path accordingly
# csv_files = glob.glob(os.path.join(data_path, "*.csv"))

# # Combine datasets
# dataframes = []
# for file in csv_files:
#     df = pd.read_csv(file, parse_dates=['date'])
#     df['Stock'] = os.path.basename(file).replace("-BSE.csv", "")
#     dataframes.append(df)

# data = pd.concat(dataframes, ignore_index=True)

# data.sort_values(by=['Stock', 'date'], inplace=True)

# # Feature Engineering
# data['SMA_10'] = data.groupby('Stock')['close'].transform(lambda x: x.rolling(window=10).mean())
# data['SMA_50'] = data.groupby('Stock')['close'].transform(lambda x: x.rolling(window=50).mean())
# data['Volatility'] = data.groupby('Stock')['close'].transform(lambda x: x.pct_change().rolling(10).std())
# data.dropna(inplace=True)

# # Prepare train and test sets
# features = ['open', 'high', 'low', 'volume', 'SMA_10', 'SMA_50', 'Volatility']
# target = 'close'

# X = data[features]
# y = data[target]

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train Model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate Model
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print(f"MAE: {mae}, RMSE: {rmse}")

# # Plot predictions
# plt.figure(figsize=(10,5))
# plt.plot(y_test.values, label='Actual', alpha=0.7)
# plt.plot(y_pred, label='Predicted', alpha=0.7)
# plt.legend()
# plt.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++==========
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Correct the data path
data_path = "D:/tarinning_material/stockdata"  # Updated to your dataset location
csv_files = glob.glob(os.path.join(data_path, "*.csv"))

# Check if files are loaded
if not csv_files:
    print("No CSV files found in the directory. Please check the path.")
else:
    print(f"Found {len(csv_files)} CSV files.")

# Load and process datasets
dataframes = []
for file in csv_files:
    df = pd.read_csv(file, parse_dates=['date'])  # Ensure 'date' column is in datetime format
    df['Stock'] = os.path.basename(file).replace("-BSE.csv", "")  # Extract stock name
    dataframes.append(df)

# Combine all datasets
data = pd.concat(dataframes, ignore_index=True)
data.sort_values(by=['Stock', 'date'], inplace=True)

# Drop unnecessary columns
data.drop(columns=['dividend_amount'], inplace=True)

# Feature Engineering
data['SMA_10'] = data.groupby('Stock')['close'].transform(lambda x: x.rolling(window=10).mean())
data['SMA_50'] = data.groupby('Stock')['close'].transform(lambda x: x.rolling(window=50).mean())
data['Volatility'] = data.groupby('Stock')['close'].transform(lambda x: x.pct_change().rolling(10).std())
data.dropna(inplace=True)

# Prepare train and test sets
features = ['open', 'high', 'low', 'volume', 'SMA_10', 'SMA_50', 'Volatility']
target = 'close'

X = data[features]
y = data[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost Model
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost - MAE: {mae_xgb}, RMSE: {rmse_xgb}")

# Reshape data for LSTM
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Build LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# Evaluate LSTM Model
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
print(f"LSTM - MAE: {mae_lstm}, RMSE: {rmse_lstm}")
