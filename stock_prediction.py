import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
# Step 1: Download Stock Data from Yahoo Finance
ticker = "TATAMOTORS.NS"  # Infosys stock ticker on NSE (Change if needed)
data = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=True)
data = data.reset_index()  # Reset index to include 'Date' column
data.to_csv("tata_stock.csv", index=False)  # Save as CSV with updated filename

# Step 2: Load Data from CSV
df = pd.read_csv("tata_stock.csv", parse_dates=["Date"])

# Step 3: Handle Missing Values (NaN)
df.ffill(inplace=True)  # Forward fill NaN values
df.dropna(inplace=True)  # Ensure no missing values remain

# Step 4: Convert Date to Numerical Format
df['Day'] = (df['Date'] - df['Date'].min()).dt.days  # Convert dates to numerical values

# Step 5: Select Features
X = df[['Day']]  # Independent variable (Days)
y = df['Close']  # Dependent variable (Stock closing price)

# Step 6: Ensure X has no NaN values before training
if X.isnull().values.any() or y.isnull().values.any():
    print("Error: X or y contains NaN values. Exiting.")
    exit()

# Step 7: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"R² Score (Accuracy): {r2 * 100:.2f}%")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Step 11: Visualize the Results
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.legend()
plt.show()
