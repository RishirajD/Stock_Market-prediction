import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import glob
import joblib
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    # Extract stock name from filename
    stock_name = os.path.basename(file_path).split('-')[0]
    return df, stock_name

# Load all datasets
def load_all_datasets(data_dir='.'):
    # Get all CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*-BSE.csv'))
    all_data = {}
    
    for file_path in csv_files:
        df, stock_name = load_data(file_path)
        all_data[stock_name] = df
        print(f"Loaded {stock_name} data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return all_data

# Create feature engineering function
def create_features(df, target_col='Close', window_size=5):
    """Create technical indicators and features for time series prediction"""
    df_copy = df.copy()
    
    # Rename target column for clarity
    df_copy['target'] = df_copy[target_col]
    
    # Technical indicators
    # 1. Moving Averages
    df_copy['MA5'] = df_copy[target_col].rolling(window=5).mean()
    df_copy['MA10'] = df_copy[target_col].rolling(window=10).mean()
    df_copy['MA20'] = df_copy[target_col].rolling(window=20).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df_copy[target_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    ema12 = df_copy[target_col].ewm(span=12, adjust=False).mean()
    ema26 = df_copy[target_col].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = ema12 - ema26
    df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands
    df_copy['BB_middle'] = df_copy[target_col].rolling(window=20).mean()
    df_copy['BB_std'] = df_copy[target_col].rolling(window=20).std()
    df_copy['BB_upper'] = df_copy['BB_middle'] + 2 * df_copy['BB_std']
    df_copy['BB_lower'] = df_copy['BB_middle'] - 2 * df_copy['BB_std']
    df_copy['BB_width'] = (df_copy['BB_upper'] - df_copy['BB_lower']) / df_copy['BB_middle']
    
    # 5. Price changes and momentum
    df_copy['price_change'] = df_copy[target_col].pct_change()
    df_copy['momentum'] = df_copy[target_col] - df_copy[target_col].shift(window_size)
    
    # 6. Volatility
    df_copy['volatility'] = df_copy[target_col].rolling(window=20).std()
    
    # 7. Volume-based indicators
    if 'Volume' in df_copy.columns:
        df_copy['volume_change'] = df_copy['Volume'].pct_change()
        df_copy['volume_ma5'] = df_copy['Volume'].rolling(window=5).mean()
        df_copy['volume_ma10'] = df_copy['Volume'].rolling(window=10).mean()
        df_copy['volume_ma_ratio'] = df_copy['Volume'] / df_copy['volume_ma5']
    
    # 8. Lag features (previous days' prices)
    for i in range(1, window_size+1):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 9. Price range features
    df_copy['high_low_ratio'] = df_copy['High'] / df_copy['Low']
    df_copy['close_open_ratio'] = df_copy[target_col] / df_copy['Open']
    
    # 10. Moving Average Crossovers
    df_copy['ma_crossover_5_20'] = df_copy['MA5'] - df_copy['MA20']
    
    # Add day of week, month, and year as features
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year
    
    # Drop rows with NaN values (due to rolling windows)
    df_copy.dropna(inplace=True)
    
    return df_copy

# Function for time series train-test split (respecting time order)
def time_series_split(data, target_col='target', test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Separate features and target
    X_train = train_data.drop(['target'], axis=1)
    y_train = train_data['target']
    X_test = test_data.drop(['target'], axis=1)
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test

# Function for LSTM data preparation
def prepare_lstm_data(data, target_col='target', seq_length=30):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_length, len(data_scaled)):
        X.append(data_scaled[i-seq_length:i])
        # Find the index of the target column
        target_idx = data.columns.get_loc(target_col)
        y.append(data_scaled[i, target_idx])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Create a visualization directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Function to build and evaluate models for a stock
def build_stock_prediction_model(stock_name, stock_data, feature_window=10, lstm_seq_length=30):
    print(f"\n{'='*80}")
    print(f"Building prediction models for {stock_name}")
    print(f"{'='*80}")
    
    # Apply feature engineering
    data = create_features(stock_data, target_col='Close', window_size=feature_window)
    
    # Display basic information
    print(f"\nFeature engineered data shape: {data.shape}")
    print(f"Features created: {list(data.columns)}")
    
    # Plot correlation heatmap
    plt.figure(figsize=(15, 12))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f'{stock_name} - Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'visualizations/{stock_name}_correlation_heatmap.png')
    plt.close()
    
    # Prepare data for ML models
    X_train, X_test, y_train, y_test = time_series_split(data, test_size=0.2)
    
    # Drop unnecessary columns for training
    drop_cols = ['target', 'day_of_week', 'month', 'year']
    features = [col for col in X_train.columns if col not in drop_cols]
    
    X_train = X_train[features]
    X_test = X_test[features]
    
    print(f"\nTraining with {len(features)} features")
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Define multiple models for comparison
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'model': model
        }
        
        print(f"{name} Results:")
        print(f"R² Score (Accuracy): {r2 * 100:.2f}%")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=importance.head(15))
            plt.title(f'{stock_name} - Feature Importance for {name}')
            plt.tight_layout()
            plt.savefig(f'visualizations/{stock_name}_{name.replace(" ", "_")}_feature_importance.png')
            plt.close()
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    # Select columns for LSTM (use a subset of features to avoid dimensionality issues)
    lstm_cols = ['Close', 'Volume', 'Open', 'High', 'Low', 'MA5', 'MA20', 'RSI', 'MACD', 'price_change']
    lstm_data = data[lstm_cols + ['target']]
    
    # Prepare sequence data for LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(
        lstm_data, target_col='target', seq_length=lstm_seq_length
    )
    
    # Build LSTM model
    model_lstm = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train LSTM model
    history = model_lstm.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate LSTM model
    y_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
    
    # Inverse transform the predictions
    y_test_inv = np.zeros((len(y_test_lstm), len(lstm_cols) + 1))
    y_pred_inv = np.zeros((len(y_pred_lstm_scaled), len(lstm_cols) + 1))
    
    # Set the target column to the predicted values
    target_idx = lstm_data.columns.get_loc('target')
    y_test_inv[:, target_idx] = y_test_lstm
    y_pred_inv[:, target_idx] = y_pred_lstm_scaled.flatten()
    
    # Inverse transform to get actual values
    y_test_actual = scaler.inverse_transform(y_test_inv)[:, target_idx]
    y_pred_actual = scaler.inverse_transform(y_pred_inv)[:, target_idx]
    
    # Calculate metrics
    mae_lstm = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2_lstm = r2_score(y_test_actual, y_pred_actual)
    
    print("\nLSTM Model Results:")
    print(f"R² Score (Accuracy): {r2_lstm * 100:.2f}%")
    print(f"Mean Absolute Error: {mae_lstm:.2f}")
    print(f"Root Mean Squared Error: {rmse_lstm:.2f}")
    
    # Add LSTM results to the results dictionary
    results['LSTM'] = {
        'mae': mae_lstm,
        'rmse': rmse_lstm,
        'r2': r2_lstm,
        'predictions': y_pred_actual,
        'model': model_lstm,
        'scaler': scaler,
        'lstm_data_columns': lstm_data.columns,
        'history': history
    }
    
    # Plot LSTM training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stock_name} - LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_name}_lstm_training_history.png')
    plt.close()

    # Visualize the results
    plt.figure(figsize=(15, 7))
    plt.plot(data.index[-len(y_test):], y_test.values, label='Actual', color='blue')

    for name, result in results.items():
        if name == 'LSTM':
            plt.plot(data.index[-len(y_test_actual):], result['predictions'], 
                    label=f'Predicted ({name})', linestyle='--')
        else:
            plt.plot(data.index[-len(y_test):], result['predictions'], 
                    label=f'Predicted ({name})', linestyle='--')

    plt.title(f'{stock_name} - Stock Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_name}_model_comparison.png')
    plt.close()

    # Compare models
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] * 100 for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score (%)', color='green')
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='red')

    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score (%)', color='green')
    ax2.set_ylabel('RMSE', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.set_ylim(0, 100)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'{stock_name} - Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(f'visualizations/{stock_name}_model_metrics_comparison.png')
    plt.close()

    # Make future predictions (next 30 days)
    def predict_future(model, features, last_data, days=30, model_type='tree'):
        """Predict future stock prices using the trained model"""
        last_date = last_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        
        # Create a dataframe to store predictions
        future_df = pd.DataFrame(index=future_dates)
        future_df['Predicted_Price'] = np.nan
        
        # Get the last available data points
        last_known_data = last_data.iloc[-1:].copy()
        
        if model_type == 'tree':
            # For tree-based models
            for i in range(len(future_dates)):
                # Use only the features the model was trained on
                prediction = model.predict(last_known_data[features].values)[0]
                
                # Update the dataframe with the prediction
                future_df.iloc[i, 0] = prediction
                
                # Update the "last known data" for the next iteration
                new_row = last_known_data.copy()
                new_row.index = [future_dates[i]]
                new_row['Close'] = prediction  # Update Close price
                
                # Simple update of lag features and technical indicators
                # This is simplified; in a production system you'd need to update all features
                for j in range(feature_window, 0, -1):
                    if f'lag_{j}' in new_row.columns:
                        if j == 1:
                            new_row[f'lag_{j}'] = last_known_data['Close'].values[0]
                        else:
                            new_row[f'lag_{j}'] = last_known_data[f'lag_{j-1}'].values[0]
                
                last_known_data = new_row
        
        elif model_type == 'lstm':
            # For LSTM model
            scaler = results['LSTM']['scaler']
            lstm_cols = results['LSTM']['lstm_data_columns']
            seq_length = lstm_seq_length
            
            # Get the last sequence of data for prediction
            last_sequence = last_data[lstm_cols].values[-seq_length:]
            last_sequence_scaled = scaler.transform(last_sequence)
            
            for i in range(len(future_dates)):
                # Reshape for LSTM input
                current_sequence = last_sequence_scaled[-seq_length:].reshape(1, seq_length, len(lstm_cols))
                
                # Predict next value
                next_pred_scaled = model.predict(current_sequence)
                
                # Create a template for inverse scaling
                next_pred_template = np.zeros((1, len(lstm_cols)))
                target_idx = np.where(lstm_cols == 'target')[0][0]
                next_pred_template[0, target_idx] = next_pred_scaled[0, 0]
                
                # Inverse transform to get actual price
                next_pred_actual = scaler.inverse_transform(next_pred_template)[0, target_idx]
                
                # Add to future predictions
                future_df.iloc[i, 0] = next_pred_actual
                
                # Update sequence for next prediction
                new_row = np.zeros_like(last_sequence_scaled[0])
                new_row[target_idx] = next_pred_scaled[0, 0]  # Set target (Close price)
                
                # Add the new prediction to the sequence and remove the oldest entry
                last_sequence_scaled = np.vstack([last_sequence_scaled[1:], new_row])
        
        return future_df

    # Identify the best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    print(f"\nBest model for {stock_name}: {best_model_name} with R² Score: {results[best_model_name]['r2'] * 100:.2f}%")
    
    # Make future predictions using the best model
    if best_model_name != 'LSTM':
        best_model = results[best_model_name]['model']
        future_predictions = predict_future(best_model, features, data, days=30, model_type='tree')
    else:
        best_model = results[best_model_name]['model']
        future_predictions = predict_future(best_model, None, data, days=30, model_type='lstm')
    
    plt.figure(figsize=(15, 7))
    plt.plot(data.index[-60:], data['Close'][-60:], label='Historical Prices', color='blue')
    plt.plot(future_predictions.index, future_predictions['Predicted_Price'], 
             label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{stock_name} - Future Stock Price Prediction ({best_model_name})')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_name}_future_predictions.png')
    plt.close()
    
    print("\nFuture Predictions:")
    print(future_predictions)
    
    # Save the best model
    if best_model_name != 'LSTM':
        joblib.dump(best_model, f'models/{stock_name}_best_model_{best_model_name}.pkl')
        print(f"\nBest model saved as 'models/{stock_name}_best_model_{best_model_name}.pkl'")
    else:
        best_model.save(f'models/{stock_name}_best_model_LSTM.h5')
        # Also save the scaler
        joblib.dump(results['LSTM']['scaler'], f'models/{stock_name}_lstm_scaler.pkl')
        print(f"\nBest model saved as 'models/{stock_name}_best_model_LSTM.h5'")
    
    return results, future_predictions

# Main execution function
def main():
    print("Stock Market Prediction Model")
    print("="*80)
    
    # Load all stock datasets
    all_stocks = load_all_datasets()
    
    # Summarize loaded data
    print(f"\nLoaded {len(all_stocks)} stocks: {list(all_stocks.keys())}")
    
    # Store all results
    all_results = {}
    future_predictions = {}
    
    # Build models for each stock
    for stock_name, stock_data in all_stocks.items():
        try:
            results, predictions = build_stock_prediction_model(stock_name, stock_data)
            all_results[stock_name] = results
            future_predictions[stock_name] = predictions
        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
    
    # Create summary of performance across all stocks
    summary = []
    for stock_name, results in all_results.items():
        for model_name, metrics in results.items():
            if model_name != 'LSTM' or (model_name == 'LSTM' and 'r2' in metrics):
                summary.append({
                    'Stock': stock_name,
                    'Model': model_name,
                    'R2_Score': metrics['r2'] * 100,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse']
                })
    
    summary_df = pd.DataFrame(summary)
    
    # Aggregate results by model type
    model_summary = summary_df.groupby('Model').agg({
        'R2_Score': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max']
    })
    
    print("\nModel Performance Summary Across All Stocks:")
    print(model_summary)
    
    # Save summary to CSV
    summary_df.to_csv('model_performance_summary.csv', index=False)
    model_summary.to_csv('model_summary_by_type.csv')
    
    # Create visualization of model performance across stocks
    plt.figure(figsize=(15, 10))
    
    # Plot R2 Scores by Stock and Model
    sns.barplot(x='Stock', y='R2_Score', hue='Model', data=summary_df)
    plt.title('Model R² Scores Across Stocks')
    plt.xlabel('Stock')
    plt.ylabel('R² Score (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('visualizations/all_stocks_r2_comparison.png')
    plt.close()
    
    # Plot RMSE by Stock and Model
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Stock', y='RMSE', hue='Model', data=summary_df)
    plt.title('Model RMSE Across Stocks')
    plt.xlabel('Stock')
    plt.ylabel('Root Mean Squared Error')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('visualizations/all_stocks_rmse_comparison.png')
    plt.close()
    
    # Combine all future predictions into one DataFrame for comparison
    all_predictions = pd.DataFrame()
    for stock_name, preds in future_predictions.items():
        all_predictions[stock_name] = preds['Predicted_Price']
    
    # Calculate percentage change from first prediction
    pct_change = all_predictions.pct_change(periods=len(all_predictions)-1, axis=0).iloc[-1].sort_values()
    
    # Plot predicted percentage changes
    plt.figure(figsize=(12, 8))
    sns.barplot(x=pct_change.index, y=pct_change.values * 100)
    plt.title('Predicted 30-Day Percentage Change by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Predicted % Change')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('visualizations/predicted_percentage_changes.png')
    plt.close()
    
    print("\nAnalysis complete! Check the 'visualizations' directory for insights.")
    print("Models are saved in the 'models' directory.")
    
    return all_results, future_predictions, summary_df

if __name__ == "__main__":
    main()