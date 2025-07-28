import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# Create results folder
os.makedirs("results", exist_ok=True)

# Loop over all preprocessed CSV files
data_folder = 'stock_data'  # or the exact folder name
sequence_length = 60

for filename in os.listdir(data_folder):
    if not filename.endswith(".csv"):
        continue

    stock_name = filename.replace(".csv", "")
    filepath = os.path.join(data_folder, filename)
    df = pd.read_csv(filepath)

    if 'Close' not in df.columns:
        print(f"Skipping {filename}: 'Close' column not found.")
        continue

    # Sort by date and reset index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Normalize Close prices
    scaler = MinMaxScaler()
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])

    # Create sequences
    X, y, dates = [], [], []
    for i in range(sequence_length, len(df)):
        X.append(df['Close_scaled'].values[i-sequence_length:i])
        y.append(df['Close_scaled'].values[i])
        dates.append(df['Date'].iloc[i])

    X, y = np.array(X), np.array(y)

    # Split into train/test (80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_dates = dates[split_index:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE and R2
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test_actual, color='red', label='Actual')
    plt.plot(test_dates, y_pred, color='blue', label='Predicted')
    plt.title(f"{stock_name} Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join("results", f"{stock_name}_prediction_plot.png")
    plt.savefig(save_path)
    plt.close()

    # Print results
    print(f"\n📈 Stock: {stock_name}")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.4f}")
    print(f"📅 Test Date Range: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"📁 Graph saved as: {stock_name}_prediction_plot.png")

    # Get user input for prediction
    date_input = input("📅 Enter date in format YYYY-MM-DD to get predicted price (or press Enter to skip): ")
    if date_input:
        try:
            date_obj = pd.to_datetime(date_input)
            if date_obj in test_dates:
                idx = test_dates.index(date_obj)
                predicted_price = y_pred[idx][0]
                print(f"💰 Predicted Close Price on {date_input}: ₹{predicted_price:.2f}")
            else:
                print("❌ Date not found in test set.")
        except:
            print("⚠️ Invalid date format.")
