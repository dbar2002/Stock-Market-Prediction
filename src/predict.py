import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data_loader import fetch_stock_data
from model import build_lstm_model

# Configuration
STOCK_SYMBOL = "AAPL"  # Change this to your preferred stock
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
DATA_PATH = f"data/{STOCK_SYMBOL}_stock_data.csv"
MODEL_PATH = f"models/{STOCK_SYMBOL}_lstm_model.h5"
TIME_STEP = 60  # Number of past days used for prediction

def load_and_preprocess_data():
    """
    Loads and preprocesses stock data for making predictions.

    Returns:
        np.array: Preprocessed input data for prediction.
        scaler: Scaler object to reverse transformations.
    """
    if not os.path.exists(DATA_PATH):
        print("Stock data not found. Fetching new data...")
        df = fetch_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    else:
        print("Loading existing stock data...")
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)

    if df is None or df.empty:
        print("Error: No stock data available.")
        return None, None

    df = df[['Close']]  # Use only closing prices
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)

    # Prepare the last TIME_STEP days for prediction
    last_X = df_scaled[-TIME_STEP:]
    last_X = np.reshape(last_X, (1, TIME_STEP, 1))  # Reshape for LSTM input

    return last_X, scaler

def predict_next_price():
    """
    Loads the trained model and predicts the next stock price.
    """
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found. Train the model first.")
        return

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load and preprocess data
    last_X, scaler = load_and_preprocess_data()
    if last_X is None:
        return

    # Make prediction
    predicted_price = model.predict(last_X)
    predicted_price = scaler.inverse_transform(predicted_price)  # Convert back to original scale

    print(f"Predicted Next Day Stock Price for {STOCK_SYMBOL}: ${predicted_price[0][0]:.2f}")

if __name__ == "__main__":
    predict_next_price()
