import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import fetch_stock_data
from model import build_lstm_model

# Configuration
STOCK_SYMBOL = "AAPL"  # Change to your preferred stock
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
DATA_PATH = f"data/{STOCK_SYMBOL}_stock_data.csv"
MODEL_SAVE_PATH = f"models/{STOCK_SYMBOL}_lstm_model.h5"
TIME_STEP = 60  # Number of past days used for prediction

# Ensure directories exist
os.makedirs("models", exist_ok=True)

def preprocess_data(df):
    """
    Prepares stock data for LSTM training.

    Parameters:
        df (pd.DataFrame): Raw stock data.

    Returns:
        X_train, Y_train: Training data for LSTM.
        scaler: Scaler object for inverse transformations.
    """
    df = df[['Close']]  # Use only closing price
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)

    # Create time series sequences
    X, Y = [], []
    for i in range(len(df_scaled) - TIME_STEP):
        X.append(df_scaled[i:i + TIME_STEP, 0])
        Y.append(df_scaled[i + TIME_STEP, 0])
    
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, Y, scaler

def train_model():
    """
    Trains the LSTM model on stock data.
    """
    # Load or fetch stock data
    if not os.path.exists(DATA_PATH):
        print("Fetching stock data...")
        df = fetch_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    else:
        print("Loading existing stock data...")
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)

    if df is None or df.empty:
        print("Error: No data available.")
        return

    # Preprocess the data
    X_train, Y_train, scaler = preprocess_data(df)

    # Build the model
    model = build_lstm_model((TIME_STEP, 1))

    # Set up callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='loss', mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Train the model
    print("Training the model...")
    model.fit(X_train, Y_train, epochs=50, batch_size=32, callbacks=[checkpoint, early_stop])

    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
