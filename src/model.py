import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    Builds an LSTM model for stock price prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # Prevents overfitting
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)  # Predicting a single output (future stock price)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

if __name__ == "__main__":
    # Example usage
    model = build_lstm_model((60, 1))  # Assuming 60 timesteps and 1 feature
    model.summary()  # Prints the model architecture
