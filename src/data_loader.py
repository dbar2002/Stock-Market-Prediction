import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol: str, start_date: str, end_date: str, save_csv=True, data_dir="data"):
    """
    Fetches historical stock data from Yahoo Finance.

    Parameters:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_csv (bool): Whether to save the data to a CSV file.
        data_dir (str): Directory to save the file.

    Returns:
        pd.DataFrame: The historical stock data.
    """
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            print(f"Warning: No data found for {symbol}. Check the ticker and dates.")
            return None
        
        # Create data directory if not exists
        if save_csv:
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, f"{symbol}_stock_data.csv")
            stock_data.to_csv(file_path)
            print(f"Data saved to {file_path}")

        return stock_data

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    data = fetch_stock_data(symbol, start_date, end_date)
    print(data.head())
