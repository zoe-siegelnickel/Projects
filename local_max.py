import yfinance as yf
import pandas as pd
import requests # Import the requests library
from io import StringIO # Import StringIO
from scipy.signal import argrelmin, argrelmax
import matplotlib.pyplot as plt

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
# Use StringIO to wrap the response text to avoid the FutureWarning
tickers = pd.read_html(StringIO(response.text))[0]
print(tickers.head())

data = yf.download(tickers.Symbol.to_list(),'2026-1-1','2026-9-25', auto_adjust=True)['High']

#data.to_csv('stock_data.csv', index= False)

local_maxima_results = {}
local_minima_results = {}
# Let's clean up any columns with all NaN values first if they exist
clean_data = data.dropna(axis=1, how='all')

#Find the local maxima and minima for each ticker
for ticker in clean_data.columns:
    # Get the non-null values for the current ticker
    ticker_series = clean_data[ticker].dropna()

    # Find indices of local maxima
    max_indices = argrelmax(ticker_series.values, order=5)[0]
    min_indices = argrelmin(ticker_series.values, order=5)[0]

    # Extract corresponding dates and values
    max_dates = ticker_series.index[max_indices]
    max_values = ticker_series.iloc[max_indices].values
    min_dates = ticker_series.index[min_indices]
    min_values = ticker_series.iloc[min_indices].values

    # Store as a DataFrame for easy viewing
    local_maxima_results[ticker] = pd.DataFrame({
        'Date': max_dates,
        'Local_Maximum_Price': max_values
    })

    local_minima_results[ticker] = pd.DataFrame({
        'Date': min_dates,
        'Local_Minimum_Price': min_values
    })
#find companies that have cocurrent local maxima and local minima
def get_cooccurring_extremes(target_ticker, order=5):
    # Ensure data is loaded
    global data
    if 'data' not in globals() and 'data' not in locals():
        if os.path.exists('stock_data.csv'):
            data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)
        else: 
            raise FileNotFoundError("Please run the first cell to download and save 'stock_data.csv' first.")
    
    clean_data = data.dropna(axis=1, how='all')
    
    if target_ticker not in clean_data.columns:
        print(f"Ticker {target_ticker} not found in the dataset.")
        return [], []
    
    # Get target series and its local maxima dates
    target_series = clean_data[target_ticker].dropna()
    target_max_idx = argrelmax(target_series.values, order=order)[0]
    target_max_dates = set(target_series.index[target_max_idx])
    
    if not target_max_dates:
        print(f"No local maxima found for {target_ticker} with order={order}.")
        return [], []
        
    co_maxima_tickers = set()
    co_minima_tickers = set()
    
    # Iterate through all other tickers to find overlapping peak/trough dates
    for ticker in clean_data.columns:
        if ticker == target_ticker:
            continue
        
        ticker_series = clean_data[ticker].dropna()
        if ticker_series.empty:
            continue
            
        # Local Maxima for current ticker
        max_idx = argrelmax(ticker_series.values, order=order)[0]
        max_dates = set(ticker_series.index[max_idx])
        
        # Local Minima for current ticker
        min_idx = argrelmin(ticker_series.values, order=order)[0]
        min_dates = set(ticker_series.index[min_idx])
        
        # Check if they share any dates with target ticker's maxima
        if not target_max_dates.isdisjoint(max_dates):
            co_maxima_tickers.add(ticker)
        if not target_max_dates.isdisjoint(min_dates):
            co_minima_tickers.add(ticker)
            
    return list(co_maxima_tickers), list(co_minima_tickers)

#example: 
target = 'AAPL'
max_co, min_co = get_cooccurring_extremes(target, order=5)

print(f"Results for {target}:")
print(f"Number of tickers with co-occurring local maxima: {len(max_co)}")
print(f"Sample of tickers peaking on {target}'s peak days: {max_co[:15]}")
print(f"\nNumber of tickers with co-occurring local minima: {len(min_co)}")
print(f"Sample of tickers bottoming on {target}'s peak days: {min_co[:15]}")