# on any given day, prodive a list of stocks that are at a local minima and do least squares regression. 
# return a list of stocks that are likely to go up in a given time period.  
import yfinance as yf
import pandas as pd
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import urllib.request
import time
import requests
from io import StringIO  # Απαραίτητο για τη διόρθωση του ValueError
from tqdm import notebook



def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI) for a series of prices."""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def find_minima_by_indicators(tickers, period="3mo"):
    """
    Scans a list of tickers to find those that are oversold (RSI <= 30)
    and near/below their Lower Bollinger Band (indicating a local bottom).
    """
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if len(df) < 20:
                continue

            close_prices = df['Close'].values
            todays_close = close_prices[-1]

            # 1. Calculate RSI (14-day)
            rsi_values = calculate_rsi(close_prices, period=14)
            todays_rsi = rsi_values[-1]

            # 2. Calculate Bollinger Bands (20-day Simple Moving Average +/- 2 Standard Deviations)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['STD_20'] = df['Close'].rolling(window=20).std()
            df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
            
            todays_lower_bb = df['Lower_BB'].iloc[-1]

            # We flag a local minimum if Today's Close is near/below Lower BB AND RSI is low (< 35)
            is_oversold = todays_rsi <= 35
            is_at_lower_band = todays_close <= (todays_lower_bb * 1.02) # within 2% of the lower band

            if is_oversold or is_at_lower_band:
                results.append({
                    'Ticker': ticker,
                    'Today Close': round(todays_close, 2),
                    'RSI (14)': round(todays_rsi, 2),
                    'Lower BB': round(todays_lower_bb, 2),
                    'Distance to Lower BB (%)': round(((todays_close - todays_lower_bb) / todays_lower_bb) * 100, 2)
                })
        except Exception as e:
            print(f"Could not process {ticker}: {e}")

    return pd.DataFrame(results)

# Example list of tickers to scan
test_tickers = ["BENF", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "T", "NKE", "PFE"]

# Run the indicator-based scanner
minima_df = find_minima_by_indicators(test_tickers)
#display(minima_df)


def linear_trend(x, m, c):
    return m * x + c

def analyze_trends_least_squares(tickers, period="30d"):
    """
    Performs ordinary least squares (OLS) linear regression on the provided tickers
    over the inputted time period to analyze their behavior/trend.
    """
    if not tickers:
        print("No tickers provided for regression.")
        return

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"No historical data found for {ticker} over period {period}.")
                continue
                
            # Create an independent variable x representing the day number
            df['Day'] = np.arange(len(df))
            x = df['Day'].values
            y = df['Close'].values
            
            # Perform least squares regression using curve_fit (or polyfit)
            popt, pcov = curve_fit(linear_trend, x, y)
            slope, intercept = popt
            
            # Calculate predicted values
            y_pred = linear_trend(x, slope, intercept)
            
            # Determine behavior
            direction = "UPWARD" if slope > 0 else "DOWNWARD"
            print(f"\nTicker: {ticker} ({period} period)")
            print(f"  Regression Line: Price = {slope:.4f} * Day + {intercept:.2f}")
            print(f"  Trend Direction: {direction} (Slope: {slope:.4f})")
            
            # Plotting the regression line along with actual close prices
            plt.figure(figsize=(10, 4))
            plt.plot(df.index, y, label='Actual Close Price', color='blue', marker='o', markersize=3)
            plt.plot(df.index, y_pred, label=f'Least Squares Fit (Slope: {slope:.4f})', color='red', linestyle='--')
            plt.title(f"{ticker} Least Squares Linear Regression ({period})")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            #plt.show()
            
        except Exception as e:
            print(f"Error processing regression for {ticker}: {e}")

# Get the tickers identified by the indicator scanner
identified_tickers = minima_df['Ticker'].tolist() if not minima_df.empty else []

# Perform least squares regression over the past 30 days
analyze_trends_least_squares(identified_tickers, period="30d")

close_prices_dict = {}

for ticker in identified_tickers:
    stock = yf.Ticker(ticker)
    df = stock.history(period="30d")
    if not df.empty:
        close_prices_dict[ticker] = df['Close']

# Combine into a single DataFrame for easy view
if close_prices_dict:
    close_prices_df = pd.DataFrame(close_prices_dict)
    #display(close_prices_df)
else:
    print("No active tickers found to display close prices.")

### Now return a list of stock that are likely to go up a given percent in a given time period: 
def find_upward_candidates(tickers, target_pct=5.0, days_back=30):
    """
    Screens a list of tickers to find stocks likely to continue going up.
    Uses Least Squares Regression slope and momentum indicators (RSI + MACD).
    """
    candidates = []
    period = f"{days_back * 2}d"  # Fetch extra days to calculate indicators reliably
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if len(df) < days_back:
                continue
                
            # Trim to the target days_back window for regression
            analysis_df = df.iloc[-days_back:].copy()
            
            # 1. Least Squares Regression Slope
            x = np.arange(len(analysis_df))
            y = analysis_df['Close'].values
            slope, intercept = np.polyfit(x, y, 1)
            
            # Normalized slope (expressed as daily % change relative to the intercept price)
            normalized_slope_pct = (slope / intercept) * 100
            
            # 2. Technical Indicators (RSI & MACD)
            # Calculate 14-day RSI
            close_prices_all = df['Close'].values
            rsi_vals = calculate_rsi(close_prices_all, period=14)
            today_rsi = rsi_vals[-1]
            
            # Calculate MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            today_macd = df['MACD'].iloc[-1]
            today_signal = df['Signal'].iloc[-1]
            
            # Filter Criteria:
            # - Positive trend slope
            # - RSI between 40 and 70 (not overbought, but strong positive momentum)
            # - MACD is above the signal line (bullish crossover)
            if normalized_slope_pct > 0.1 and (40 <= today_rsi <= 70) and (today_macd > today_signal):
                # Calculate expected return over the period based on current slope projection
                projected_gain_pct = normalized_slope_pct * days_back
                
                if projected_gain_pct >= target_pct:
                    candidates.append({
                        'Ticker': ticker,
                        'Current Close': round(y[-1], 2),
                        'Daily Trend Slope (%)': round(normalized_slope_pct, 3),
                        'Projected Gain (%)': round(projected_gain_pct, 2),
                        'RSI (14)': round(today_rsi, 2)
                    })
                    
        except Exception as e:
            pass
            
    return pd.DataFrame(candidates)

# Example usage: find all stocsk in the S&P that will go up 5% in 60 days:

# using the function by 
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    response = requests.get(url, headers=headers)
    
    # ΔΙΟΡΘΩΣΗ: Μετατρέπουμε το text σε StringIO stream πριν το διαβάσει η pandas
    html_stream = StringIO(response.text)
    tables = pd.read_html(html_stream, flavor='lxml')
    df = tables[0]
    
    # Μετατροπή τελείας σε παύλα για συμβατότητα με το Yahoo Finance
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

SP = get_sp500_tickers()
candidates_df = find_upward_candidates(SP, target_pct=5.0, days_back=60)
display(candidates_df)