import yfinance as yf
import csv
import pandas
import numpy

"""
Get historical OHLCV data for the S&P 500 constituents and save it to a CSV file.
"""

def get_historical_data(symbol, start, end, interval='1d'):
    """
    Get historical OHLCV data.

    Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    return df

def load_snp_500():
    f = open('data/prices.csv','r')
    d = csv.reader(f)

    data_matrix = []

    for row in d:
        data_matrix += [row]
    
    return data_matrix

# Read in the tickers from the CSV file
f = open('data/constituents.csv','r')

constituents = csv.reader(f)
tickers = []
for row in constituents:
    tickers += [row[0]]

tickers.pop(0)

# Get the historical data for each ticker and store it in a dictionary
data = {}
for ticker in tickers:
    d = get_historical_data(ticker,'2025-01-01', '2026-01-01')
    if d.size == 1750:
        data[ticker] = numpy.log10 (d['Close'] / d['Open'])

# Save the data to a CSV file
pandas.DataFrame.from_dict(data).to_csv('data/prices.csv')