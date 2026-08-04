import yfinance as yf
import csv
import pandas
import numpy
f = open('constituents.csv','r')
d = csv.reader(f)
tickers = []
for row in d:
    tickers += [row[0]]

tickers.pop(0)


def get_historical_data(symbol, start, end, interval='1d'):
    """
    Get historical OHLCV data.

    Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    return df


data = {}



for ticker in tickers:
    d = get_historical_data(ticker,'2025-01-01', '2026-01-01')
    if d.size == 1750:
        data[ticker] = numpy.log10 (d['Close'] / d['Open'])


pandas.DataFrame.from_dict(data).to_csv('prices.csv')

# df = get_historical_data('AAPL', '2025-01-01', '2026-01-01')
# print(df.tail())