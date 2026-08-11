import yfinance as yf
import csv
import pandas
import numpy as np

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

def load_snp_500(p = None):
    price_df = pandas.read_csv('data/prices.csv', index_col=0)

    # Incomplete tickers are dropped when prices.csv is written, so any NaN here
    # means the file is stale. Fail loudly rather than let it reach the EM.
    n_nan = int(price_df.isnull().sum().sum())
    if n_nan:
        raise ValueError(
            f"data/prices.csv contains {n_nan} NaN; regenerate it with "
            "`python -m data.SNP500`"
        )

    data_matrix = price_df.to_numpy()
    NAMES = price_df.columns.tolist()

    if p is not None:
        data_matrix = data_matrix[:,:p]
        NAMES = NAMES[:p]
    
    return data_matrix, NAMES

if __name__ == "__main__":
    # Read in the tickers from the CSV file
    with open('data/constituents.csv', 'r') as f:
        tickers = [row[0] for row in csv.reader(f)]

    tickers.pop(0)

    # Get the historical data for each ticker and store it in a dictionary
    data = {}

    for ticker in tickers:
        d = get_historical_data(ticker,'2020-01-01', '2026-01-01')
        # A failed fetch (BRK.B, BF.B -- yfinance wants BRK-B, BF-B) returns an
        # EMPTY frame, whose null count is 0, so it would pass a bare isnull()
        # check and become an all-NaN column once from_dict aligns the dates.
        if len(d) == 0:
            print(f"No data returned for {ticker}, skipping.")
        elif d.isnull().sum().sum():
            print(f"Missing data for {ticker}, skipping.")
        else:
            data[ticker] = np.log10 (d['Close'] / d['Close'].shift(1))
            data[ticker].pop(d.first_valid_index()) #First row is left empty after shift

    price_df = pandas.DataFrame.from_dict(data)

    # from_dict unions the dates, so a ticker that listed part-way through the
    # window (ABNB, GEV, KVUE, ...) is NaN before its first trading day even
    # though its own frame had no gaps. Drop those here, so prices.csv is clean
    # on disk and load_snp_500 never has to deal with NaN.
    incomplete = price_df.columns[price_df.isnull().any()].tolist()
    if incomplete:
        print(f"Dropping {len(incomplete)} tickers that do not span the full "
              f"window ({', '.join(incomplete)})")
        price_df = price_df.drop(columns=incomplete)

    print(f"Saving {price_df.shape[0]} dates x {price_df.shape[1]} tickers")
    price_df.to_csv('data/prices.csv')
