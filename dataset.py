#To fetch stock prices for a list of stocks and save them to CSV files
import yfinance as yf

stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'ITC.NS', 'HDFCBANK.NS', 
          'BAJFINANCE.NS', 'ASIANPAINT.NS', 'WIPRO.NS', 'TATAMOTORS.NS', 'BHARTIARTL.NS']

data = {}
for stock in stocks:
    df = yf.download(stock, start='2014-07-01', end='2024-07-01')
    data[stock] = df[['Close']]
    # Save to CSV
    df.to_csv(f"{stock}_data.csv")
