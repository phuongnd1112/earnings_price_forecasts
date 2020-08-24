import yfinance as yf 
import pandas as pd 

ticker = 'MSFT'

msft = yf.Ticker(ticker) 

df = msft.history(period= 'max') 
df2 = msft.quarterly_earnings

print(df) 
print(df2) 
