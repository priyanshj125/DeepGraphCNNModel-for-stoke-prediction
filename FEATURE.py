import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Get your raw OHLCV data
data = yf.Ticker("MSFT").history(period="1y")

# 2. Automatically calculate and append features!
# We just tell it what features we want.
data.ta.rsi(length=14, append=True)
data.ta.macd(append=True)
data.ta.bbands(length=20, append=True)

# 3. Your DataFrame now has new feature columns
# e.g., 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0'
print(data.tail())
