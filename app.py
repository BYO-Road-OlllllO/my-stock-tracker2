# Save as api.py
# Run with: uvicorn api:app --reload

from fastapi import FastAPI
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date

app = FastAPI()

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@app.get("/api/stock/{ticker}")
def get_stock_data(ticker: str, years: int = 1):
    # 1. Fetch historical data
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # 2. Prep data for Prophet
    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    # 3. Generate Projection
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=years * 365)
    forecast = m.predict(future)
    
    # 4. Format data to send to the Android app
    historical = df_train.tail(30).to_dict(orient="records") # Last 30 days
    projection = forecast[['ds', 'yhat']].tail(years * 365).to_dict(orient="records")
    
    return {
        "ticker": ticker,
        "historical_data": historical,
        "projected_data": projection
    }
