import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os
import json
import statsmodels.api as sm
from risk_metrics import standard_deviation, sortino_ratio

def get_fama_french_factors():
    """Fama French Factors Daily from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"""
    df = pd.read_csv('F-F_Research_Data_Factors_daily.csv', skiprows=3, index_col=0, parse_dates=True)
    df = df.iloc[:-1]  # Removes the last row
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    # Fama-French dataset reports percentage returns, convert back to decimal
    return df / 100

def calculate_volatility(ticker, filing_date, ff_factors, risk_measurement='std'):
    """Calculate post-event volatility"""
    # Get trading calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Get valid trading days
    schedule = nyse.schedule(
        start_date=filing_date,
        end_date=filing_date + timedelta(days=400)
    )
    trading_days = schedule.index.date
    
    # Check if we have enough days
    if len(trading_days) < 252:
        return None
    
    start_date = trading_days[5]  # 6th trading day
    end_date = trading_days[251]  # 252nd trading day
    
    # Get stock data
    stock_data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date + timedelta(days=1)  # Ensure end date is included
    )
    if len(stock_data) < 10:  # Minimum data check
        return None
    
    # Get returns
    returns = stock_data['Close'].pct_change().dropna().rename(columns={ticker: 'Return'})
    returns = returns.reset_index()
    returns['Date'] = pd.to_datetime(returns['Date'])
    
    # Merge with Fama-French factors
    merged = pd.merge(
        returns,
        ff_factors,
        left_on='Date',
        right_index=True,
        how='inner'
    )
    
    if len(merged) < 10:
        return None
    
    # Calculate excess returns
    merged['Excess Return'] = merged['Return'] - merged['RF']
    
    # Run regression
    X = merged[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    y = merged['Excess Return']
    
    model = sm.OLS(y, X, missing='drop').fit()
    if risk_measurement=='std':
        risk_value = standard_deviation(model.resid)
    elif risk_measurement=='sortino':
        risk_value = sortino_ratio(model.resid)
    else:
        raise ValueError(f"Invalid risk_measurement: {risk_measurement}. Expected 'std' or 'sortino'.")
        
    return risk_value

####################
# Main Processing
####################

def process_year(year):
    """Process a single year"""
    ff_factors = get_fama_french_factors()
    results = []
    filing_folder = os.path.join('./datasets/EXTRACTED_FILINGS', str(year), '10-K')
    cik2ticker = json.load(open('./cik2ticker.json'))
    
    for filing in os.listdir(filing_folder):
        filing_path = os.path.join(filing_folder, filing)
        data = json.load(open(filing_path))
        cik = data['cik']
        ticker = cik2ticker[cik]
        company = data['company']
        mda = data['item_7']
        filing_date = pd.Timestamp(data['filing_date'])
        volatility = calculate_volatility(ticker, filing_date, ff_factors)
        if volatility:
            results.append({
                'CIK': cik,
                'Company': company,
                'MD&A': mda,
                'Volatility': volatility
            })
            
    # Save results
    if results:
        pd.DataFrame(results).to_csv(f"10k_volatility_{year}.csv", index=False)

if __name__ == "__main__":
    # Example usage - you need to provide actual NASDAQ tickers for each year
    # Process years (this will take significant time)
    for year in range(2024, 2020, -1):
        print(f"Processing year {year}")
        process_year(year)