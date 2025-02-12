import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os
import re
import statsmodels.api as sm

# Configuration
YOUR_EMAIL = "xuewen@u.nus.edu"  # Required for SEC Edgar

###########################
# Step 1: Get MD&A Section
###########################

def extract_mda(text):
    """Extract MD&A section from 10-K text (simplified example)"""
    item7_pattern = re.compile(
        r'ITEM\s*7\s*[\.\-]?\s*MANAGEMENT\'?S?\s*DISCUSSION\s*AND\s*ANALYSIS',
        re.IGNORECASE | re.DOTALL
    )
    item8_pattern = re.compile(
        r'ITEM\s*8\s*[\.\-]?\s*FINANCIAL STATEMENTS',
        re.IGNORECASE | re.DOTALL
    )
    
    start_match = item7_pattern.search(text)
    end_match = item8_pattern.search(text)
    
    if start_match and end_match:
        start = start_match.end()
        end = end_match.start()
        return text[start:end].strip()
    return None

#################################
# Step 2: Post-Event Volatility
#################################

def get_fama_french_factors():
    "Fama French Factors Daily from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    df = pd.read_csv('F-F_Research_Data_Factors_daily.csv', skiprows=3, index_col=0, parse_dates=True)
    df = df.iloc[:-1]  # Removes the last row
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    # Fama-French dataset reports percentage returns, convert back to decimal
    return df / 100

def calculate_volatility(ticker, filing_date, ff_factors):
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
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1)  # Ensure end date is included
    )
    if len(stock_data) < 10:  # Minimum data check
        return None
    
    # Get returns
    returns = stock_data['Adj Close'].pct_change().dropna()
    returns = returns.reset_index()
    returns['Date'] = returns['Date'].dt.date
    
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
    merged['Excess Return'] = merged['Adj Close'] - merged['RF']
    
    # Run regression
    X = merged[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    y = merged['Excess Return']
    
    model = sm.OLS(y, X, missing='drop').fit()
    return np.sqrt(np.mean(model.resid**2))

####################
# Main Processing
####################

def process_year(year, nasdaq_tickers):
    """Process a single year"""
    dl = Downloader("National University of Singapore", YOUR_EMAIL)
    ff_factors = get_fama_french_factors()
    results = []
    
    for ticker in nasdaq_tickers:
        try:
            # Download 10-K filings
            dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year}-12-31")
            
            # Process filings
            filings_dir = os.path.join("sec-edgar-filings", ticker, "10-K")
            for filing in os.listdir(filings_dir):
                filing_path = os.path.join(filings_dir, filing, "full-submission.txt")
                
                with open(filing_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    
                    # Extract filing date
                    filed_match = re.search(r'FILED AS OF DATE:\s*(\d+)', text)
                    if filed_match:
                        filing_date = datetime.strptime(filed_match.group(1), '%Y%m%d').date()
                    else:
                        continue
                    
                    # Extract MD&A
                    mda = extract_mda(text)
                    if not mda:
                        continue
                    
                    # Calculate volatility
                    volatility = calculate_volatility(ticker, filing_date, ff_factors)
                    if volatility:
                        results.append({
                            'Company': ticker,
                            'MD&A': mda,
                            'Volatility': volatility
                        })
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Save results
    if results:
        pd.DataFrame(results).to_csv(f"10k_volatility_{year}.csv", index=False)

if __name__ == "__main__":
    # Example usage - you need to provide actual NASDAQ tickers for each year
    nasdaq_tickers_2023 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']  # Replace with actual list
    
    # Process years (this will take significant time)
    for year in range(2001, 2025):
        print(f"Processing year {year}")
        process_year(year, nasdaq_tickers_2023)