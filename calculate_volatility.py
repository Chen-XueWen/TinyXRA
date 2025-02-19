import pandas as pd
import numpy as np
import yfinance as yf
from nltk.tokenize import sent_tokenize
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os
import json
import statsmodels.api as sm
from tqdm import tqdm 
from sec_cik_mapper import StockMapper
from risk_metrics import calculate_standard_deviation, calculate_skewness, calculate_kurtosis, sortino_ratio

def get_fama_french_factors():
    """Fama French Factors Daily from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"""
    df = pd.read_csv('./datasets/F-F_Research_Data_Factors_daily.csv', skiprows=3, index_col=0, parse_dates=True)
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
    
    #start_date = trading_days[6]  
    start_date = trading_days[0] # 0th trading day, but if we follow tsai then it is 6th
    end_date = trading_days[251]  # 252nd trading day
    
    for tick in ticker:
        stock_data = yf.download(
            tickers=tick,
            start=start_date,
            end=end_date + timedelta(days=1),
            progress=False) 
        if stock_data.empty != False:
            break # If there is stock data then take the one with stock data

    if len(stock_data) < 10:  # Minimum data check
        return None
    
    # Get returns
    returns = stock_data['Close'].pct_change().dropna().rename(columns={tick: 'Return'})
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
    std_risk_value = calculate_standard_deviation(model.resid)
    skewness_risk_value = calculate_skewness(model.resid)
    kurtosis_risk_value = calculate_kurtosis(model.resid)
    sortino_risk_value = sortino_ratio(model.resid)

    return (std_risk_value, skewness_risk_value, kurtosis_risk_value, sortino_risk_value)

####################
# Main Processing
####################

def process_year(year):
    """Process a single year"""
    ff_factors = get_fama_french_factors()
    results = []
    filing_folder = os.path.join('./datasets/EXTRACTED_FILINGS', str(year), '10-K')
    mapper = StockMapper() # for cik to ticker map
    cik2ticker = mapper.cik_to_tickers
    
    for filing in tqdm(os.listdir(filing_folder), total=len(os.listdir(filing_folder)), desc=f"Processing {year} filings"):
        filing_path = os.path.join(filing_folder, filing)
        data = json.load(open(filing_path))
        cik = data['cik']
        company = data['company']
        mda = data['item_7']
        try:
            ticker = list(cik2ticker[cik.zfill(10)])
        except:
            print(f"cik2ticker for company {company} not found")
            continue

        mda_clean = mda.replace("\n", " ").replace("’", "'")
        if len(sent_tokenize(mda_clean)) < 3:
            print(f"Incomplete extracted MD&A for {company}")
            continue
        filing_date = pd.Timestamp(data['filing_date'])
        volatility = calculate_volatility(ticker, filing_date, ff_factors)
        if volatility:
            results.append({
                'CIK': cik,
                'Company': company,
                'MD&A': mda_clean,
                'Std_value': volatility[0],
                'Skewness_value': volatility[1],
                'Kurtosis_value': volatility[2],
                'Sortino_value': volatility[3] * 1e15 # Daily sortino value is too small, so multiply by 1e15 so that to_json will not round to 0.
            })
    
    # Save results
    if results:
        pd.DataFrame(results).to_json(f"./datasets/10k_volatility/{year}.json", 
           orient="records", 
           indent=4,
           double_precision=15,
           force_ascii=False)

if __name__ == "__main__":
    # Example usage - you need to provide actual NASDAQ tickers for each year
    # Process years (this will take significant time)
    if not os.path.isdir('datasets/10k_volatility/'):
        os.mkdir('datasets/10k_volatility/')
    for year in range(2012, 2002, -1):
        print(f"Processing year {year}")
        process_year(year)