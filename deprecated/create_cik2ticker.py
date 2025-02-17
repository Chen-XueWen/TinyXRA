import requests
import json

# URL of the SEC's company_tickers.json file
url = 'https://www.sec.gov/files/company_tickers.json'

def fetch_company_tickers(url):
    headers = {
        'User-Agent': 'National University of Singapore (xuewen@u.nus.edu)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def create_cik_to_ticker_map(data):
    cik_to_ticker = {}
    for item in data.values():
        cik_str = str(item['cik_str'])
        ticker = item['ticker']
        cik_to_ticker[cik_str] = ticker
    return cik_to_ticker

def main():
    # Fetch the company tickers data
    data = fetch_company_tickers(url)
    
    # Create the CIK to Ticker mapping
    cik_to_ticker_map = create_cik_to_ticker_map(data)
    
    # Output the mapping as a JSON-formatted string
    with open("cik2ticker.json", "w") as json_file:
        json.dump(cik_to_ticker_map, json_file, indent=4)  # `indent=4` makes it human-readable

if __name__ == '__main__':
    main()
