import json
import os
from sec_edgar_downloader import Downloader

# Load the JSON file
with open('./sp500_companies.json', 'r') as f:
    data = json.load(f)

# Extract all unique company symbols
unique_companies = set()
for year, companies in data.items():
    unique_companies.update(companies)

# Convert to a sorted list for better readability
unique_companies = sorted(unique_companies)

print(f"Total unique companies: {len(unique_companies)}")

# Configuration
YOUR_EMAIL = "xuewen@u.nus.edu"  # Required for SEC Edgar

# Initialize the SEC EDGAR downloader
DOWNLOAD_DIR = "sec_filings"
dl = Downloader("National University of Singapore", YOUR_EMAIL)  # Replace with your details

# Download 10-K filings for each unique company
for ticker in unique_companies[:10]:  # Limit to first 10 for testing
    print(f"Downloading 10-K filings for {ticker}...")
    try:
        dl.get("10-K", ticker)
    except Exception as e:
        print(f"Error fetching 10-K for {ticker}: {e}")

print("Download complete. Check the 'sec_filings' folder.")
