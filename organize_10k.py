import os
import shutil
import re

# Define the base directory where 10-K filings are stored
BASE_DIR = "./sec-edgar-filings"
DEST_DIR = "./sec-edgar-yearly"

# Regex to extract the year from the filename
year_pattern = re.compile(r"-([0-9]{2})-[0-9]{5}")

# Iterate over all companies (tickers) in the BASE_DIR
for ticker in os.listdir(BASE_DIR):
    ticker_path = os.path.join(BASE_DIR, ticker, "10-K")
    
    if not os.path.isdir(ticker_path):
        continue  # Skip non-directory files
    
    # Iterate through all 10-K filing folders
    for filing in os.listdir(ticker_path):
        filing_path = os.path.join(ticker_path, filing)

        if not os.path.isdir(filing_path):
            continue  # Skip files, only process directories
        
        # Extract the year from the folder name
        match = year_pattern.search(filing)
        if not match:
            print(f"Skipping {filing} - No valid year found")
            continue
        
        # Skip everything before 2000
        if int(match.group(1)) > 50:
            continue 
        year = '20' + match.group(1)  # Convert '02' to '2002'
        year_folder = os.path.join(DEST_DIR, f"{year}_10K")
        os.makedirs(year_folder, exist_ok=True)  # Create year folder if not exists

        # Source and destination file paths
        source_file = os.path.join(filing_path, "full-submission.txt")
        destination_file = os.path.join(year_folder, f"{ticker}.txt")

        # Move and rename the file
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"Moved {source_file} → {destination_file}")
        else:
            print(f"File missing: {source_file}")

print("Organizing complete!")
