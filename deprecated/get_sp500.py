import argparse
from collections import defaultdict
import requests
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import json

def get_sp500_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the main S&P 500 company list table
    tables = soup.find_all("table", {"class": "wikitable"})
    
    # The first table is the current S&P 500 list
    df_sp500 = pd.read_html(str(tables[0]))[0]
    
    return df_sp500

def get_sp500_changes():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the table with S&P 500 changes
    tables = soup.find_all("table", {"class": "wikitable"})
    
    # The second table contains the historical changes
    df_changes = pd.read_html(str(tables[1]))[0]
    df_changes.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df_changes.columns]
    df_changes["Date_Date"] = pd.to_datetime(df_changes["Date_Date"].squeeze(), errors="coerce")
    df_changes['Year'] = pd.to_datetime(df_changes['Date_Date']).dt.year  # Extract year from Date_Date
    
    return df_changes

def get_sp500_companies_by_year(year):
    current_companies = get_sp500_companies()
    changes = get_sp500_changes()
    # Sort the DataFrame by Date_Date in descending order
    changes_sorted = changes.sort_values(by='Date_Date', ascending=False)

    # Get the current year companies as a starting point
    companies_set = set(current_companies["Symbol"])
    sp500_dict = {}
    extra_companies = []
    for i in range(datetime.datetime.now().year-1, year-1, -1):
        print(f"Processing for Year {i}")
        added_cnt = 0
        removed_cnt = 0
        changes_filtered = changes_sorted[changes_sorted["Date_Date"].dt.year == i]
        for _, row in changes_filtered.iterrows():
            added = row.get("Added_Ticker", "")
            removed = row.get("Removed_Ticker", "")
            if pd.notna(removed) and removed:
                removed = removed.strip()
                companies_set.add(removed)
                added_cnt += 1
                print(f"Added {removed}")
            if pd.notna(added) and added:
                added = added.strip()
                # Check if exist first
                if added in companies_set:
                    companies_set.discard(added)
                else:
                    extra_companies.append(added)
                print(f"Removed {added}")
                removed_cnt += 1
        print(f"Added {added_cnt}, Removed {removed_cnt}")
        print(f"Total Company {len(companies_set)}")
        sp500_dict[i] = sorted(companies_set)
    print(f"Extra companies not removed: {extra_companies}")
    return sp500_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--year", default="2001", type=int)
    args = parser.parse_args()
    
    year = args.year
    
    if year > datetime.datetime.now().year:
        print("Year cannot be in the future.")
    else:
        sp500_companies = get_sp500_companies_by_year(year)
        # Save to a JSON file
        with open("sp500_companies.json", "w") as json_file:
            json.dump(sp500_companies, json_file, indent=4)  # `indent=4` makes it human-readable
