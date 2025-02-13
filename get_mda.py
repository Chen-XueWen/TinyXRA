from edgar_crawler import download_filings, extract_items 

years = range(2024, 2020, -1)
cik_tickers = ["AAPL", "MSFT"]
for year in years:
    config = {
        "download_filings": 
            {"start_year": year,	
             "end_year": year,
             "quarters": [1, 2, 3, 4],
             "filing_types": ["10-K"],	
             "cik_tickers": cik_tickers,	
             "user_agent": "National University of Singapore (xuewen@u.nus.edu)",	
             "raw_filings_folder": f"RAW_FILINGS/{year}",	
             "indices_folder": f"INDICES/{year}",	
             "filings_metadata_file": f"FILINGS_METADATA/{year}.csv",	
             "skip_present_indices": True},
        "extract_items": {	
             "raw_filings_folder": f"RAW_FILINGS/{year}",	
             "extracted_filings_folder": f"EXTRACTED_FILINGS/{year}",	
             "filings_metadata_file": f"FILINGS_METADATA/{year}.csv",	
             "filing_types": ["10-K"],	
             "include_signature": False,	
             "items_to_extract": ["7"],	
             "remove_tables": True,	
             "skip_extracted_filings": True}
        }
    download_filings.main(config)
    extract_items.main(config)