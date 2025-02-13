# FINBERT-XRC Enhanced

1. Get the company tickers (in this case S&P 500)
```bash
python get_sp500.py
```
2. Get the 10k filings for the s&p500 companies. Make sure to wait for all the download to be completed first, before running the organize script.
```bash
python get_10k.py
```
3. Organize the 10k filings in years before we process them
```bash
python organize_10k.py
```


Extra:
https://github.com/lefterisloukas/edgar-crawler