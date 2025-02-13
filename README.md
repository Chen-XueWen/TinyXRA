# FINBERT-XRC Enhanced

1. Get the company tickers (in this case S&P 500)
```bash
python get_sp500.py
```
2. Get the MD&A for the s&p500 companies.
```bash
python get_mda.py
```
3. Calculate the post event volatility
```bash
python calculate_volatility.py
```

Reference for Edgar-Crawler:
https://github.com/lefterisloukas/edgar-crawler


The line of code np.sqrt(np.mean(model.resid**2)) calculates the standard deviation of the residuals from the regression of excess returns on the Fama-French factors. This standard deviation represents the idiosyncratic volatility of the stock's returns, which is the portion of volatility not explained by the market risk premium, size effect, and value effect as modeled by the Fama-French factors.

In other words, while the Fama-French model accounts for systematic risk factors, the residuals capture the stock-specific, unsystematic risk. Therefore, the computed standard deviation of these residuals reflects the volatility inherent to the stock itself, after controlling for the specified systematic factors.