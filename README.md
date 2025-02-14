# FINBERT-XRC Enhanced

## Requirements (Tested on the Version below)
Please create it under the environment named finbertxrc.
```bash
conda create -n "finbertxrc" python=3.11.3
conda activate finbertxrc
```
Install the following dependencies and make adjustments based on your system:
### FinBERT-XRC Requirements
```bash
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.39.3
pip install wandb==0.16.6
pip install scipy==1.13.0
pip install numpy==1.24.4
```
### Datasets Generation Requirements
```bash
pip install requests==2.31.0
pip install beautifulsoup4==4.11.1
pip install pandas==1.5.3
pip install yfinance==0.2.52
pip install pandas-market-calendars==4.6.1
pip install statsmodels==0.14.4
pip install urllib3==1.26.11
pip install pathos==0.2.9
pip install cssutils==1.0.2
pip install click==7.1
```


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