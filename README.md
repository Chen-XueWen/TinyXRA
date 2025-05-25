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
pip install scikit-learn==1.6.1
pip install seaborn==0.13.2
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
pip install sec-cik-mapper==2.1.0
pip install nltk==3.9.1
pip install h5py==3.12.1
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

# To get datasets
```bash
python get_mda.py # to get all the 10k raw fillings and extract the md&a

python calculate_volatility.py # To calculate post event volatility, better to do it 2 seperate times or else yfinance will block you

python label_data.py # to create the 3 quartiles for the risks

python max_count.py # to get the 80th percentile word and sentence max value
```

# To run model
```bash
cd {chosen model}
python prepro.py --test_year {chosen year}
python main.py --test_year {chosen year}
```

Model can only fit 350 sentences and 40 words which is approximately the 80th percentile (396 and 44 respectively at year 2024)

*MD&A Crawler from Edgar-Crawler:
https://github.com/lefterisloukas/edgar-crawler

# For my reference

The line of code np.sqrt(np.mean(model.resid**2)) calculates the standard deviation of the residuals from the regression of excess returns on the Fama-French factors. This standard deviation represents the idiosyncratic volatility of the stock's returns, which is the portion of volatility not explained by the market risk premium, size effect, and value effect as modeled by the Fama-French factors.

In other words, while the Fama-French model accounts for systematic risk factors, the residuals capture the stock-specific, unsystematic risk. Therefore, the computed standard deviation of these residuals reflects the volatility inherent to the stock itself, after controlling for the specified systematic factors.


Skewness > 0 (Right/Positive Skewed) → More weight in the right tail (higher probability of large positive returns).
Skewness < 0 (Left/Negative Skewed) → More weight in the left tail (higher probability of large negative returns).

✔ Investors prefer positive skewness because it implies occasional big gains rather than large unexpected losses.

1. Skewness Interpretation
Definition: Skewness measures the asymmetry of the return distribution.

Skewness = 0 → The distribution is perfectly symmetrical (like a normal distribution).
Skewness > 0 (Right/Positive Skewed) → More weight in the right tail (higher probability of large positive returns).
Skewness < 0 (Left/Negative Skewed) → More weight in the left tail (higher probability of large negative returns).
Financial Interpretation
Positive Skewness (Right-skewed)

More frequent small losses but occasional large gains.
Example: Tech stocks, cryptocurrencies, or high-growth assets.
Investors may prefer this because of the potential for high upside returns.
Negative Skewness (Left-skewed)

More frequent small gains but occasional large losses.
Example: Options selling, certain hedge fund strategies, and bonds.
Risky because of large, unexpected drops.
✔ Investors prefer positive skewness because it implies occasional big gains rather than large unexpected losses.

2. Kurtosis Interpretation
Definition: Kurtosis measures the "tailedness" of a distribution—how frequently extreme returns occur.

Kurtosis = 3 (Normal Distribution)

Standard bell-curve shape (moderate tails).
Excess Kurtosis > 0 (Leptokurtic)

Fat tails → More extreme events (both gains and losses).
Example: Stock markets, cryptocurrency markets, financial crises.
Indicates higher risk because of large, unexpected moves.
Excess Kurtosis < 0 (Platykurtic)

Thin tails → Fewer extreme events.
Example: Government bonds or stable blue-chip stocks.
Less risk but also fewer opportunities for extreme gains.
Financial Interpretation
High Kurtosis (> 3)

Market returns exhibit large outlier moves (boom and bust cycles).
Example: 2008 financial crisis, Black Monday (1987).
Investors need to hedge against tail risks.
Low Kurtosis (< 3)

Less volatile, fewer outlier events.
Example: Stable dividend stocks.
Predictable returns but lower potential for extreme gains.
✔ Investors should be cautious of high kurtosis, as it suggests markets are prone to extreme events (crashes and booms).

Practical Takeaway
Metric	Value	Interpretation
Skewness	> 0	Occasional large gains (desirable)
Skewness	< 0	Occasional large losses (risky)
Kurtosis	> 3	Extreme market moves (high risk)
Kurtosis	< 3	More stable, fewer extremes
📌 Ideal scenario for investors?

Mildly positive skewness (potential for high gains).
Moderate kurtosis (~3) (balanced risk).
Avoid highly negative skewness & excessive kurtosis.
