# Explainable AI for Comprehensive Risk Assessment for Financial Reports: A Lightweight Hierarchical Transformer Network Approach

This repository contains the dataset, code, and evaluation scripts for:

> Explainable AI for Comprehensive Risk Assessment for Financial Reports: A Lightweight Hierarchical Transformer Network Approach
> Xue Wen Tan, Stanley Kok

📄 Paper: [Read it on arXiv](https://arxiv.org/abs/2506.23767)

## Requirements (Tested on the Version below)
Please create it under the environment named tinyxra.
```bash
conda create -n "tinyxra" python=3.11.3
conda activate tinyxra
```
Install the following dependencies and make adjustments based on your system:
### Tinyxra Requirements
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

### NOTE: We have provided the processed datasets under the folder labelled as json. Feel free to skip to "To run model"
Please download the datasets at the following link: 

https://drive.google.com/file/d/1lr2tEqk9nsArW5Scg53uRsQk-x-YRjgS/view?usp=sharing

and place it directly at "tinyxra_code/datasets"

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
