from sec_edgar_downloader import Downloader

# Initialize a downloader instance. Download filings to the current
# working directory. Must declare company name and email address
# to form a user-agent string that complies with the SEC Edgar's
# programmatic downloading fair access policy.
# More info: https://www.sec.gov/os/webmaster-faq#code-support
# Company name and email are used to form a user-agent of the form:
# User-Agent: <Company Name> <Email Address>
dl = Downloader("National University of Singapore", "xuewen@u.nus.edu")
# Get all 10-K filings for Microsoft
ticker = "AAPL"

years = range(2001, 2024, 1)

for year in years:
    dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year}-12-31")
    print("Hello World")

print('Hello World')
