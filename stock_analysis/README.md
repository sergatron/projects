# Scraping Stock Quotes and Simple Analysis
Data can be obtained using Pandas' **pandas_datareader** module or scraped from the web. The web-scraping script is simply to practice some techniques of web-scraping, learn the Beautiful Soup module, and Python programming. Additionally, this project involves preparing data for analysis. After scraping, the dates needed to be parsed and remaining columns need to be converted to numerical data type.

# File Description
 - **stock_analysis.py**: imports data from a CSV file, calculates some simple metrics for the SPY ETF (tracks SP500 index) ticker such as RSI, Stochastics, and Moving Averages (20 day, 50 day).
 
 - **stock_scraper.py**: extracts specific data from given `url`, creates a data frame, accomplishes cleaning/preparation, and then writes to a CSV file.
