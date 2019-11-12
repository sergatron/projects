# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:47:37 2019


GOALS:
    1. Scrape most recent info for stock/ETF
    2. Clean/check/convert data types
        2a. Price values must be formatted, and of type float
        2b. Volume values must be of type integer
        2c. Index must be set to date

@author: smouz
"""





#%% Scrape 
# =============================================================================
# -Earnings
# -Revisions
# -EPS
# -Growth


#%%
# =============================================================================
# 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
from dateutil import parser


if 'stock_analysis' not in os.getcwd():
    #os.listdir()
    os.chdir('Python Practice\data_projects\stock_analysis')
    
# %% 
# DATE INPUT AND TIME STAMP
# =============================================================================


# =============================================================================
# # NOTE:
#    Max rows retrieved is 100
#    Iterate the dates to acqurie more than 100 rows
# =============================================================================
    
    
# ENTER DATES
# =============================================================================
# 1. enter date for which to obtain stock market data, EX: 2019-02-21
#   a. DEFAULT: current date
# 2. enter amount of days for which to obtain data (max = 100), EX: 50

# current date
todays_date = str(datetime.today().date())
num_days    = 90
ticker      = 'spy'


#def calc_time_delta(date1, date2, format_date='%Y-%m-%d'):
#    """
#    Date input must be in string dtype
#    Returns change in days between the two dates
#    date1 - date2
#    """
#    try:
#        # convert to datetime object
#        end_day = datetime.strptime(date1, '%Y-%m-%d').date()
#        start_day = datetime.strptime(date2, '%Y-%m-%d').date()
#        # datetime math
#        change = end_day-start_day
#    except TypeError:
#        print('Dates must be strings!')
#    return change.days
#
#calc_time_delta(todays_date, '2018-12-01',)

# TIMEDELTA AND DATETIME CONVERSIONS
# current date as datetime object
todays_date_obj = datetime.today()

def timedelta_to_date(date_in, day_change=-50, date_time = '%Y/%m/%d'):
    """
    Input date as datetime object.
    Calculates new date by adding or subtracting days.
    Returns date in string format.
    """
    try:
        convert_this = datetime.date(np.add(date_in, timedelta(days=day_change)))
        day = pd.to_datetime(convert_this, format=date_time).date()
    except TypeError:
        print('Date must be a datetime object!')
    return np.str(day)
prior_date = timedelta_to_date(datetime.today(), day_change=-num_days)


# Function: convert date input to timestamp
def to_timestamp(Date, Format='%Y-%m-%d'):
    """
    Takes input date as string
    """
    return np.int(datetime.timestamp(datetime.strptime(Date, Format)))
#to_timestamp(np.str(todays_date_obj.date()))
#to_timestamp(prior_date)



# %%
# CONCAT URL STRING
# =============================================================================

# select or input ticker symbol
#ticker_array = np.char.array(['SPY', 'SOXX', 'GDX'])
#ticker = ticker_array[0]

#url1 = 'https://finance.yahoo.com/quote/'
#ticker = ticker
#url2 = '/history?period1='
#period1 = to_timestamp(prior_date)
#url3 = '&period2='
#period2 = to_timestamp(np.str(todays_date_obj.date()))
#url4 = '&interval=1d&filter=history&frequency=1d'
#
#full_fString = f'{url1}{ticker}{url2}{period1}{url3}{period2}{url4}'


# ALTERNATIVE PARAMETERS 
parameters = {'period1': to_timestamp(prior_date),
              'period2': to_timestamp(np.str(todays_date_obj.date())),
              'interval': '1d',
              'frequency': '1d',
              }

#%% MAKE REQUEST 
# =============================================================================
#page = requests.get(full_fString) # content or text; is one more efficient or faster?

page = requests.get(f'https://finance.yahoo.com/quote/{ticker}/history', params=parameters)
page.url

print('----------------------------------------------------------------------')

if page.status_code != 200:
    print('CONNECTION NOT ESTABLISHED!')
else:
    print('Connected!', page.headers['Date'])

page_soup = BeautifulSoup(page.content, 'html.parser')
print('----------------------------------------------------------------------')


#%% ARRAY OF DATES
# =============================================================================
# extract and create array of dates using numpy arrays

# find id for each date
date_id = page_soup.find_all("td",{"class":"Py(10px) Ta(start) Pend(10px)"})
#date_id
#len(date_id)
  

date_arr = np.array([date_id[i].text for i in range(len(date_id))], dtype=str)
len(date_arr)

# NOTE: 
#   Some dates may be recorded twice and if there was a SPLIT or DIVIDEND
# drop duplicate dates
date_arr = pd.Series(date_arr).drop_duplicates().values
    
#%% LIST OF PRICES
# =============================================================================
# find id of prices
# OPEN, HIGH, LOW, CLOSE, ADJ CLOSE, VOLUME
price_ids = page_soup.find_all("td",{"class":"Py(10px) Pstart(10px)"})


# DAY OPEN PRICE
# =============================================================================
#price_ids[0].text
#price_ids[6].text
#len(price_ids)

# NOTE: opening price is every 6th value
# try using numpay array with list comprehension
#price_open_list = price_ids[0::6]
#price_open = []
#for i in range(len(price_open_list)):
#    price_open.append(price_open_list[i].text)
#    
#price_open_arr = np.array([price_open_list[i].text for i in range(len(price_open_list))])
#price_open == price_open_arr 

# 
# =============================================================================
def retrieve_price(resultSet, start_n, N):
    ''' 
    resultSet:  element tag, found using '.find_all' method
    N:          select every Nth position in list
    start_n:    start from this element in list
        
    A new list of element tags is created to iterate over. Text/value 
    contained within element is appended to a list of values
    '''
    price_list = resultSet[start_n::N]
    price_result = np.array([price_list[i].text for i in range(len(price_list))])
    return price_result

retrieve_price(price_ids, 0, 6) # start_n=0, this is open price
 
# open price: data-reactid="53"
#price_open_id = page_soup.find('span',{'data-reactid':'53'})
#price_open_id.text


#%% DAILY PRICES
# =============================================================================
#price_ids[1].text
#price_ids[7].text

#price_high = []
#price_high_list = price_ids[1::6] # start at item 1, select every 6th

# NOTE: daily high price is every 6th value, starting from position 1
#price_high = retrieve_price(price_ids, 1, 6)

# starting position 0: OPEN
# starting position 1: HIGH
# starting position 2: LOW
# starting position 3: CLOSE
# starting position 4: ADJ CLOSE
# starting position 5: VOLUME

# Use function and iteration to obtain all daily prices
# =============================================================================
#prices_all = []
#for i in range(6):
#    prices_all.append(retrieve_price(price_ids, i, 6))
#prices_all[0]

# change style to numpy and list comprehension
all_prices = np.array([retrieve_price(price_ids, i, 6) for i in range(6)])

# assert <condition>,<error message>

for i in range(len(all_prices)):
    print('Checking size of arrays...')
    print(i, np.size(all_prices[i]))
    assert np.size(date_arr) == np.size(all_prices[i]), 'Arrays size not equal'

print('\nDate array:', len(date_arr))
print('Prices array:', len(all_prices[2]))



#%% CREATE PANDAS DF
# =============================================================================
# set option: format float in pandas
pd.options.display.float_format = '{:.2f}'.format


spy_df = pd.DataFrame.from_dict({'date': date_arr,
                                 'open': all_prices[0],
                                 'high': all_prices[1],
                                 'low': all_prices[2],
                                 'close': all_prices[3],
                                 'volume': all_prices[5]
                                 },
                                orient='index'
                                )
spy_df = spy_df.transpose()

#%% 
# CLEAN DF
# =============================================================================


# parse date strings into formatted date time object
#parser.parse(spy_df['date'][4]) #parameter must be in form of string
#parser.parse('2018 Jan 10') #parameter must be in form of string
#type(parser.parse('Feb 28 2003'))
#type(pd.to_datetime(parser.parse('Feb 28 2003')).date())

def date_to_str(series_str):
    """
    Converts date input to string with format '%Y/%m/%d'
    """
#    convert_date = datetime.date(parser.parse(series_str))
#    date_time = '%Y/%m/%d'
#    date_str = str(pd.to_datetime(convert_date, format=date_time).date())
#    current_date = str(datetime_now)
    return str(parser.parse(series_str).date())

date_to_str('Feb 28 - 2019')


dates = map(date_to_str, spy_df['date'])
spy_df['date'] = pd.Series(dates)

# %% 
# FIX INDEX
# =============================================================================
# set index to date
spy_df = spy_df.set_index('date')


# show columns which contain missing values
if np.any(spy_df.isnull()):
    # drop rows which contain missing values
    spy_df = spy_df.dropna(axis=0, how='any')

# assert that there are no missing values
assert ~np.any(spy_df.isnull())

# convert values to float and integer
for item in spy_df.columns[:4]:
    spy_df[item] = spy_df[item].astype(np.float32)
  
# ALTERNATIVE: convert values to float and integer
#open_ = pd.Series(map(float,spy_df['open']))

# remove comma in volume column
spy_df['volume'] = spy_df['volume'].str.replace(',', '').astype(np.int32)
## convert to integer
#spy_df['volume'] = spy_df['volume'].astype(int)

# create column for ticker
spy_df['ticker'] = ticker

#print(spy_df.head(), spy_df.info())


spy_df = spy_df.sort_index()
write_df = spy_df.copy()



# %% 
# IMPORT AND COMBINE DATAFRAMES
# =============================================================================
# CONCAT new DF with existing
# df1 = existing/previous stock data
# df2 = most recent stock data
# df3 = pd.concat([df2,df1]) # df2 is stacked on top of df1


#file_name1 = 'SPY_1292019.csv'
##file_name2 = 'SPY_historical_data.csv'
##file_name3 = 'Chart-20190123-185551.csv'
#spy_orig = pd.read_csv(file_name1,
#                  sep = ',',
#                  low_memory=False,
#                  index_col='Date'
#                  )
#spy_orig.info()
#
#df_new = pd.concat([spy_df, spy_orig])
#df_new.head(25)
#
## sort by index
#df_new = df_new.sort_index()
#
## drop duplicates
#df_new = df_new.drop_duplicates()
## drop duplicate indecies
#df3 = df_new[~df_new.index.duplicated(keep='first')]
#
#
#df3.tail(30)
#
#SPY = df3.copy()

#%% 
# WRITE FILE TO CSV
# =============================================================================
print('----------------------------------------------------------------------')
print('Writing dataframe to file...\n')
print(write_df.tail())

output_file = f'yahoo_{ticker}.csv'
print('\n')
print('Filename:', output_file)

write_df.to_csv(output_file,
              encoding='utf-8',
              index=True,
              index_label='date'
              )

print('\nFinished writing.')
print('----------------------------------------------------------------------')










