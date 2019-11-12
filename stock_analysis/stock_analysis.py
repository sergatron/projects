# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:35:40 2019


GOALS:
    1. Overbough/Oversold Signal
    2. EMA crossover signal
    3. Test strategies, create portfolio
    4. Fibonacci


@author: smouz
"""



# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

import pandas as pd
import numpy as np
import ta
import quandl
import matplotlib.pyplot as plt
import os 
from pandas_datareader import data
from datetime import datetime, timedelta
from dateutil import parser

#%% GET SYMBOLS
#nd = data.get_nasdaq_symbols()
#len(nd)

#type(data.get_summary_iex())
#type(data.get_markets_iex())

#data.get_markets_iex().info()
#data.get_summary_iex().info()
#%% SET WORKING DIRECTORY
## get current working directory
#os.getcwd()
## change directory
#os.chdir('Python Practice\stock_analysis')


if 'stock_analysis' not in os.getcwd():
    os.chdir('Python Practice\data_projects\stock_analysis')
    
    
# format float in pandas
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_columns = 6
pd.options.display.max_rows = 100


# %% IMPORTING DATA FROM QUANDL
# =============================================================================

#quandl.ApiConfig.api_key = 'zCwpZpWLjA4nspkmsDDy'
## Data Time index data
#stock = quandl.get('EOD/EEM',
#                   start_date = '2008-01-01',
#                   end_date = '2019-01-23',
#                   )
#stock.info()
#stock.index.max()
#stock.shape

#fred = quandl.get('FRED/GDP', start_date='2000-04-01', end_date='2019-02-01')
#fred.info()

#%% IMPORTING DATA USING PANDAS
# =============================================================================

#stock_df = data.DataReader('spy', start='1990', end='2019', data_source='yahoo')
#stock_df.head()
#stock_df.info()

#from pandas_datareader import data as pdr
#df = pdr.get_data_yahoo('GDX','2018-08-01', '2019-03-22')
#df.tail()

    

#%% UPLOADING DATA FROM CSV
# =============================================================================


ticker_array = np.char.array(['SPY', 'SOXX', 'GDX'])
ticker = ticker_array[0]


# SP 500 ETF (SPY)
#file_name = 'yahoo_SPY.csv'
file_name = f'yahoo_{ticker}.csv'
scraped_df = pd.read_csv(file_name,
                  sep = ',',
                  low_memory=False,
#                  index_col='date'
                  )

scraped_df = scraped_df.set_index('date')

# first record
scraped_df.index.min()

# last record (most recent)
scraped_df.index.max()


#%% IMPORT AND COMBINE DATAFRAMES
# =============================================================================
scraped_df.info()
scraped_df.tail(15)
scraped_df.head(15)

"""
If imported data is SPY then preprocess data
Else analyze recent data which was scraped
"""

if scraped_df['ticker'][0] == 'SPY':
    file_name2 = 'SPY_2000.csv'
    #file_name3 = 'Chart-20190123-185551.csv'
    spy_orig = pd.read_csv(file_name2,
                      sep = ',',
                      low_memory=False,
                      index_col='Date'
                      )
    
    spy_orig.info()
    spy_orig.tail(5)
    spy_orig.head(5)
    
    # convert columns names to lower case
    def map_lowercase(str_object):
        """
        Checks for data type and returns data type in lower case.
        """
        if isinstance(str_object, pd.DataFrame):
            return list(map(str.lower, str_object.columns.values))
        if isinstance(str_object, str):
            return str_object.lower()
        else:
            return list(map(str.lower, str_object))
    spy_orig.columns = map_lowercase(spy_orig)
    
    
    spy_orig.tail()
    spy_orig['adj_close'] = spy_orig['adj close']
    
    spy = spy_orig[['open', 'high', 'low', 'close', 'volume']]

    
    # concat
    df_new = pd.concat([scraped_df, spy], sort=True)
    df_new['ticker'] = df_new['ticker'].fillna(ticker)
    
    # sort by index
    df_new = df_new.sort_index()
    
    # drop duplicates
    df_new = df_new.drop_duplicates()
    # drop duplicate indecies
    df3 = df_new[~df_new.index.duplicated(keep='first')]
    
    #tick = df3['ticker'].values[len(df3)-1]
    #df3.loc[:, 'ticker'] = df3.loc[:, 'ticker'].fillna(tick)
    
    # check for any NaNs
    assert ~np.any(df3.isna())
    
    df = df3.copy()
    print('Working on:', df['ticker'][0])

else:
    df = scraped_df.copy()
    print('\nWorking on:', scraped_df['ticker'][0])

# %%=============================================================================

#quandl.ApiConfig.api_key = 'zCwpZpWLjA4nspkmsDDy'
#
## Load data in form of table
#df = quandl.get_table('WIKI/PRICES',
##                        qopts = { 'columns': ['ticker', 'date', 'close'] },
#                        ticker = ['MSFT'],
#                        date = { 'gte': '2008-01-01', 'lte': '2019-01-20' }
#                        )
#    
#df.info()
#df.columns
#        
## select columns
#df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'ex-dividend']]
#df['date'].max()

#%% STOCHASTICS
# =============================================================================
#spy = spy.loc['2009-01-01':'2019-01-18', :]
#df = spy.copy()


# def stoch_signal(high, low, close, n=14, d_n=3, fillna=False)
df['stoch'] = ta.stoch_signal(high = df['high'],
                              low = df['low'],
                              close = df['close'],
                              fillna=True
                              )
df['stoch'].describe()

# %% MACD
# =============================================================================
df['macd'] = ta.macd(df['close'])
df['macd_diff'] = ta.macd_diff(df['close']) # relationship between MACD and MACD Signa
df['macd_signal'] = ta.macd_signal(df['close']) # EMA of MACD

df['macd'].describe()

#%% EASE OF MOVEMENT
# =============================================================================
df['ease_movement'] = ta.ease_of_movement(high = df['high'],
                                          low = df['low'],
                                          close = df['close'],
                                          volume = df['volume']
                                          )


#%% CHAIKIN MOMEY FLOW n=20
# =============================================================================
df['chaikin'] = ta.chaikin_money_flow(high = df['high'],
                                      low = df['low'],
                                      close = df['close'],
                                      volume = df['volume']
                                      )

df['chaikin'].describe()
#%% MOVING AVERAGE
# =============================================================================
df['ema_20'] = ta.ema_indicator(df['close'], n=20)
df['ema_50'] = ta.ema_indicator(df['close'], n=50)
df['ema_100'] = ta.ema_indicator(df['close'], n=100)
df['ema_200'] = ta.ema_indicator(df['close'], n=200)


#%% RSI
# =============================================================================
df['rsi'] = ta.rsi(df['close'])

df['rsi'].describe()


#%% DATETIME CONVERSION
# =============================================================================
def to_datetime_obj(date_string, date_format='%Y-%m-%d'):
    """Convert date from string format to datetime object
    datetime.strptime(date_string, date_format).date()"""
    return datetime.strptime(date_string, date_format).date()
to_datetime_obj('2014-12-07')

date_obj = datetime.strptime('2014-10-07', '%Y-%m-%d').date()
#datetime.strptime('2014-10-07', '%Y-%m-%d').day
#datetime.strptime('2014-10-07', '%Y-%m-%d').month
#datetime.strptime('2014-10-07', '%Y-%m-%d').year


def date_to_str(year, month, day):
    """Convert date from datetime object to string
    str(datetime(year,month,day).date())"""
    return str(datetime(year,month,day).date())
date_to_str(2014,11,25)

#%% PERFORMANCE
# =============================================================================
# performance = start_date price - end_date price / start_date price * 100
def performance(df, start, end):
#    start = to_date_str(start)
    '''Calculate percent change within given date range based on closing price.'''
    try:
        p = (df.loc[end, 'close'] - df.loc[start, 'close']) / df.loc[start, 'close'] * 100
        print('Start price: ' + str(df.loc[start, 'close']))
        print('End price: ' + str(df.loc[end, 'close']))
        return float('{:.2f}'.format(p))
    except KeyError:
        print('Value does not exist')

performance(df, '2019-01-04', '2019-02-19')

#year,month,day = 2014,10,25 
#d = to_date_str(year,month,day)

#%% RECENT LOW
# =============================================================================
# detecting a low close price within a date range
def find_low(df, dates, price_type='close'):
    '''
    Finds a low within given date range.
    dates:          input string as a list. Ex.: [start, end]
    price_type:     open, high, low, or closing price
    '''
    # min price in date range
    min_price = np.min(df.loc[dates[0]:dates[1], price_type])
    # date for the min_price occurance
    date_low = df[np.equal(df[price_type],min_price)][price_type].index.values
    return df.loc[date_low, price_type]

find_low(df, ['2018-11-01', '2018-12-31'], 'close')
find_low(df, ['2018-01-01', '2018-05-31'], 'low')

    
# min price in date range
min_price = np.min(df.loc['2018-11-01':'2018-12-31', 'close'])

# date for the min_price occurance
#date = df[np.equal(df['close'],min_price)]['close'].index.values
#df.loc[date, 'low']


#%% OVERSOLD
# =============================================================================
# stoch < 20
# rsi < 20


# 2018 stats
# =============================================================================
#date_time = '%Y/%m/%d'
#datetime_now = pd.to_datetime(datetime.today(), format=date_time).date()
#current_date = str(datetime_now)
#
#
#df_sub = df.loc['2018':'2019', :]
#df_sub[np.less(df_sub['stoch'],20)]
#
#np.all(df_sub.loc['2018-12-24', ['stoch', 'rsi']] < 20)
#
#indx = np.max(np.where(df_sub.loc[:, 'stoch'] < 20)[0])
#df_sub.iloc[indx, 0]


# EXPLORE:
#   How long (average time) do these overbought conditions last?

#%% OVERBOUGHT

# GENERATE SIGNAL
# =============================================================================
date_time = '%Y/%m/%d'
datetime_now = pd.to_datetime(datetime.today(), format=date_time).date()
current_date = str(datetime_now)

def display_func(Statement, String):
    
    print('-------------------------------------')
    print(Statement)
    print(String)
    print('-------------------------------------')
#display_func(df.close, 'Displaying tail!')   
    
# OVERBOUGHT SIGNAL     
# =============================================================================
# 1. return stats for overbought conditions
# 2. return boolean
    
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
timedelta_to_date(datetime.today(), day_change=-len(df))
    
# =============================================================================
# GENERATE OVERBOUGHT SIGNAL

# start at past date
start_date = timedelta_to_date(datetime.today(), day_change=-len(df))
# end at current date
end_date = np.str(datetime.today().date())

overbought_df = df.loc[start_date: end_date,  ['ticker', 'close', 'rsi', 'stoch']]
# initialize new column to 0.0
overbought_df['ovr_bought_moment'] = 0
# create condition for the signal
overbought_df['ovr_bought_moment'] = np.where(overbought_df[['rsi', 'stoch']]>80, 1, 0)
# create overbought signal at first instance
overbought_df['ovr_bought'] = overbought_df['ovr_bought_moment'].diff(2)

print('\n')
print('Overbought conditions')
print('-'*50)
print(overbought_df[overbought_df['ovr_bought'] > 0])
#print('Last date of occurance:', overbought_df[overbought_df['ovr_bought'] > 0].index[-1])
print('-'*50)
print('\n')


#overbought_df[overbought_df['ovr_bought_moment'] > 0].index
#overbought_df[10:50]
#
## sum days which are overbought, over 80
#np.sum(overbought_df['positions'] > 0)
## number of days within normal range
#np.sum(overbought_df['positions']==0)
#
#len(overbought_df)
#
#
#overbought_df['stoch_diff'] = overbought_df['rsi'].diff(2)
#overbought_df['stoch_diff'].plot(kind='hist', bins=34, normed=True)
# =============================================================================


# NOTE: error occurs when date entered is not contained within range specified   

# if all(df_sub.loc['DATE', ['stoch', 'rsi']] < 20):
#       df_sub.loc['DATE', ['stoch', 'rsi', 'open', 'close']]
#       print('Conditions are oversold')



#%% 
# OVERSOLD SIGNAL
# =============================================================================
# 1. return stats for oversold conditions
# 2. return boolean



# =============================================================================
# GENERATE OVERSOLD SIGNAL

# start at past date
start_date = timedelta_to_date(datetime.today(), day_change=-len(df))
# end at current date
end_date = np.str(datetime.today().date())


oversold_df = df.loc[start_date: end_date, ['ticker', 'close', 'rsi', 'stoch']]
oversold_df['ovr_sold_moment'] = 0
# create condition for the signal where both indicators are less than 20
#oversold_df['moment_signal'] = np.where(((oversold_df['stoch']<20) & (oversold_df['rsi']<20)), 1,0)
oversold_df['ovr_sold_moment'] = np.where(oversold_df[['stoch']] < 20, 1,0)

# GENERATE TRADE ORDER
oversold_df['ovr_sold'] = oversold_df['ovr_sold_moment'].diff(2)
oversold_df[oversold_df['ovr_sold'] > 0]

print('\n')
print('Oversold conditions')
print('-'*50)
print(oversold_df[oversold_df['ovr_sold'] > 0])
print('-'*50)
print('\n')

## number of days below 20
#np.sum(oversold_df['positions'] > 0)
#
## number of days within normal range
#np.sum(oversold_df['positions'] == 0)



#%%
# FUNCTION: SIGNAL AND TRADE ORDER
# =============================================================================

def create_df_sub(start_date, end_date, *features):
    """
    Create a subset dataframe with given dates and features.
    """
    df_sub = df.loc[start_date: end_date, features]
    return df_sub

df_temp = create_df_sub('2014-01-04', str(datetime_now), 'ticker', 'close', 'rsi', 'stoch')

def create_series_lt(data_frame, threshold, comp=True, *features):
    """
    Returns series with features less than threshold
    """
    data_frame['signal'] = 0
    if comp:
        assert threshold < 25
        data_frame['signal'] = np.where((data_frame[np.char.array(features)] < threshold), 1, 0)
        return data_frame['signal'].diff(2)
    else:
        assert threshold > 75
        data_frame['signal'] = np.where((data_frame[np.char.array(features)] > threshold), 1, 0)
        return data_frame['signal'].diff(2)


sig_diff = create_series_lt(df_temp, 10, True, 'stoch')
sig_diff[sig_diff > 0]
np.sum(sig_diff[sig_diff > 0])

df_temp['oversold_sig'] = sig_diff
df_temp[df_temp['oversold_sig']>0]


    


# =============================================================================
#%% EMA CROSSOVER SIGNAL
# =============================================================================

# STEP 1
# =============================================================================
#   When condition evaluates to True
#   Return date of initial crossover
#   Display results 

all_features = np.char.array(['ticker','close', 'rsi', 'stoch', 'ema_20', 'ema_50'])
compare_feats = np.char.array(['ema_20', 'ema_50'])
start = '2000-01-04'
end = '2019-02-21'

# create subset with EMA
ema_df = df.loc[start: end, all_features]
ema_df['ema_signal'] = 0
# create signal 1 or 0
# when condition is True, the 0.0 will be overwritten to 1.0
ema_df['ema_signal'][20:] = np.where(ema_df[compare_feats[0]][20:] > ema_df[compare_feats[1]][20:], 1, 0)

# generate trading order using diff()
# DIFF(): calculates current row - previous row
ema_df['positions'] = ema_df['ema_signal'].diff(2)

# number of days with ema_20 above ema_50
len(ema_df[ema_df['ema_signal'] > 0]) 
# number of days with ema_20 below ema_50
len(ema_df[ema_df['ema_signal'] == 0]) 

# count NaNs
np.sum(ema_df.isnull())
np.any(ema_df.isnull())

# show columns which contain missing values
if np.any(ema_df.isnull()):
    # drop rows which contain missing values
    ema_df = ema_df.dropna(axis=0, how='any')


# buy signal
buy_mask = np.greater(ema_df['positions'].values, 0)
ema_df[buy_mask]
# sell signal
sell_mask = np.less(ema_df['positions'], 0)
ema_df[sell_mask]

# performance between first buy signal and first sell signal



# STEP 2: Test
# =============================================================================
#   Add ticker to portfolio when ema_20 is above ema_50
#   Hold until condition evaluates to False
#   Sell ticker, calculate performance



# 
#%% 
# FIBONACCI ARRAY
# =============================================================================
# Xn = Xn-1 + Xn-2
fib = np.arange(22)
fib[0] = 0
fib[1] = 1
for i in range(2,22):
    fib[i] = fib[i-1] + fib[i-2]

# RATIOS
# =============================================================================   
# divide any number in the sequence by the next number; the ratio is always approximately 0.618
#   EX: Xn/Xn+1 = 0.618
    
fib_ratios = np.ones(22)
for i in range(np.size(fib_ratios)-1):
    fib_ratios[i] = np.float32(fib[i] / fib[i+1])

ratio1 = [np.float(fib[i] / fib[i+1]) for i in range(21)]
ratio2 = [np.float(fib[i] / fib[i+2]) for i in range(20)]
ratio3 = [np.float(fib[i] / fib[i+3]) for i in range(19)]

np.size(fib_ratios)
# =============================================================================
# FIB SUPPORTS
# =============================================================================

# start at past date
start_date = timedelta_to_date(datetime.today(), day_change=-len(df))
# end at current date
end_date = np.str(datetime.today().date())

#pd.Series.resample()


#df.loc[start_date: end_date, ['close']].resample('1M').agg('mean').dropna(axis=0, how='all')
df_3m_sub = df.loc[start_date: end_date, ['open', 'high', 'low', 'close']]

def calc_fib_support(df_n, dates, func=np.median, price='close'):
    df = df_n.loc[dates[0]: dates[1], ['open', 'high', 'low', 'close']]
    
    fib = np.arange(22)
    fib[0] = 0
    fib[1] = 1
    for i in range(2,22):
        fib[i] = fib[i-1] + fib[i-2]
        
    fib_ratios = np.ones(22)
    for i in range(np.size(fib_ratios)-1):
        fib_ratios[i] = np.float32(fib[i] / fib[i+1])
    
    ratio1 = [np.float(fib[i] / fib[i+1]) for i in range(21)]
    ratio2 = [np.float(fib[i] / fib[i+2]) for i in range(20)]
    ratio3 = [np.float(fib[i] / fib[i+3]) for i in range(19)]
    
    # using high and low closing price
    # HIGH - (HIGH - LOW) * RATIO
    min_val = np.min(df[price].values)
    max_val = np.max(df[price].values)
    support3 = max_val-(max_val- min_val)*func(ratio1)
    support2 = max_val-(max_val- min_val)*func(ratio2)
    support1 = max_val-(max_val- min_val)*func(ratio3)
    
    print('-------------------------------')
    print('Current max:', max_val)
    print('Current min:', min_val)
    print('Supports based on daily', price, 'price')
    print('-------------------------------')
    print('S1:', '{:.2f}'.format(support1))
    print('S2:', '{:.2f}'.format(support2))
    print('S3:', '{:.2f}'.format(support3))
    print('-------------------------------')
    return support1, support2, support3
 

support1, support2, support3 = calc_fib_support(df, dates=[start_date, end_date], price='close')



# =============================================================================

# %% JANUARY PERFORMANCE
# =============================================================================

#df.loc['2018-01-01':'2018-01-31', 'close'].values
#performance(df, start='2018-01-02', end='2018-01-31')
#performance(df, start='2017-01-03', end='2017-01-31')
#performance(df, start='2016-01-04', end='2016-01-29')


# %% FUNCTION: TIME SERIES PLOT
# =============================================================================
# create functino for a quick plot of some feature for date range

#def plot_it(start_date, end_date, feature, size=(10,6)):
#    label_location = df.loc[start_date:end_date, feature].index[::21]
#    label_date = pd.to_datetime(label_location).strftime('%m-%Y')
#    
#    plt.figure(figsize=size)
#    plt.plot(df.loc[start_date:end_date, feature[0]], color='black')
#    if np.size(feature) > 1:
#        for item in feature:
#            plt.plot(df.loc[start_date:end_date, item])
#    plt.xlabel('Time')
#    plt.xticks(label_location, label_date, rotation=60)
#    plt.ylabel('Close Price')
#    plt.title(str(feature).upper())
#    plt.show()
#    
#
#plot_it('2018-08',
#        str(datetime.now().date()),
#        feature=['close', 'ema_20']
#        )
#plot_it('2018-12',
#        str(datetime.now().date()),
#        feature=['rsi', 'stoch']
#        )

# quick plot function
#plot_it('2018-01', '2019-02', ['rsi'], size=(12,2))
#plot_it('2018-01', '2019-02', ['stoch'], size=(12,2))



#%% RSI SLOPE
# =============================================================================
# use np.polyfit

#start_date = '2018-02-01'
#end_date = '2019-02-20'
#plot_feat = 'rsi'
##df['rsi'].rolling(14).mean()
#
## Ax + B
## Ax^2 + Bx + C
##y_p = df.loc[start_date: end_date, 'rsi']
##x_p = np.arange(len(y_p))
##a, b, c, d, *e = np.polyfit(x=x_p, y=y_p, deg=4)
##y_lin = a*(x_p**4) + b*(x_p**3) + c*x_p**2 + d*x_p**1 + e[0]
##
#
#rsi_lab_loc = df.loc[start_date:end_date, plot_feat].index[::21]
#rsi_lab_month = pd.to_datetime(rsi_lab_loc).strftime('%m-%Y')
#
#plt.figure(figsize=(12,2))
#plt.plot(df.loc[start_date:end_date, plot_feat])
#plt.plot(df.loc[start_date:end_date, plot_feat].rolling(7).mean())
##plt.plot(x_p, y_lin, color='red')
#plt.xlabel('Time')
#plt.xticks(rsi_lab_loc, rsi_lab_month, rotation=60)
#plt.ylabel(str(plot_feat))
#plt.title(str(plot_feat).upper())
#plt.hlines(y=80,
#           xmin = np.min(df.loc[start_date:end_date, :].index.values),
#           xmax = np.max(df.loc[start_date:end_date, :].index.values),
#           alpha=0.6
#           )
#plt.hlines(y=20,
#           xmin = np.min(df.loc[start_date:end_date, :].index.values),
#           xmax = np.max(df.loc[start_date:end_date, :].index.values),
#           alpha=0.6
#           )
#plt.show()


# %%
# EMA SLOPE
# =============================================================================
# slope = deltaY / deltaX
# Y = change in price / change in time
# Y = ema_20
# X = date


start_date = '2000-01-01'
end_date = '2019-02-18'

df_subset1 = df.loc[start_date:end_date, ['close','ema_20']]
date_max = np.max(df_subset1.index.values)
date_min = np.min(df_subset1.index.values)

# create slope of ema_20
df_subset1['ema_slope'] = df_subset1['ema_20'].diff(2)
df_subset1['ema_slope'].describe()

df_subset1[df_subset1['ema_slope'] <= -1.5]

df_subset1[df_subset1['ema_slope'] >= .80]


#%% SLOPE HISTOGRAM
# =============================================================================
#x_max = np.max(df_subset1['ema_slope'])
#x_min = np.min(df_subset1['ema_slope'])
#n_bins = np.int(np.sqrt(len(df_subset1)))
#plt.figure(figsize=(12,6))
#plt.hist(x=df_subset1['ema_slope'].values,
#         bins=n_bins,
#         normed=True,
#         range=(x_min, x_max),
##         cumulative=True
#         )
#plt.margins(0.02)
#plt.show()


#%% =============================================================================

#start_date = np.min(df_subset1.index.values)
#end_date = np.max(df_subset1.index.values)
#
## math with dates
##days_val = np.str(parser.parse(date_max) - parser.parse(date_min))
##delta_days = datetime.date(end_day) - datetime.date(start_day)
#
## convert to datetime object
#end_day = datetime.strptime(end_date, '%Y-%m-%d').date()
#start_day = datetime.strptime(start_date, '%Y-%m-%d').date()
#
#def calc_time_delta(date1, date2, format_date='%Y-%m-%d'):
#    """
#    Calculates change in days.
#    Returns integer
#    date1 - date2
#    """
#    # convert to datetime object
#    end_day = datetime.strptime(date1, '%Y-%m-%d').date()
#    start_day = datetime.strptime(date2, '%Y-%m-%d').date()
#    # datetime math
#    change = end_day-start_day
#    return change.days
#
#calc_time_delta(str(datetime_now), '2010-01-01',)
#
## datetime math
#change = end_day-start_day
#change.days
#
## PRICE CHANGE PER DAY
#delta_y = (df_subset1.loc[end_date, 'ema_20'] - df_subset1.loc[start_date, 'ema_20']) / change.days
#delta_y 



#%% MACD crossover
# =============================================================================




#%% CHAIKIN MF SLOPE
# =============================================================================



# %% PORTFOLIO DATAFRAME
# =============================================================================






#df = spy.copy()

#%% PLOTS
# =============================================================================

start_date = '2018-01'
end_date = str(datetime_now.year) + '-' + str(datetime_now.month)


label_loc = df.loc[start_date: end_date, 'close'].index[::21]
label_month = pd.to_datetime(label_loc).strftime('%m-%Y')

def tolerance(s):
    return s*.0005
tolerance(support1)

#plt.subplots(1,1)
plt.figure(figsize=(12,8))
#plt.plot(df.loc['2018-01-01':'2019-01-18', 'close'], marker = '.', linestyle = 'none', alpha = 0.6, color = 'red')
plt.plot(df.loc[start_date:end_date, 'close'])
plt.axhspan(ymin=support3-tolerance(support3), ymax=support3+tolerance(support3), alpha=0.5, color='#bd9545')
plt.axhspan(ymin=support2-tolerance(support2), ymax=support2+tolerance(support2), alpha=0.5, color='#bd5555')
plt.axhspan(ymin=support1-tolerance(support1), ymax=support1+tolerance(support1), alpha=0.5, color='#bd1545')
#plt.yticks([support1, support2, support3])  
#plt.text()
#plt.eventplot()
plt.xlabel('Time')
plt.xticks(label_loc, label_month, rotation=60)
plt.ylabel('Closing Price')
plt.title(np.str(df['ticker'][0]))
plt.show()

# =============================================================================
#plt.subplots(2,2)
plt.figure(figsize=(12,1))
plt.plot(df.loc[start_date:end_date, 'stoch'])
plt.xlabel('Time')
plt.xticks(label_loc, label_month, rotation=60)
plt.ylabel('Stochastic')
plt.title(np.str(df['ticker'][0]))
plt.hlines(y=80,
           xmin = np.min(df.loc[start_date: end_date, :].index.values),
           xmax = np.max(df.loc[start_date: end_date, :].index.values),
           alpha=0.6
           )
plt.hlines(y=20,
           xmin = np.min(df.loc[start_date: end_date, :].index.values),
           xmax = np.max(df.loc[start_date: end_date, :].index.values),
           alpha=0.6
           )
plt.show()

#
#
##%% =============================================================================
#date_time = '%Y/%m/%d'
#spy_dt = pd.to_datetime(spy['close'], format=date_time)
#spy_resample = spy_dt.resample('6M').agg('mean').dropna(axis=0, how='all')

#dates_ = df['2014':'2015'].index[::21]
#labels_ = dates_.strftime('%b-%Y')
#plt.xticks(dates_re, labels_re, rotation=60)


#%% =============================================================================


# NOTES
# explore price movements once indicators reach max or min.
# EX.: 
# if stoch > 80:
#       performance = start_date price - end_date price / start_date price * 100
#       plot closing_price during descending stoch/RSI
#       return performance


# CALCULATE SLOPE OF EMA200
# what happens when the slope becomes zero or negative? How long?

#%%
# =============================================================================
# 





# =============================================================================





