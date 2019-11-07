
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

pd.options.display.max_columns = 30
pd.options.display.max_rows = 50
pd.options.display.width = 100



# LOAD DATA
df = pd.read_csv('data/Military Expenditure.csv')
codes_df = pd.read_csv('data/country-continent.csv')
gdp_df = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_422026.csv',
                     skiprows=4)









#                               GDP: MELT
gdp_years = gdp_df.columns[gdp_df.columns.str.isnumeric()].tolist()
gdp_ids = gdp_df.iloc[:, :4].columns

gdp_df = pd.melt(gdp_df, id_vars=gdp_ids, value_vars=gdp_years,
                 var_name='year', value_name='gdp')
gdp_df['year'] = gdp_df['year'].astype('datetime64').dt.year

# 'amount' -> float
gdp_df['gdp'] = gdp_df['gdp'].astype('float')

# GDP: RENAME
gdp_df = gdp_df.rename(columns={'Country Name': 'country',
                                'Country Code': 'country_code'})

# GDP: LOWERCASE
lower_cols = ['country', 'country_code']
gdp_df[lower_cols] = gdp_df[lower_cols].applymap(str.lower)

# GDP: DROP COLS
gdp_df.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)



#                   MILITARY EXPENDITURES
# MELT
# transform data such that all years are in one column

# extract all 'years' from columns, and ids to unpivot on
all_years = df.columns[df.columns.str.isnumeric()].tolist()
ids = df.iloc[:, :4].columns

# unpivot using melt()
df = pd.melt(df, id_vars=ids, value_vars=all_years,
             var_name='year', value_name='amount')

# CONVERT DATA TYPES
# 'year' -> datetime, year
df['year'] = df['year'].astype('datetime64').dt.year

# 'amount' -> float
df['amount'] = df['amount'].astype('float')


# RENAME COLUMNS
df = df.rename(columns={'Name': 'country', 'Indicator Name': 'indicator'})

# contents of columns to lwoercase
cols_to_lower = ['country', 'Code', 'Type', 'indicator']
df[cols_to_lower] = df[cols_to_lower].applymap(str.lower)
df.columns = df.columns.str.lower()

# MISSING VALUES
# drop rows with missing values
df = df.dropna(subset=['amount'], axis=0).reset_index(drop=True)
df.drop(['indicator'], axis=1, inplace=True)

# EXTRACT COUNTRIES
# exract only countries
df = df[df['type'] == 'country']



#                               CONTINENTS
# drop columns
codes_df = codes_df.drop(['Two_Letter_Country_Code', 'Country_Number'],
                         axis=1)
# CONTINENTS: LOWERCASE
cols_to_lower = ['country', 'Code', 'Type', 'indicator']

# lowercase contents of all columns, and columns
codes_df = codes_df[codes_df.columns].applymap(str).applymap(str.lower)
codes_df.columns = codes_df.columns.str.lower()

# CONTINENTS: MISSING VALUES
# replace 'nan' string with north america, 'na'
codes_df['continent_code'] = codes_df['continent_code'].str.replace('nan', 'na')







#                              MERGE DATAFRAMES
# bring together into one DF
countries_df = pd.merge(df, gdp_df,
                        left_on=['country', 'year'],
                        right_on=['country', 'year'])

countries_df = countries_df.dropna(subset=['gdp'], axis=0)
countries_df = countries_df.drop(['code', 'type'], axis=1).reset_index(drop=True)

countries_df['country'].nunique()
countries_df = countries_df.rename(columns={'amount': 'military_expend'})

# MILITARY VS GDP
# compute percent of GDP
countries_df['percent'] = countries_df['military_expend'] / countries_df['gdp'] * 100

countries_df.sort_values('percent', ascending=False)[:20]

countries_df.sort_values('gdp', ascending=False)[:20]


#
# =============================================================================
# PLOT WORLD MILITARY EXPENDITURE
df[df['country'] == 'world'].plot(kind='scatter', x='year', y='amount')
df[df['country'] == 'south africa'].plot(kind='scatter', x='year', y='amount')


df['amount'].describe()
df['amount'].plot(kind='hist', bins=50)

df_2 = df[df['amount'] > df['amount'].mean()]

df_2['country'].unique()
df_2['country'].nunique()

df_2[df_2['country'] == 'germany'].plot(kind='scatter', x='year', y='amount')
df_2[df_2['country'] == 'united states'].plot(kind='scatter', x='year', y='amount')

df_2['amount'].plot(kind='hist', bins=50)
#df_2[['country', 'year', 'amount']].plot(kind='scatter', x='year', y='amount')


# =============================================================================




df_2 = df[df['amount'] > df['amount'].mean()]
df_2['country'].unique()
top_countries = df_2['country'].unique().tolist()

clean_df = countries_df[countries_df['country'].isin(top_countries)]
clean_df.reset_index(drop=True, inplace=True)

clean_df['country'].nunique()

group_data = clean_df.groupby(['year', 'country']).agg(['mean'])
group_data.loc[(1980, 'australia'):(2000, 'australia'), :]['percent']

# MEAN GDP
agg_gdp = clean_df.groupby(['country'])['military_expend','gdp','percent'].agg(['mean'])

agg_gdp.sort_values(('gdp', 'mean'), ascending=False)[:10]
agg_gdp.sort_values(('percent', 'mean'), ascending=False)[:10]
agg_gdp.sort_values(('military_expend', 'mean'), ascending=False)[:10]

# take top 10 countries from each group and make plot for each

