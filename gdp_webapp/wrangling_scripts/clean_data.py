# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:05:42 2019

@author: smouz

"Military Expenditure and GDP Web App"

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

pd.options.display.max_columns = 30
pd.options.display.max_rows = 50
pd.options.display.width = 100


def clean_data():
    # LOAD DATA
    df = pd.read_csv('data/Military Expenditure.csv')
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

    # extract countries with higher GDP
    df_2 = df[df['amount'] > df['amount'].mean()]
    top_countries = df_2['country'].unique().tolist()

    clean_df = countries_df[countries_df['country'].isin(top_countries)]
    clean_df.reset_index(drop=True, inplace=True)

    return clean_df

def filter_data(clean_data):
    # FILTER COUNTRIES BY TOP 10
    # groupby Country and compute mean
    agg_gdp = clean_data.groupby(['country'])['military_expend','gdp','percent'].agg(['mean'])

    # TOP 10: Military Expenditure to GDP Ratio
    mil_10 = agg_gdp.sort_values(('military_expend', 'mean'), ascending=False)[:10].index.tolist()
    top_military_df = clean_data[clean_data['country'].isin(mil_10)]


    # TOP 10: Highest GDP
    gdp_10 = agg_gdp.sort_values(('gdp', 'mean'), ascending=False)[:10].index.tolist()
    top_gdp_df = clean_data[clean_data['country'].isin(gdp_10)]


    # TOP 10: Highest Military Expenditure
    percent_10 = agg_gdp.sort_values(('percent', 'mean'), ascending=False)[:10].index.tolist()
    top_ratio_df = clean_data[clean_data['country'].isin(percent_10)]

    return top_military_df, top_gdp_df, top_ratio_df

def return_figures():

    df = clean_data()

    # define list of countries and initialize empty list
    country_ls = df['country'].unique().tolist()
    graph_one = []

    # iterate over countries to define y-values
    for country in country_ls:
        # define x and y
        # x=years, y=GDP
        x_val = df[df['country'] == country].year.tolist()
        y_val =  df[df['country'] == country].gdp.tolist()
        mark_size = df[df['country'] == country].percent*10

        graph_one.append(
                go.Scatter(
                        x=x_val,
                        y=y_val,
                        mode='markers',
                        marker=dict(size=mark_size),
                        name=country
                        )
                )
    # define layout; title, x-axis, y-axis
    layout_one = dict(title='Change in GDP between 1960 and 2019',
                      xaxis=dict(title='Year',
                                 autotick=True,
                                 tick0=1960,
                                 dtick=5
                                 ),
                      yaxis=dict(title='GDP (current USD)')
                      )
    figures = [
            dict(data=graph_one, layout=layout_one),

            ]



    # PLOTLY EXPRESS
    # Plot 1: Military Expense
    data = filter_data(clean_data())[0]
    fig_one = px.line(data,
                      x="year",
                      y="military_expend",
                      title='Military Expenditure 1960 - 2019',
                      color="country",
                      line_group="country_code",
                      labels={'military_expend': 'Military Expenditure',
                              'year': 'Year'
                              }
                      )
    # Plot 2: GDP of nations
    data_1 = filter_data(clean_data())[1]
    fig_two = px.line(data_1,
                      x="year",
                      y="gdp",
                      title='GDP between 1960 and 2019',
                      color="country",
                      line_group="country_code",
                      labels={'gdp': 'Gross Domestic Product (GDP)',
                              'year': 'Year'
                              }
                      )

    # Plot 3: ratio of Military Expense to GDP
    data_2 = filter_data(clean_data())[2]
    fig_three = px.line(data_2,
                        x="year",
                        y="percent",
                        title='Military Expenditure to GDP ratio 1960 - 2019',
                        color="country",
                        line_group="country_code",
                        labels={'percent': 'Military Expenditure to GDP Ratio',
                              'year': 'Year'
                              }
                        )
    # Plot 4: Scatter ratio of Military Expense to GDP
    data_2_scat = filter_data(clean_data())[2]
    # create new column for size of marker
#    data_2_scat['marker_size'] = data_2_scat['percent']
    fig_four = px.scatter(data_2_scat,
                          x="year",
                          y="gdp",
                          log_y=True,
                          title='Military Expenditure to GDP ratio 1960 - 2019',
                          color="country",
                          hover_data=['country', 'percent', 'gdp'],
                          size='percent',
                          size_max=75,
                          opacity=0.3,
                          labels={'percent': 'Military Expenditure to GDP Ratio',
                                  'year': 'Year'
                                  }
                          )


    # append all data into list
    fig_ls = []
    for item in [fig_one.to_dict(), fig_two.to_dict(),
                 fig_three.to_dict(), fig_four.to_dict()]:
        fig_ls.append(item)


    return fig_ls

