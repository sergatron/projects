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
    import plotly.express as px
#    gapminder = px.data.gapminder()
    data = clean_data()
    fig = px.line(data,
                  x="year",
                  y="percent",
                  color="country",
#                  line_group="country_code"
                  )
    fig_ls = []
    fig_ls.append(fig.to_dict())

    return fig_ls

