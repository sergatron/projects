# Predicting the Cost of a Solar Array Installation

Summary of results are published on [Medium blog](https://medium.com/@smouzykin/solar-panel-array-what-does-it-cost-46aaa083502)

## Tabel of Contents:

- [**Data wrangling**](https://github.com/sergatron/projects/blob/master/solar_array/Data_Wrangling.ipynb)
- [**NREL API script**](https://github.com/sergatron/projects/blob/master/solar_array/NREL_API.ipynb)
- [**Incentives API script**](https://github.com/sergatron/projects/blob/master/solar_array/incentives_API.ipynb)
- [**Exploratory Data Analysis**](https://github.com/sergatron/projects/blob/master/solar_array/EDA.ipynb)
- [**Statistical Analysis**](https://github.com/sergatron/projects/blob/master/solar_array/statistical_analysis.ipynb)
- [**Machine Learning (Predicting the cost of a solar array)**](https://github.com/sergatron/projects/blob/master/solar_array/solar_array_final_models.ipynb)

## File Descriptions 
- **openpv_all.csv**: original data
- **pv_df_clean.csv**: written from Data_Wrangling.ipynb, intended for use in Exploratoty Data Analysis (EDA)
- **pv_df_clean_2.csv**: written from EDA.ipynb, intended for use in Statistical Analysis, and Machine Learning
- **Data_Wrangling.ipynb**: data wrangling/cleaning for the original data, writes new file **pv_df_clean.csv**
- **NREL_API.ipynb**: obtains additional data in order to fill missing values in variable *annual_insolation*. Writes new file **insolation_df.csv** which is then used in Data Wrangling to fill missing values.
- **incentives_API.ipynb**: This script was used to obtain a count of incentives available at each *county, state* combination. Writes new file **incentives_df.csv**.
- **EDA.ipynb**: exploratory data analysis
- **statistical_analysis.ipynb**: statistical analysis
- **solar_array_final_models.ipynb**: last iteration of machine learning models

## Acknowledgements
Original data obtained from [NREL](https://openpv.nrel.gov/)
  - NREL url: ([National Renewable Energy Lab](https://emp.lbl.gov/tracking-the-sun/))
  - NREL API has plenty of useful data https://developer.nrel.gov/ and easy to use.
  
