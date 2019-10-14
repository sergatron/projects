# PROJECT CONTENTS

Summary of results are published on [Medium](https://medium.com/@smouzykin/solar-panel-array-what-does-it-cost-46aaa083502)

This project contains the following:

- [**Data wrangling**](https://github.com/sergatron/projects/blob/master/solar_array/Data_Wrangling.ipynb)
- [**NREL API script**](https://github.com/sergatron/projects/blob/master/solar_array/NREL_API.ipynb)
  - This script was used to obtain data for annual insolation in order to fill missing values
- [**Incentives API script**](https://github.com/sergatron/projects/blob/master/solar_array/incentives_API.ipynb)
  - This script was used to obtain a count of incentives available at each *county, state* combination
- [**Exploratory Data Analysis**](https://github.com/sergatron/projects/blob/master/solar_array/EDA.ipynb)
- [**Statistical Analysis**](https://github.com/sergatron/projects/blob/master/solar_array/statistical_analysis.ipynb)
- [**Machine Learning (Predicting the cost of a solar array)**](https://github.com/sergatron/projects/blob/master/solar_array/solar_array_final_models.ipynb)
  - Last iteration of models for the project

## Data 
- *openpv_all.csv*: original data obtained from [here](https://openpv.nrel.gov/)
- *pv_df_clean.csv*: written from Data_Wrangling.ipynb, intended for use in Exploratoty Data Analysis (EDA)
- *pv_df_clean_2.csv*: written from EDA.ipynb, intended for use in Statistical Analysis, and Machine Learning
