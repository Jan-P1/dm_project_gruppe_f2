import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import sweetviz as sv


# pip install sweetviz
def data_analysis_sweetviz(df):
    # analyzing the dataset
    advert_report = sv.analyze(df)
    # display the report
    advert_report.show_html('Advertising.html')

# importing sweetviz
import sweetviz as sv

with open('../data/gdppickle.sec', 'rb') as f:
    df_gdp = pd.read_pickle(f)

with open('../data/naics_patternpickle.sec', 'rb') as f:
    df_pattern = pd.read_pickle(f)

with open('../data/naics_occupationpickle.sec', 'rb') as f:
    df_occupation = pd.read_pickle(f)

#analyzing the dataset
gdp_report = sv.analyze(df_gdp)
pattern_report = sv.analyze(df_pattern)
occupation_report = sv.analyze(df_occupation)

#display the report
gdp_report.show_html('GDP.html')
pattern_report.show_html('Pattern.html')
occupation_report.show_html('Occupation.html')


data_analysis_sweetviz(df_gdp)
print("\n\n-\n\n")
data_analysis_sweetviz(df_pattern)
print("\n\n-\n\n")
data_analysis_sweetviz(df_occupation)
