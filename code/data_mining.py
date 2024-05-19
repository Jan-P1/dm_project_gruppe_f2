import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

with open('../data/gdppickle.sec', 'rb') as f:
    df_gdp = pd.read_pickle(f)

with open('../data/naics_patternpickle.sec', 'rb') as f:
    df_pattern = pd.read_pickle(f)

with open('../data/naics_occupationpickle.sec', 'rb') as f:
    df_occupation = pd.read_pickle(f)

print(df_gdp.head())
print(df_pattern.head())
print(df_occupation.head())
