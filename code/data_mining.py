import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

def data_quality_report(df):
    print(f"Data Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nPercentage of Missing Values:\n", (df.isnull().sum() / len(df)) * 100)
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe())
    print("\nDuplicate Rows:\n", df.duplicated().sum())
    print("\nClass Distribution:\n", df['FIPS'].value_counts())

    # Visualizations
    df.hist(bins=30, figsize=(15, 10))
    plt.show()


with open('../data/gdppickle.sec', 'rb') as f:
    df_gdp = pd.read_pickle(f)

with open('../data/naics_patternpickle.sec', 'rb') as f:
    df_pattern = pd.read_pickle(f)

with open('../data/naics_occupationpickle.sec', 'rb') as f:
    df_occupation = pd.read_pickle(f)

# Interesting columns:
# 2017 - 2022, naics, emp_total_county_naics, emp_occupation, emp_nf, qp1_nf,
data_quality_report(df_gdp)
print("\n\n-\n\n")
data_quality_report(df_pattern)
print("\n\n-\n\n")
data_quality_report(df_occupation)
