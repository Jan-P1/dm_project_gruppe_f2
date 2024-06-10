import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import sweetviz as sv
from sklearn.preprocessing import StandardScaler

# Load the datasets
with open('../data/gdppickle.sec', 'rb') as f:
    df_gdp = pd.read_pickle(f)

with open('../data/naics_patternpickle.sec', 'rb') as f:
    df_pattern = pd.read_pickle(f)

with open('../data/naics_occupationpickle.sec', 'rb') as f:
    df_occupation = pd.read_pickle(f)



# Fill missing values in GDP data
# Create a consolidated 'gdp' column starting with 2022 data
df_gdp['gdp'] = df_gdp['2022']

# List of years in descending order
years = ['2022', '2021', '2020', '2019', '2018', '2017']

# Fill missing values in 'gdp' column with values from previous years
for year in years[1:]:
    df_gdp['gdp'].fillna(df_gdp[year], inplace=True)

# Drop individual year columns
df_gdp.drop(columns=years, inplace=True)

# Ensure FIPS codes are county-level (5 digits)
def ensure_county_fips(fips):
    fips = str(fips)
    if len(fips) > 5:
        return fips[:5]  # Crop to the first 5 digits
    elif len(fips) < 5:
        return None  # Mark for removal
    return fips

# Apply the function to ensure FIPS codes are correct
df_gdp['FIPS'] = df_gdp['FIPS'].apply(ensure_county_fips)
df_pattern['FIPS'] = df_pattern['FIPS'].apply(ensure_county_fips)
df_occupation['FIPS'] = df_occupation['FIPS'].apply(ensure_county_fips)

# Filter out rows where FIPS codes are not 5 digits (not county-level)
df_gdp = df_gdp[df_gdp['FIPS'].notnull()]
df_pattern = df_pattern[df_pattern['FIPS'].notnull()]
df_occupation = df_occupation[df_occupation['FIPS'].notnull()]

# Select necessary features from each dataset
df_gdp = df_gdp[['FIPS', 'gdp']]
df_pattern = df_pattern[['FIPS', 'naics', 'emp', 'qp1', 'ap', 'est']]
df_occupation = df_occupation[['FIPS', 'naics', 'emp_total_county_naics', 'emp_occupation']]

# Merge GDP and Pattern datasets on 'FIPS' and 'naics'
df_merged = pd.merge(df_pattern, df_gdp, on='FIPS', how='inner')

# Merge the above result with Occupation dataset on 'FIPS' and 'naics'
df_merged = pd.merge(df_merged, df_occupation, on=['FIPS', 'naics'], how='inner')

# Normalize numerical features (e.g., emp, qp1, ap, GDP values)
# Selecting columns to normalize
columns_to_normalize = ['emp', 'qp1', 'ap', 'est', 'emp_total_county_naics', 'emp_occupation', 'gdp']

scaler = StandardScaler()
df_merged[columns_to_normalize] = scaler.fit_transform(df_merged[columns_to_normalize])

# Selecting the most relevant features for clustering
features = ['FIPS', 'naics', 'emp', 'qp1', 'ap', 'est', 'emp_total_county_naics', 'emp_occupation', 'gdp']

df_final = df_merged[features]

# Display final dataframe info
print(df_final.info())
print(df_final.head())

# Analyzing the datasets
df_final_report = sv.analyze(df_final)

# Display the reports
df_final_report.show_html('Final.html')

# Save the final dataframe for clustering analysis
df_final.to_csv('final_dataset_for_clustering.csv', index=False)
