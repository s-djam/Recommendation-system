# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:12:44 2024

@author: sdjam
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
bts = pd.read_csv('bts_audio.csv')

# Columns of the DataFrame
columns = bts.columns.tolist()
print("Columns:")
print(columns)

bts.drop(['Unnamed: 0', 'playlistID', 'TrackID', 'SampleURL', 'Genres', 'Popularity'], axis=1)

# Missing values
bts.isnull().sum()

# Types
bts.dtypes
## Release year
bts['Year'] = bts['ReleaseYear'].str.split('-').str[0]
## To datetime
bts['ReleaseDate'] = pd.to_datetime(bts['ReleaseYear'])
bts['Year'] = bts['Year'].values.astype(np.int64)
bts = bts.drop(['ReleaseYear'], axis=1)
print(bts.dtypes)
bts = bts.sort_values(by='ReleaseDate')
bts.head()

# Export processed data
bts.to_csv('bts_processed.csv', index=False)

# Summary
bts.describe()

# Histograms
bts.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(20,20), color='orange', edgecolor='black')
plt.tight_layout()
plt.show()

# Correlations
## Select only numeric columns
bts_num = bts.select_dtypes(include=[np.number])
## Compute the correlation matrix
correlation_matrix = bts_num.corr()
## Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
