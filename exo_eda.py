# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:12:46 2024

@author: sdjam
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
exo = pd.read_csv('exo_audio.csv')

# Columns of the DataFrame
columns = exo.columns.tolist()
print("Columns:\n", columns)

exo.drop(['Unnamed: 0', 'playlistID', 'TrackID', 'SampleURL', 'Genres', 'Popularity'], axis=1)
 
# Missing values
exo.isnull().sum()

# Types
exo.dtypes
## Release year
exo['Year'] = exo['ReleaseYear'].str.split('-').str[0]
## To datetime
exo['ReleaseDate'] = pd.to_datetime(exo['ReleaseYear'])
exo['Year'] = exo['Year'].values.astype(np.int64)
exo = exo.drop(['ReleaseYear'], axis=1)
print(exo.dtypes)
exo = exo.sort_values(by='ReleaseDate')
exo.head()

# Export processed data
exo.to_csv('exo_processed.csv', index=False)

# Summary
exo.describe()

# Histograms
exo.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(20,20), color='pink', edgecolor='black')
plt.tight_layout()
plt.show()

# Correlations
## Select only numeric columns
exo_num = exo.select_dtypes(include=[np.number])
## Compute the correlation matrix
correlation_matrix = exo_num.corr()
## Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()