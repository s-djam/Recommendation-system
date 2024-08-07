# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:43:33 2024

@author: sdjam
"""

# Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Datasets

data_bts = pd.read_csv('bts_processed.csv')
data_exo = pd.read_csv('exo_processed.csv')

data_bts.set_index('TrackName', inplace=True)
data_exo.set_index('TrackName', inplace=True)

data_bts.drop(columns=['ReleaseDate'], inplace=True)
data_exo.drop(columns=['ReleaseDate'], inplace=True)

# First method : Cosine similarities

## Normalization of the data
scaler = StandardScaler()
combined_data = pd.concat([data_bts, data_exo])
combined_data_normalized = scaler.fit_transform(combined_data)
data_bts_normalized = combined_data_normalized[:len(data_bts)]
data_exo_normalized = combined_data_normalized[len(data_bts):]

## Similarity
similarity_bts_exo = cosine_similarity(data_bts_normalized, data_exo_normalized)
similarity_exo_bts = cosine_similarity(data_exo_normalized, data_bts_normalized)

## Recommendations

def get_recommendations(song_title, num_recommendations):

    if song_title in data_bts.index:
        # get exo recommendations for a given bts song
        song_index = data_bts.index.get_loc(song_title)
        similarity_matrix = similarity_bts_exo[song_index]
        data_artist_B = data_exo
        ## bts song's info
        song_features = data_bts.loc[[song_title]]
    elif song_title in data_exo.index:
        # get bts recommendations for a given exo song
        song_index = data_exo.index.get_loc(song_title)
        similarity_matrix = similarity_exo_bts[song_index]
        data_artist_B = data_bts
        ## exo song's info
        song_features = data_exo.loc[[song_title]]
    else:
        raise ValueError("couldn't find the song in bts or exo dataset")
    # song's info
    print(f"'{song_title}' info :")
    print(song_features)

    # Get indexes of the most similar songs in artist B dataset
    if similarity_matrix.size == 0:
        raise ValueError("similary matrix is empty")
    similar_songs_indices = similarity_matrix.argsort()[-num_recommendations-1:-1][::-1]

    # Verify indices and if they're not out of bound
    if any(index >= len(data_artist_B) or index < 0 for index in similar_songs_indices):
        raise IndexError("some indexes are out of bound")

    # Get these songs' info
    recommended_songs = data_artist_B.iloc[similar_songs_indices]

    return recommended_songs

## Test
get_recommendations('Run', 10)