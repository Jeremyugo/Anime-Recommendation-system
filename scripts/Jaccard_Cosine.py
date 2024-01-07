import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import sys

df = pd.read_csv("/home/ubuntu/datascience/anime_rec/data/processed/clean_data.csv")

def clean_genre(genre):
    genre_list = genre.split(',')
    genre = ",".join([x.strip(' ') for x in genre_list])
    return genre

df['genre'] = df['genre'].apply(clean_genre)
# we only need 1 entry of each movie since each movie has similar genres across its entries
df_subset = df.drop_duplicates(subset="name")
df_dummies = df_subset['genre'].str.get_dummies(sep=',')
df_dummies = pd.concat([df_dummies, df_subset['Audience'].str.get_dummies(), df_subset['type'].str.get_dummies()], axis=1)


def jaccard_distance():
    # calculating the jaccard distance between all genres
    jaccard_distance = pdist(df_dummies.values, metric="jaccard")

    # converting it to squareform
    squared_jaccard_distance = squareform(jaccard_distance)
    jaccard_array = 1 - squared_jaccard_distance

    # converting to dataframe
    distance_df = pd.DataFrame(jaccard_array, index=df_subset['name'], columns=df_subset['name'])
    return distance_df



df2 = pd.read_csv("/home/ubuntu/datascience/anime_rec/data/processed/cosine.csv")
# dataframe containing User_ID, name, and Feedback
uis = df2[['User_ID', 'name', 'Feedback']]


def get_cosine_similarity():
    # pivot table of uis dataframe
    uis_table = uis.pivot_table(index='User_ID', columns='name', values='Feedback')

    # average rating by row, i.e each movie
    avg_ratings = uis_table.mean(axis=1)

    # centering ratings around 0
    uis_pivot = uis_table.sub(avg_ratings, axis=0)

    uis_pivot.fillna(0, inplace=True)

    # Transpose data to get anime_pivot
    anime_pivot = uis_pivot.T
    
    # calculating Anime cosine similarity based on user ratings
    similarities = cosine_similarity(anime_pivot)

    # dataframe of cosine similarity
    cosine_similarity_df = pd.DataFrame(
        similarities, index=anime_pivot.index, columns=anime_pivot.index
    )
        
    return cosine_similarity_df