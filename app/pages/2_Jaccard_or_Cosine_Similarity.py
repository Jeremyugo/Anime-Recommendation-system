import pandas as pd
import numpy as np
import streamlit as st
import sys

sys.path.append('/home/ubuntu/datascience/anime_rec/')
from scripts.Jaccard_Cosine import jaccard_distance, get_cosine_similarity

@st.cache_data
def load_jaccard_data():
    return jaccard_distance()

distance_df = load_jaccard_data()

anime_list = list(distance_df.index)

@st.cache_data
def get_jaccard_recommendation(name):
    temp_df = distance_df[[name]].reset_index().rename({'name':'Anime', name:'jaccard_similarity'}, axis=1)
    temp_df = temp_df.sort_values(by='jaccard_similarity', ascending=False)[1:11]
    return temp_df.reset_index(drop=True)

@st.cache_data
def load_cosine_data():
    return get_cosine_similarity()
    
cosine_similarity_df = load_cosine_data()

@st.cache_data
def get_cosine_recommendations(name):
    return cosine_similarity_df[name].sort_values(ascending=False)[1:11].reset_index().rename({name: "Cosine similarity", "name":"Anime"}, axis=1)


emoji_1, emoji_2 = "\U0001F3AC", "\U0001F37F"
st.title(f'Anime Recommender {emoji_1} {emoji_2}')

with st.form("my_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        rec_option = st.radio("Pick one", ["Jaccard", "Cosine"])
    
    with col2:
        anime = st.selectbox("Anime", anime_list)

    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if rec_option == "Jaccard":
            rec_df = get_jaccard_recommendation(anime)
        else:
            rec_df = get_cosine_recommendations(anime)
        
        
        st.subheader(f"Your Anime Recommendations using {rec_option} similarity")
        st.dataframe(rec_df)
            
        