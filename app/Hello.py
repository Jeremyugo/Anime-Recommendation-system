import streamlit as st

emoji_1, emoji_2 = "\U0001F3AC", "\U0001F37F"
st.title(f'Anime Recommender {emoji_1} {emoji_2}')

st.markdown(
    """
    This is a streamlit app for recommending Animes.
    There are 2 options located in the siderbar:
    
    1. TensorFlow Recommenders
    2. Jaccard or Cosine Similarity
    """
)


st.image("/home/ubuntu/datascience/anime_rec/app/anime.jpeg")