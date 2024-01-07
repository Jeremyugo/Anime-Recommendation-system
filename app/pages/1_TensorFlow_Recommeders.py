import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import sys

sys.path.append("/home/ubuntu/datascience/anime_rec/")
from scripts.TFRSModel import build_model, return_anime_names


# list of genres
genres = ['Action', 'Adventure', 'Ai', 'Arts', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 
          'Harem', 'Historical', 'Horror', 'Josei', 'Kids', 'Life', 'Magic', 'Martial', 'Mecha', 'Military', 'Music', 
          'Mystery', 'Parody', 'Police', 'Power', 'Psychological', 'Romance', 'Samurai', 'School', 'Scifi', 'Seinen', 
          'Shoujo', 'Shounen', 'Slice', 'Space', 'Sports', 'Super', 'Supernatural', 'Thriller', 'Vampire', 'Yaoi', 'Yuri']



# list of types
types = ['TV', 'Movie', 'ONA', 'OVA', 'Special']

# popularity
popularity = ["Niche", "Universal", "Spectacle"]
# calling anime names
anime_names = return_anime_names()

# loading model
@st.cache_resource
def load_weights():
    model = build_model()
    model.load_weights("/home/ubuntu/datascience/anime_rec/model/model_weights/")
    return model

# loading the model
model = load_weights()

emoji_1, emoji_2 = "\U0001F3AC", "\U0001F37F"
st.title(f'Anime Recommender {emoji_1} {emoji_2}')

with st.form("my_form"):
    genres_ = st.multiselect('Genre', genres)

    type_ = st.selectbox("Type", types)

    popularity_ = st.selectbox("Popularity", popularity)

    submit = st.form_submit_button("Submit")

    if submit:
        if genres_ != []:
            raw_input = {
                "genre": ", ".join(genres_),
                "Audience": popularity_,
                "type": type_
            }

            index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=10)
            index.index_from_dataset(
                tf.data.Dataset.zip(
                    (anime_names.batch(1000), anime_names.batch(1000).map(model.anime_model)))
            )

            input_dict = {key: tf.constant(
                np.array([value])) for key, value in raw_input.items()}

            _, animes = index(input_dict)

            Feedback = {}
            for anime in animes.numpy()[0]:
                raw_input['name'] = anime

                input_dict = {key: tf.constant(np.array([value])) for key, value in raw_input.items()}

                trained_anime_embeddings, trained_user_embeddings, predicted_Feedback = model(input_dict)
                Feedback[anime] = predicted_Feedback


            sorted_dict = sorted(Feedback.items(), key=lambda x: x[1], reverse=True)
            sorted_dict = {'Anime':[k.decode("utf-8") for k, v in sorted_dict], 'Predicted Feedback': [round(v.numpy()[0,0], 1) for k, v in sorted_dict]}

            st.subheader("Your Anime Recommendations")
            st.dataframe(pd.DataFrame(sorted_dict))
        
        else:
            st.error("Please select at least on Genre")
