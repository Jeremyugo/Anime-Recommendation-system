import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs


df = pd.read_csv('/home/ubuntu/datascience/anime_rec/data/processed/clean_data.csv')

# converting the df to dictionary
df_dict = {name: np.array(val) for name, val in df.items()}

# converting df dictionary to tensor slices
data = tf.data.Dataset.from_tensor_slices(df_dict)

# dictionary of unique values
vocabularies = {}
for feature in df_dict:
    if feature != 'Feedback':
        vocab = np.unique(df_dict[feature])
        vocabularies[feature] = vocab


anime_names = tf.data.Dataset.from_tensor_slices(vocabularies['name'])


def return_anime_names():
    return anime_names


class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        # Genre
        self.genre_vectorizer = keras.layers.TextVectorization(
            max_tokens=max_tokens, split="whitespace")
        self.genre_vectorizer.adapt(vocabularies['genre'])
        self.genre_text_embedding = keras.Sequential([
            self.genre_vectorizer,
            keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            keras.layers.GlobalAveragePooling1D()
        ])

        # Type
        self.type_embedding = keras.Sequential([
            keras.layers.StringLookup(
                vocabulary=vocabularies['type'],
                mask_token=None),
            keras.layers.Embedding(len(vocabularies['type'])+1, 32)
        ])

        # Audience
        self.audience_embedding = keras.Sequential([
            keras.layers.StringLookup(
                vocabulary=vocabularies['Audience'],
                mask_token=None),
            keras.layers.Embedding(len(vocabularies['Audience'])+1, 32)
        ])

    def call(self, inputs):

        return tf.concat([
            self.genre_text_embedding(inputs['genre']),
            self.type_embedding(inputs['type']),
            self.audience_embedding(inputs['Audience'])
        ], axis=1)


class AnimeModel(keras.Model):

    def __init__(self,):
        super().__init__()

        max_tokens = 10_000

        # Anime name
        self.anime_vectorizer = keras.layers.TextVectorization(
            max_tokens=max_tokens)
        self.anime_vectorizer.adapt(anime_names)
        self.anime_text_embedding = keras.Sequential([
            self.anime_vectorizer,
            keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            keras.layers.GlobalAveragePooling1D()
        ])

        self.anime_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=vocabularies['name'],
                mask_token=None),
            tf.keras.layers.Embedding(len(vocabularies['name'])+1, 32)
        ])

    def call(self, inputs):

        return tf.concat([
            self.anime_embedding(inputs),
            self.anime_text_embedding(inputs),
        ], axis=1)


tf.random.set_seed(7)
np.random.seed(7)


class TFRSModel(tfrs.models.Model):

    def __init__(self,):
        super().__init__()

        # handles how much weight we want to assign to the rating and retrieval task when computing loss
        self.rating_weight = 0.5
        self.retrieval_weight = 0.5

        # UserModel
        self.user_model = keras.Sequential([
            UserModel(),
            keras.layers.Dense(32)
        ])

        # AnimeModel
        self.anime_model = keras.Sequential([
            AnimeModel(),
            keras.layers.Dense(32)
        ])

        # Deep & Cross layer
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=None, kernel_initializer='he_normal')

        # Dense layers with l2 regularization to prevent overfitting
        self._deep_layers = [
            keras.layers.Dense(512, activation='relu',
                               kernel_regularizer='l2'),
            keras.layers.Dense(256, activation='relu',
                               kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'),
        ]

        # output layer
        self._logit_layer = keras.layers.Dense(1)

        # Multi-task Retrieval & Ranking
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=anime_names.batch(128).map(self.anime_model)
            )
        )

    def call(self, features) -> tf.Tensor:
        user_embeddings = self.user_model({
            'genre': features['genre'],
            'type': features["type"],
            'Audience': features["Audience"]
        })

        anime_embeddings = self.anime_model(
            features['name']
        )

        x = self._cross_layer(tf.concat([
            user_embeddings,
            anime_embeddings], axis=1))

        for layer in self._deep_layers:
            x = layer(x)

        return (
            user_embeddings,
            anime_embeddings,
            self._logit_layer(x)
        )

    def compute_loss(self, features, training=False):
        user_embeddings, anime_embeddings, rating_predictions = self.call(
            features)
        # Retrieval loss
        retrieval_loss = self.retrieval_task(user_embeddings, anime_embeddings)
        # Rating loss
        rating_loss = self.rating_task(
            labels=features['Feedback'],
            predictions=rating_predictions
        )

        # Combine two losses with hyper-parameters (to be tuned)
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)


def build_model():
    """ instantiates a model anmd compiles it """

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # instantiating the model
    model = TFRSModel()
    model.compile(optimizer=keras.optimizers.Adam(0.01))

    return model
