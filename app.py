import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.title("🎵 Music Recommendation System")

# Load dataset
df = pd.read_csv("spotify-tracks-dataset.csv")

# Clean data
df = df.dropna().drop_duplicates()
df['track_name'] = df['track_name'].str.lower()

# Features
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Scale features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Recommendation function
def recommend(song_name, n=5):
    song_name = song_name.lower()

    if song_name not in df['track_name'].values:
        return ["Song not found"]

    index = df[df['track_name'] == song_name].index[0]

    song_vector = df[features].iloc[index].values.reshape(1, -1)
    similarities = cosine_similarity(song_vector, df[features]).flatten()

    similar_indices = similarities.argsort()[::-1][1:n+1]

    return df.iloc[similar_indices]['track_name'].values


# Input box
song_input = st.text_input("Enter a song name:")

# Button
if st.button("Recommend"):
    if song_input:
        results = recommend(song_input)

        st.subheader("Recommended Songs:")
        for i, song in enumerate(results, 1):
            st.write(f"{i}. {song}")
    else:
        st.warning("Please enter a song name")