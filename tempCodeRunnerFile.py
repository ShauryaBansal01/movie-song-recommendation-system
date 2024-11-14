import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
import ast
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser

app = Flask(__name__)

# Step 1: Set up Spotify API credentials
SPOTIPY_CLIENT_ID = '5e0e815dff934fc09953867015fc0578'
SPOTIPY_CLIENT_SECRET = '7b03a6e61fbe44c5b13c71f8e226726a'

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Load movie data
credits = pd.read_csv(r"C:\ai project\Movie-recommendation-system-using-AI-main\tmdb_5000_credits.csv")
movies = pd.read_csv(r"C:\ai project\Movie-recommendation-system-using-AI-main\tmdb_5000_movies.csv")
movies = movies[movies['overview'].notna()]

credits_column_renamed = credits.rename(columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# Parse genres as list of genre names
movies_cleaned['genres'] = movies_cleaned['genres'].apply(lambda x: [genre['name'].lower() for genre in ast.literal_eval(x)])
movies_cleaned['tags'] = movies_cleaned['genres'].astype(str) + ' ' + movies_cleaned['overview']

# Extract unique genres
unique_genres = sorted(set(genre for genres in movies_cleaned['genres'] for genre in genres))

# Vectorize tags
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['tags'])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
rating_similarity = cosine_similarity(movies_cleaned[['vote_average']])
combined_similarity = 0.7 * sig + 0.3 * rating_similarity

# Mapping
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

# Music recommendations based on emotion
emotion_music_dataset = {
    "joy": {
        "en": [
            "Happy by Pharrell Williams", 
            "Good as Hell by Lizzo",
            "Shake It Off by Taylor Swift",
            "Can't Stop the Feeling! by Justin Timberlake",
            "Uptown Funk by Mark Ronson ft. Bruno Mars",
            "Best Day of My Life by American Authors",
            "I'm So Excited by The Pointer Sisters",
            "Walking on Sunshine by Katrina and the Waves"
        ],
        "hi": [
            "जश्न-ए-बरात (Jashn-e-Barat) by Vishal Dadlani", 
            "दिल धड़कने दो (Dil Dhadakne Do) by Priyanka Chopra & Farhan Akhtar",
            "तेरे जैसे यार (Tere Jaise Yaar) by A.R. Rahman",
            "सुनो सुनो (Suno Suno) by Vishal-Shekhar",
            "मन की माया (Man Ki Maya) by Shreya Ghoshal",
            "खुश रहो (Khush Raho) by A.R. Rahman",
            "शुभ मंगल सावधान (Shubh Mangal Saavdhan) by Arijit Singh",
            "यादें (Yaadein) by Atif Aslam"
        ]
    },
    "sadness": {
        "en": [
            "Someone Like You by Adele", 
            "Fix You by Coldplay",
            "Hurt by Christina Aguilera",
            "The Night We Met by Lord Huron",
            "Tears Dry On Their Own by Amy Winehouse",
            "Back to December by Taylor Swift",
            "Let Her Go by Passenger",
            "Someone You Loved by Lewis Capaldi"
        ],
        "hi": [
            "तुम ही हो (Tum Hi Ho) by Arijit Singh", 
            "चन्ना मेरेया (Channa Mereya) by Arijit Singh",
            "तेरे बिन (Tere Bin) by Rabbi Shergill",
            "आशिकी 2 (Aashiqui 2) by Arijit Singh",
            "सुन रहे हो न (Sun Rahe Ho Na) by Sonu Nigam",
            "तुमसे ही (Tumse Hi) by Mohit Chauhan",
            "इश्क़ सारा (Ishq Sara) by Atif Aslam",
            "रास्ता (Raasta) by Rahat Fateh Ali Khan"
        ]
    },
    # ... other emotions and music
}

def search_spotify_track(track_name):
    results = sp.search(q=track_name, type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_url = track['external_urls']['spotify']
        return track['name'], track['artists'][0]['name'], track_url
    else:
        return None

def recommend_music_on_spotify(emotion, language):
    if emotion in emotion_music_dataset:
        songs = emotion_music_dataset[emotion][language]
        recommendations = []
        for song in songs:
            result = search_spotify_track(song)
            if result:
                song_name, artist_name, url = result
                recommendations.append((f"{song_name} by {artist_name}", url))
        return recommendations
    else:
        return [("No recommendations available for this emotion.", None)]

# Recommendation functions
def give_recommendations_by_title(title, combined_similarity=combined_similarity, num_recommendations=10):
    idx = indices.get(title, None)
    if idx is None:
        return []
    sim_scores = list(enumerate(combined_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    return [i[0] for i in sim_scores]

def give_recommendations_by_genre(genre, num_recommendations=10):
    genre = genre.lower().strip()
    genre_movies = movies_cleaned[movies_cleaned['genres'].apply(lambda genres: genre in genres)]
    if genre_movies.empty:
        return [{"title": "No movies found for this genre", "rating": ""}]
    return [{'title': row['original_title'], 'rating': row['vote_average']} for _, row in genre_movies.head(num_recommendations).iterrows()]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    music_recommendations = []
    if request.method == "POST":
        movie_title = request.form.get("title_input", "").strip()
        genre_input = request.form.get("genre_input", "").strip()
        emotion = request.form.get("emotion", "").strip()
        language = request.form.get("language", "en").strip()
        
        # Movie Recommendations
        if movie_title:
            recommendations = give_recommendations_by_title(movie_title)
        elif genre_input:
            recommendations = give_recommendations_by_genre(genre_input)
        
        # Music Recommendations
        if emotion:
            music_recommendations = recommend_music_on_spotify(emotion, language)

    return render_template("index.html", recommendations=recommendations, genres=unique_genres, music_recommendations=music_recommendations)

if __name__ == "__main__":
    app.run(debug=True)
