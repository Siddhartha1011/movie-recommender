
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
import gradio as gr
import requests
from typing import Optional

# ---------- Data Preparation ----------
# Load dataset (update the path to your local file)
df = pd.read_csv('/Users/imac/Desktop/Python/movie recomender/movies_metadata.csv', low_memory=False)

# Data preprocessing
df = df[['title', 'overview', 'genres', 'vote_average', 'vote_count']].dropna()
df = df[df['vote_count'] > 50]

def extract_genres(genre_str):
    try:
        genres = eval(genre_str)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''

df['genres'] = df['genres'].apply(extract_genres)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ---------- FastAPI Setup ----------
app = FastAPI()

@app.get('/recommend')
def recommend_movies(title: str, num_movies: Optional[int] = 5):
    if title not in indices:
        return {"error": "Movie not found"}

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_movies+1]

    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][['title', 'vote_average', 'genres']].to_dict(orient='records')
    return {"recommendations": recommendations}

# ---------- Gradio Interface ----------
def fetch_recommendations(movie_title):
    try:
        response = requests.get(f"http://localhost:8000/recommend?title={movie_title}")
        if response.status_code == 200:
            data = response.json()
            if "recommendations" in data:
                return "\n".join([f"{rec['title']} ({rec['vote_average']}) - {rec['genres']}" 
                               for rec in data["recommendations"]])
            return "Movie not found"
        return "Error fetching recommendations"
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=fetch_recommendations,
    inputs=gr.Textbox(label="Enter Movie Title"),
    outputs=gr.Textbox(label="Recommended Movies"),
    title="Movie Recommender System",
    description="Enter a movie title to get similar recommendations based on plot description."
)

# ---------- Main Execution ----------
if __name__ == "__main__":
    # To run both FastAPI and Gradio concurrently, use:
    # uvicorn.run(app, host="0.0.0.0", port=8000) in one terminal
    # and run the Gradio interface separately
    demo.launch()

