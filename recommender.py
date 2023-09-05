import pandas as pd
from data_processing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_game(game_id):
    
    vectorizer = TfidfVectorizer()
    matrix = games['genres'].apply(lambda x: ' '.join(x))
    matrix = vectorizer.fit_transform(matrix)
    game = games.loc[game_id]
    concat_genres = ' '.join(game['genres'])
    vector = vectorizer.transform([concat_genres])
    cos_sim = cosine_similarity(matrix, vector)
    scores = list(enumerate(cos_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[1:6]
    indexes = [score[0] for score in scores]
    recommended_games = games.reset_index().loc[indexes]
    recommended_games['affinity'] = [float(score[1]) for score in scores]
    
    return recommended_games
