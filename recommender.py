import pandas as pd
from data_processing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Recommender:
    
    
    def __init__(self, games_df):
        self.df = games_df
        self.vectorizer = TfidfVectorizer()
        self.matriz = None
        self.cosine_sim = None
        
    def fit_values(self, features):
        gamesdf = self.df.reset_index()
        gamesdf = gamesdf[features]
        gamesdf = gamesdf.apply(lambda row: ' '.join(row.astype(str)), axis=1)
        matriz = self.vectorizer.fit_transform(gamesdf)
        self.cosine_sim = linear_kernel(matriz,matriz)

        return True
    
    def recommend_game(self, game_id: int):
        
        df = games.reset_index()
        game_index = df[df['id'] == game_id].index[0]
        scores = list(enumerate(self.cosine_sim[game_index]))
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[1:6]
        indexes = [score[0] for score in scores]
        recommended_games = games.reset_index().loc[indexes]
        recommended_games['affinity'] = [score[1] for score in scores]
        
        return recommended_games
    
    def recommend_by_user(self, user_id):
        
        user = user_items[user_items['user_id'] == user_id]
        user_games = pd.DataFrame(user['items'])
        user_games.sort_values('playtime_forever', ascending=False, inplace=True).head(5)
        
        df = pd.DataFrame()
        
        for index, game in user_games.iterrows():
            
            recommended_game = self.recommend_game(game['game_id'].astype(int))
            pd.concat(df, recommended_game)
        
        df.drop_duplicates(subset['game_id'], keep='first', inplace=True)
        df.sort_values('affinity', ascending=False, inplace=True)
        return df.head(5)