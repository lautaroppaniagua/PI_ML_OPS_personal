import pandas as pd
import os
from fastapi import FastAPI
from data_processing import games, user_items, user_reviews
from NLP import *
import nltk
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

#Inicializacion NLP

all_reviews = extract_reviews(user_reviews)
nlp_languages = ['spanish','english','russian','indonesian']
extra_sp = ['game','get','play']
nlp_model = NLP_Model(list(all_reviews['review']),list(all_reviews['recommend']),nlp_languages, extra_sp)
nlp_model.fit_values()


app = FastAPI()
app.title = 'PI_ML_OPS'

@app.get('/userdata')
def userdata(User_id: str):
    
    try:
        items = pd.DataFrame(*user_items[user_items['user_id'] == User_id]['items'])['item_id'].astype(int)
        items = games.merge(items, how='inner',left_on='id',right_on='item_id')
        TotalSpent = items['price'].astype(float).sum()
        TotalItems = items['item_id'].count()
        recomendations = pd.DataFrame(*user_reviews[user_reviews.user_id == User_id].reviews).recommend.value_counts(normalize=True)
    
        result = ({'Total gastado': TotalSpent,'Porcentaje de recomendacion':str(recomendations[0]*100) + '%', 'Total de items': int(TotalItems)})
    
        return result

    except:
        return None

def ConvertDate(date: str):
    try:
        return pd.to_datetime(date, format='%B %d, %Y')
    except:
        return pd.to_datetime('1678-01-01')     

@app.get('/countreviews')
def countreviews(FechaInicio, FechaFinal: str):
    FechaInicio = pd.to_datetime(FechaInicio)
    FechaFinal = pd.to_datetime(FechaFinal)
    series_list = []
    for index, user in user_reviews.iterrows():
        for review in user.reviews:
            if FechaInicio < ConvertDate(review['posted'][7:].replace('.','')) < FechaFinal:
                serie = pd.Series(data=review)
                serie['user_id'] = user.user_id
                series_list.append(serie)
            else:
                pass
    
    try:
        reviewsdf = pd.DataFrame(series_list)
        totalusuarios = reviewsdf.user_id.nunique()
        recomendacion_por = str(round(reviewsdf['recommend'].value_counts(normalize=True).values[0]*100))+'%'
        return {'Cantidad usuarios':totalusuarios, 'Porcentaje de recomendacion': recomendacion_por}
        
    except:
        return None
    
@app.get('/developer')
def developer(desarrollador: str):
    developer_games = games[games['developer'] == desarrollador]
    developer_games['year'] = (pd.to_datetime(developer_games['release_date'])).dt.year
    developer_games = developer_games[['year','price']]
    developer_games['price'] = developer_games['price'].astype(float)
    
    free_peryear = developer_games[developer_games['price'] == 0].groupby('year')['price'].count() 
    total_peryear = developer_games.groupby('year')['price'].count()
    
    result = ((free_peryear / total_peryear)*100).to_frame().fillna(0)
    result.rename(columns = {'price': 'year_percentage'}, inplace=True)
    
    return result.to_dict()

@app.get('/genre')
def genre(genero: str):
    genres_path = 'datasets/genresrank.csv'
    if not os.path.exists(genres_path):
        genres = games.genres.explode().value_counts().reset_index()
        genres['count'] = 0
        for index, user in user_items.iterrows():
            user_games = pd.DataFrame(user['items'])
            if len(user_games) > 0:
                user_games.item_id = user_games.item_id.apply(int)   
                user_games = games.merge(user_games, how='right', left_index=True, right_on='item_id')[['genres', 'playtime_forever']]
                user_games = user_games.explode('genres').groupby('genres').sum().reset_index()
                played_hours = genres.merge(user_games, left_on='genres',right_on='genres')['playtime_forever']
                
                genres['count'] = genres['count'] + played_hours
                genres.fillna(0, inplace=True)
            
        genres.to_csv(genres_path)

        return genre(genero)
        
    else:
        genres_rank = pd.read_csv(genres_path)
        genres_rank = genres_rank.sort_values('count', ascending=False).reset_index(drop=True)
        puesto_rank = int(genres_rank[genres_rank['genres'] == genero].index[0]) + 1
        return {genero: puesto_rank}
        


@app.get('/userforgenre')
def userforgenre(genero: str):
    SeriesList = []
    genre_games = games.explode('genres')
    genre_games = genre_games[genre_games['genres'] == genero]
    for index, user in user_items.iterrows():
        user_games = pd.DataFrame(user['items'])
        if len(user_games)>0:
            
            user_games = user_games[['item_id','playtime_forever']]
            user_games['item_id'] = user_games['item_id'].astype(int)
        
            played_hours = games.merge(user_games, how='inner', left_index=True, right_on='item_id')['playtime_forever'].sum()
        
            SeriesList.append(pd.Series({'user_id': user['user_id'], 'user_url': user['user_url'], 'played_hours': played_hours}))
        else:
            pass
    
    MaxHoursUser = pd.DataFrame(SeriesList).sort_values('played_hours',ascending=False).iloc[0]
    MaxHoursUser['played_hours'] = MaxHoursUser['played_hours'] / 60
    
    return MaxHoursUser.to_dict()
    

@app.get('/sentiment_analysis')
def sentiment_analysis(empresa_desarrolladora: str):
    
    developer_ids = list(games[games['developer']==empresa_desarrolladora].index)
    developer_ids = [str(id) for id in developer_ids]
    
    dev_reviews = extract_dev_reviews(user_reviews, developer_ids)
    
    total_count = {'Positive': 0,
                   'Negative': 0,
                   'Neutral': 0}
    
    for review in dev_reviews:
        sentiment_result = nlp_model.predict(review)
        total_count[sentiment_result] += 1
        
    return total_count
            
