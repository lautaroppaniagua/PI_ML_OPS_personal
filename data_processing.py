import pandas as pd


#dataframe games

games = pd.read_json('datasets/games.json.gzip', compression='gzip')


#dataframe user_items

user_items = pd.read_json('datasets/user_items.json.gzip', compression='gzip')

#dataframe user_reviews

user_reviews = pd.read_json('datasets/user_reviews.json.gzip', compression='gzip')
