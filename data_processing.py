import pandas as pd
import ast

#dataframe games

games = pd.read_json('datasets/steam_games.json', lines=True)
games[games['price'] == 'Starting at $499.00'] = 499.00
games[games['price'] == 'Starting at $449.00'] = 449.00
games.dropna(subset=['id'],inplace=True)
games.id = games.id.apply(int)
games.set_index('id', inplace=True)
games.index = games.index.astype(int)
games.loc[~games['price'].apply(lambda x: isinstance(x, float)), 'price'] = 0

#dataframe user_items

rows = []
with open('datasets/users_items.json',  encoding = 'utf-8-sig') as a:
    for row in a.readlines():
        rows.append(ast.literal_eval(row))
user_items = pd.DataFrame(rows)
user_items = user_items

#dataframe user_reviews

rows = []
with open('datasets/user_reviews.json',  encoding = 'utf-8-sig') as a:
    for row in a.readlines():
        rows.append(ast.literal_eval(row))
user_reviews = pd.DataFrame(rows)