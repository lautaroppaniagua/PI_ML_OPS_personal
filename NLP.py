import pandas as pd
import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
wordnet_lemmatize = WordNetLemmatizer()


class NLP_Model:
    
    def __init__(self,texts:list, label_values ,sp_languages, additional_stopwords):
        self.texts = texts
        self.label_values = label_values
        self.sp_languages = sp_languages
        self.additional_stopwords = additional_stopwords
        self.log = None
        self.stopwords = full_stopwords(self.sp_languages, self.additional_stopwords)
        self.count_vec = CountVectorizer(max_features=500,ngram_range=(1,2))
        self.threshold = 0.80

        
    def fit_values(self):
    
        reviews = self.texts
    
    
        lemm_reviews = [process_text(review,self.stopwords) for review in reviews]
    
        
        matriz = self.count_vec.fit_transform(lemm_reviews)
        x = matriz.toarray()
        y = self.label_values
        self.log = LogisticRegression(max_iter=1000)
        self.log.fit(x,y)
        
    def predict(self, text):
        
        values_cat = {
            True: 'Positive',
            False: 'Negative',
            None: 'Neutral'
        }
        
        if self.log is not None:
            text = process_text(text,self.stopwords)
            
            x = self.count_vec.transform([text])
            predict = self.log.predict(x)
            maxprob = np.max(self.log.predict_proba(x),axis=1)[0]
            if maxprob < self.threshold:
                predict = values_cat[None]
                
            else:
                predict = values_cat[predict[0]]
                
            return predict
        
def full_stopwords(languages, additional_words = []):
    stopwords = []
    for lang in languages:
            stopwords.extend(nltk.corpus.stopwords.words(lang))
    stopwords.extend(additional_words)
    return stopwords

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag,wordnet.NOUN)

def process_text(text, stopwords):
    text = re.sub("[^a-zA-Z]"," ",str(text))
    text = text.lower()
    text = nltk.word_tokenize(text)
    lemmatize = [wordnet_lemmatize.lemmatize(w, get_wordnet_pos(w)) for w in text]
    text = [palabra for palabra in text if len(palabra)>3]
    text = [palabra for palabra in text if palabra not in stopwords]
    
    text = ' '.join(text)
    return text

def extract_reviews(df):
    reviewslist = []
    for index, user in df.iterrows():
        if len(user['reviews']) > 0:
            for review in user['reviews']:
                reviewslist.append(review)
            
    return pd.DataFrame(reviewslist)


def extract_dev_reviews(df, dev_ids):
    review_list = []
    for index, user in df.iterrows():
        if len(user['reviews']) > 0:
            for review in user['reviews']:
                if review['item_id'] in dev_ids:
                    review_list.append(review['review'])
                
    return review_list

