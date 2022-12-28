import numpy as np
import pandas as pd
import os
import bz2
import pickle
import _pickle as cPickle

import re
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def data_read():
    
    """function to read data into pandas dataframe,
    further convert sentiment column into numerical values"""
    
    global df
    
    df = pd.read_csv("data/IMDB Dataset.csv")
    df['sentiment']= df['sentiment'].apply(lambda x : 1 if x == 'positive' else 0)
    
def html_tag_remover():
    
    """function to remove html tags using regex, 
    and store a copy of dataframe in variable"""
    
    global df_removed_tag
    
    df['review'] = df['review'].str.replace(r'<[^<>]*>', '', regex = True)
    df_removed_tag = df
    
def url_remover():
    
    """function to remove url using regex, 
    and store a copy of dataframe in variable"""
    
    global df_removed_url
    
    df['review'] = df['review'].str.replace(r'https ? ://\s+|www\.\s+', '', regex = True)
    df_removed_tag = df
    
def lowercase():
    
    """function to convert review into lowercase, 
    and store a copy of dataframe in variable"""
    
    global df_lower
    
    df['review'] = df['review'].str.lower()
    df_lower = df
    
def punctuation_remover():
    
    """function to remove punctuation using regex, 
    and store a copy of dataframe in variable"""
    
    global df_punc_removed
    
    df['review'] = df['review'].str.replace('[{}]'.format(string.punctuation), '', regex = True)
    df_punc_removed = df
    
def stopword_remover():
    
    """function to remove stopwords, 
    and store a copy of dataframe in variable"""
    
    global df_stopword_removed
    
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
    df_stopword_removed = df
    
def lemmatize_text():
    
    """function to lemmatize reviews, 
    and store a copy of dataframe in variable"""
    
    global df_lemmatized
    
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    df['review'] = df['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)]))
    df_lemmatized =  df

def data_preprocess():
    
    data_read()
    html_tag_remover()
    url_remover()
    lowercase()
    punctuation_remover()
    stopword_remover()
    lemmatize_text()
    
    return

def model_directory():
    try:
        os.makedirs("model")
    except OSError:
        pass

def compressed_pickle(directory, title, data):

    os.chdir(directory)

    with bz2.BZ2File(title + ".pbz2", "w") as f: 
        cPickle.dump(data, f)

    os.chdir("../")

def model_preparation():

    """split the dataset : 75% for training, 25% for testing

    Natural Language Processing technique of text modeling known as Bag of Words model.
    > Whenever we apply any algorithm in NLP, it works on numbers. 
    > We cannot directly feed our text into that algorithm. 
    > Hence, Bag of Words model is used to preprocess the text by converting it into a bag of words, 
    > which keeps a count of the total occurrences of most frequently used words

    saved the model in model.pkl
    """

    train, test = train_test_split(df, test_size = 0.25, random_state = 42)
    X_train, y_train = train['review'], train['sentiment']
    X_test, y_test = test['review'], test['sentiment']

    cnt_vec = CountVectorizer(ngram_range = (1, 3), binary = False)
    x_train_vector = cnt_vec.fit_transform(X_train)
    x_test_vector = cnt_vec.transform(X_test)

    multi_clf = MultinomialNB()
    multi_clf.fit(x_train_vector, y_train.values)
    predict_NB = multi_clf.predict(x_test_vector)

    delattr(cnt_vec, "stop_words_")

    compressed_pickle("model", "model", multi_clf)
    compressed_pickle("vectorizer", "vec", cnt_vec)

    #pickle.dump(multi_clf, open("model/model.pkl", "wb"))
    #pickle.dump(cnt_vec, open("model/vec.pkl", 'wb'))



def main():

    data_preprocess()
    model_directory()
    #compressed_pickle(title, data)
    model_preparation()

    return

if __name__ == "__main__":
    main()