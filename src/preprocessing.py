# preprocessing.py

import config
import pandas as pd
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from ftfy import fix_text
from unidecode import unidecode
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer 
import re
import os

#stop words that may arise from company names and words that seemingly are not useful for example pound sign or journey
EXTRA_STOP_WORDS = set([])

#regex for punctuations ro remove
REGEX_TO_REMOVE = '[' + re.escape(''.join(''.join(punctuation))) + ']'

STEMMER = PorterStemmer()
LEMMER = WordNetLemmatizer()

def combine_text(row, text_col='v2', bad_cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']):
    """Given the non empty bad columns creates a new feature by that has all the texts available

    Args:
        row : row of a dataframe

    Returns:
        the row with added column `text`
    """
    text = row[text_col]
    for col in bad_cols:
        if pd.notna(row[col]):
            text = text + ' ' + row[col]
    row['text'] = text
    return row


def get_stopwords(extra_stopwords=EXTRA_STOP_WORDS):
    """Function to create stop words to remove
    Args:
        extra_stopwords ([type], optional): given as  set of extra stop words to add. Defaults to EXTRA_STOP_WORDS.

    Returns:
        stop words
    """
    # spacy's english language ahas a good number of stop words to remove
    sp = spacy.load('en_core_web_sm')
    stop_words = sp.Defaults.stop_words
    stop_words |= extra_stopwords
    return stop_words

STOP_WORDS = get_stopwords()

MIN_LEN = 2 #minimum character length of a token 
	
def remove_words(s, words={}, min_len=MIN_LEN):
    """Function to remove words in a string
    Args:
        s ([type]): main string
        words ([type], optional): words to be removed. Defaults to {}.
        min_len ([type], optional): min_len	do not include words with len of min_len or smaller. Defaults to MIN_LEN.

    Returns:
        the modified string
    """
    text_tokens = word_tokenize(s)
    tokens_without_w = [word for word in text_tokens if not word in words and len(word)  > min_len]
    return ' '.join(tokens_without_w)
    

def clean_text(row, stopwords = STOP_WORDS, rx_to_remove = REGEX_TO_REMOVE, lemmer=LEMMER, stemmer=STEMMER,min_len=MIN_LEN):
    """	Function to clean a given text from a row of data frame
    Args:
        row ([type]): row of dataframe with a string column
        stopwords ([type], optional): stop words. Defaults to STOP_WORDS.
        rx_to_remove ([type], optional): any regex to remove by default the punctuations. Defaults to REGEX_TO_REMOVE.
        lemmer ([type], optional): a lemmatizer object. Defaults to LEMMER.
        stemmer ([type], optional): a stemmer object. Defaults to STEMMER.
        min_len ([type], optional): token of this size or smaller are not acceptable. Defaults to MIN_LEN.

    Returns:
        the modified row with a new column
    """
    s = row['text']
    # fix text
    s = fix_text(s)
    # replace bad chars
    s = unidecode(s)
    # lower case
    s = s.lower()
    # remove stop wrods
    s = remove_words(s, stopwords, min_len)
    # remove punctuations
    s = ' '.join(re.sub(rx_to_remove, ' ', s).split())
    # remove numbers
    s = ''.join(ch for ch in s if not ch.isdigit())
    # tokenize for lemmatization and stemming
    s_tokens = word_tokenize(s)
    # lemmatize
    s_tokens = [lemmer.lemmatize(word, pos='v') for word in s_tokens]
    # stem
    s = ' '.join([stemmer.stem(word) for word in s_tokens])
    s = ' '.join([ch for ch in s.split() if len(ch) > min_len])
    # remove stop words again to make sure the roots are also removed
    s = remove_words(s, stopwords, min_len)
    if s.strip() == '':
        s = np.nan
    row['cleaned_text']= s
    return row

def full_cleaning_routine(df, vectorizer, combine=False, bad_cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], target_col='v1',text_col='v2'):    
    if combine:
        df = df.apply(combine_text, axis = 1)
    else:
        df.loc[:,'text'] = df[text_col]
    if set(df.columns ) == set(bad_cols):
        df = df.drop(bad_cols, axis = 1)

    # if data has the labels
    if target_col is not None:
        if target_col in df.columns:
            target_map = {'ham':0, 'spam':1}
            df.loc[:,'target'] = df[target_col].map(target_map)

    df = df.sample(frac=1).reset_index(drop=True)
    # drop duplicates
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df.apply(clean_text, axis = 1)
    df=df.loc[pd.notna(df['cleaned_text'])]
    X = df['cleaned_text'].values
    X = vectorizer.transform(X)
    return X

if __name__ == "__main__":
    #read data
    df = pd.read_csv(os.path.join(config.INPUT_DIR, 'spams_kfolds.csv'), encoding='latin-1')

    df = df.apply(clean_text, axis = 1)
    # make sure no nas
    df=df.loc[pd.notna(df['cleaned_text'])]
    df = df[['cleaned_text', 'kfold', 'target']]
    # save data file
    df.to_csv(os.path.join(config.INPUT_DIR, 'train.csv'), index=False)