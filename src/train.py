# train.py

import config
import os
import pandas as pd
import numpy as np
import joblib
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

def train(df, fold, vectorizer, vec):
    
    df_train = df.loc[df['kfold'] != fold].reset_index(drop=True)
    df_test  = df.loc[df['kfold'] == fold].reset_index(drop=True)

    X_train = df_train['cleaned_text'].values
    y_train = df_train['target'].values

    X_test = df_test['cleaned_text'].values
    y_test = df_test['target'].values
    
    
    
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = MultinomialNB()

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    report = metrics.classification_report(y_test, preds)

    print(f"Fold={fold}, Classification Report from vectorizer {vec} \n {report}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_DIR, f"model_nb_{fold}_{vec}.pkl"))

    


if __name__ == "__main__":
    vectorizer_dict = {
        'count':CountVectorizer(tokenizer=word_tokenize, token_pattern = None),
        'tfidf':TfidfVectorizer(tokenizer=word_tokenize, token_pattern = None),
    }
    df = pd.read_csv(os.path.join(config.INPUT_DIR, 'train.csv'))
    X = df['cleaned_text'].values
    
    for vec in ['count', 'tfidf']:
        vectorizer = vectorizer_dict[vec]
        vectorizer.fit(X)
        #save the vectorizer
        joblib.dump(vectorizer, os.path.join(config.MODEL_DIR, f"vectorizer_nb_{vec}.pkl"))
        for fold in range(5):
            train(df, fold, vectorizer, vec)

    

