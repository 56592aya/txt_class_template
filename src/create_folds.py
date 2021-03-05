# create_folds.py

import config
import os
import pandas as pd
from preprocessing import combine_text
from sklearn.model_selection import StratifiedKFold

def create_folds(n_splits = 5, bad_cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], target_col='v1'):
    df = pd.read_csv(config.INPUT_FILE, encoding = 'latin-1')
    df = df.apply(combine_text, axis = 1)
    df = df.drop(bad_cols, axis = 1)

    #randomize the rows
    df = df.sample(frac=1).reset_index(drop=True)
    # drop duplicates
    df.drop_duplicates(subset=[['text', 'target']], inplace=True)

    #create new fold column
    df['kfold'] = -1 
    
    #map values to numbers for the target column
    target_map = {'ham':0, 'spam':1}
    df.loc[:,'target'] = df[target_col].map(target_map)
    y = df['target'].values
    
    kf = StratifiedKFold(n_splits=5)

    for i, (_train_id, _test_id) in enumerate(kf.split(X=df, y = y)):
        df.loc[_test_id, 'kfold'] = i
    
    df = df[['text','kfold','target']]
    df.to_csv(os.path.join(config.INPUT_DIR, 'spams_kfolds.csv'), index=False)

if __name__ == "__main__":
    create_folds()

