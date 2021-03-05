from flask import Flask, render_template, url_for, request, flash
import pickle
import joblib
import sys
import os
sys.path.insert(1, '../src/')
import config
import preprocessing 
import pandas as pd


app = Flask(__name__)

#removed key in app
# app.config['SECRET_KEY'] = '***************************'


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

#make options for reading the data fram
#make options for training within the app
#make jsonify
@app.route('/predict',methods=['POST'])
def predict():
    model = joblib.load(open(os.path.join(config.MODEL_DIR, 'model_nb_1_count.pkl'), 'rb'))
    vectorizer = joblib.load(open(os.path.join(config.MODEL_DIR, 'vectorizer_nb_count.pkl'), 'rb'))
    # df = pd.read_csv('./spam.csv', encoding = 'latin-1')
    # X = preprocessing.full_cleaning_routine(df, vectorizer = vectorizer)
    if request.method == 'POST':
        message = request.form['message']
        message_df=pd.DataFrame({'v2':[message]})
        X_msg = preprocessing.full_cleaning_routine(message_df, vectorizer = vectorizer, target_col = None, text_col = 'v2')
        my_prediction = model.predict(X_msg)
    return render_template('result.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
    