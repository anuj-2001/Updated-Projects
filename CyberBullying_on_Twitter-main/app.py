from flask import Flask,render_template,url_for,request
from utils import preprocess_text
from utils import clean_txt
import string
import joblib
import os
import tweepy as tw
from twython import Twython
import pandas as pd
import re
import string
import nltk
import random


app = Flask(__name__,template_folder='frontend_app/src')

path ='frontend_app/src/weight' # path to weight

vectorizer = joblib.load(os.path.join(path,'vectorizer.pkl')) # load tfidfvectorizer

model = joblib.load(os.path.join(path,'model.pkl')) # get model

output_map = ['Hate Speech','Offensive Language','Neither Hateful nor offensive Language'] # map output

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        inp_text = request.form.get('text') # get text input
        
        value = inp_text

        inp_text = inp_text.translate(str.maketrans('', '', string.punctuation)) # check if input is not only set of punctuation

        if len(inp_text.strip())==0 or inp_text is None: # check for empty string
            return render_template('base.html')
        
        consumer_key = "yaOWyO5TweXAP1OlpFW3eAsfr"
        consumer_secret = "jbT75bxaCgGDo51haDJ8w4Rpm3UmZKk4QF0uhiie0OGUfs7SHQ"
        access_token = "1217087118356316160-QTRnq6Rhfsnv7X0tCRJwvJplut2Axs"
        access_token_secret = "AiBmkCV1ygtXZ83BQTgJjfSRC0JmFCLMMC0wXCQvief6u"

        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)
        
        
        res = api.user_timeline(screen_name=inp_text, count=100, include_rts=True)
        tweets = [tweet.text for tweet in res]
        clean_text = ''.join(str(tweet) for tweet in tweets)
        
        text = preprocess_text(clean_text)
        # print(text)
        
        text = vectorizer.transform([text]) # convert text to numbers

        prediction = model.predict(text) # make prediction

        prediction = output_map[prediction[0]] # map prediction

        prediction = 'Prediction: '+ prediction
        if prediction[0]==0 or prediction[1]==1:
            api.report_spam(screen_name = inp_text)

    return render_template('base.html',output=prediction,message=value)

if __name__=='__main__':
    app.run()
