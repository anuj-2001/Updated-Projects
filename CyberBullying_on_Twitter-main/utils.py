# Load Libraries
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import logging
import joblib
import string
import re
import os
import nltk
nltk.download('punkt')
stem = PorterStemmer() # load stemmer object

def clean_txt(txt):
    cleaned_txt = []
    url = re.compile(r'https?://\S+|www\.S+')
    html = re.compile(r'<.*?>')
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    punct_table = str.maketrans('', '', string.punctuation)
    for line in txt:
        line = line.lower()
        line = url.sub(r'', line)
        line = html.sub(r'', line)
        line = emoji_pattern.sub(r'', line)
        line = line.translate(punct_table)
        if len(line) >= 3:
            tokens = word_tokenize(line)
            words = [word for word in tokens if word.isalpha()]
            cleaned_txt += words
        str1 = ""
        
        for ele in cleaned_txt:
            str1 += ele+" "
    return str1

def preprocess_text(text):
    '''
    Perform preprocessing of text data

    Parameters
    ----------
    text : input raw text to preprocess

    Returns
    ---------
    processed_text : cleaned text
    '''
    text = text.lower()         # lower text

    #Substitute short forms to complete word
    text = re.sub('\'m',' am',text)
    text = re.sub('n\'t',' not',text)
    text = re.sub('ya','you ',text)
    text = re.sub('\'ve',' have',text)

    # replace links
    text = re.sub('https?://\S+',' URL ',text)

    # replace hastags
    text = re.sub('#\S+',' HASHTAGS ',text)

    # replace mentions
    text = re.sub('@\S+',' MENTIONS ',text)

    # replace emoji's
    text = re.sub('&#[0-9]+;',' EMOTICONS ',text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # perform stemming on text
    processed_text = ' '.join([stem.stem(w) for w in text.split()])

    return processed_text

def get_class_weights(y):
    '''
    Perform preprocessing of text data

    Parameters
    ----------
    y : input categorical labels for dataset

    Returns
    ----------
    class_weight : a dict containing  weights for each category
    '''

    # list of weights for labels
    weights = compute_class_weight('balanced',np.unique(y),y)

    # map list to dictionary
    class_weight = {col:val for col,val in zip(np.unique(y),weights)}

    return class_weight

def save_weights(model,vectorizer,path):
    '''
    save model and other preprocessing object as pickle file

    Parameters
    ----------
    model : model to use for inference
    vectorizer : preprocesses text to features
    path : path to folder to save pickle files

    Returns
    ----------
    None
    '''
    # saving model as pickle
    joblib.dump(model,os.path.join(path,'model.pkl'))
    # saving tfidfvectorizer as pickle
    joblib.dump(vectorizer,os.path.join(path,'vectorizer.pkl'))


def get_logger(path):

    '''
    Log info and error

    Parameters
    ----------
    path : path to save info and error

    Returns
    ----------
    logger : logger to log error and info
    '''
    logger = logging.getLogger(__name__) # create logger
    logger.setLevel(logging.DEBUG)  # set logging level to debug

    info_handler = logging.FileHandler(os.path.join(path,'info.log')) # file handler to save all log
    err_handler = logging.FileHandler(os.path.join(path,'error.err')) # file handler to save only error

    # set logging level with file handler
    info_handler.setLevel(logging.INFO)
    err_handler.setLevel(logging.ERROR)

    # set format for logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    err_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(info_handler)
    logger.addHandler(err_handler)

    return logger
