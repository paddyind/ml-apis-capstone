import os
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import time

filename = "data//twitter_sentiments.csv"

def readData(filename):
    data = pd.read_csv(filename)
    return data

def tweet_cleaner(text):
    
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    
    
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

def preprocess(dataframe):
    data = dataframe
    data['len'] = [len(t) for t in data.tweet]
    data['cleantext'] = data['tweet'].apply(lambda x: tweet_cleaner(x))
    data['newlen'] = data['cleantext'].apply(lambda x: len(x))
    
    return data


def buildVectorizer(low,high,Corpus):
    tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, 
                                       stop_words=ENGLISH_STOP_WORDS, ngram_range=(low,high))
    
    tfidf_vectorizer.fit(Corpus)
    return tfidf_vectorizer


def BuildModel(vectorizer, data):
    train, test = train_test_split(data, test_size = 0.2, stratify = data['label'], random_state=21)
    train_idf = vectorizer.transform(train.cleantext)
    test_idf  = vectorizer.transform(test.cleantext)
    rfc = RandomForestClassifier()
    rfc.fit(train_idf, train.label)
    predict = rfc.predict(test_idf)
    TrainingF1Score = f1_score(y_true= train.label, y_pred= rfc.predict(train_idf))
    TestingF1Score = f1_score(y_true= test.label, y_pred= rfc.predict(test_idf))
    
    TrainingF1Score = 0.9
    TestingF1Score = 0.6
    
    if TestingF1Score <= TrainingF1Score - (0.1*(TrainingF1Score)):
        print("Model is suffering from overfitting")
        
    return rfc
    
        

def buildPipeline(vectorizer, model, data):
    
    modelFileName = "saved_models/sentiment_Model.sav"
    
    pipeline = Pipeline(steps= [('tfidf',vectorizer),('model', model)])
    
    train, test = train_test_split(data, test_size = 0.2, stratify = data['label'], random_state=21)
    # fit the pipeline model with the training data
    pipeline.fit(train.cleantext, train.label)
    # checking that model exists or not
    backupStatus = backupCurrentModel(modelFileName)
    print('Model backup status ',backupStatus)
    dump(pipeline, filename=modelFileName)
    print("Model has been serialized")
    

def backupCurrentModel(modelFileName):
    
    status = 'false'
    #Get Timestamp to rename file
    T = time.asctime()
    T = T.replace(" ", "_") 
    T = T.replace(":", "_")
    # checking that file1.txt exists or not
	# if does not exist - will open myfile and read
    if os.path.exists(modelFileName):
        print(modelFileName," does exist.")
        # changing the file name
        os.rename(modelFileName, modelFileName+"__"+T)
        status = 'true'
    else:
        print(modelFileName," does not exist.")
    
    return status

if __name__ == '__main__':
    data = readData(filename)
    DF = preprocess(data)
    Corpus = [each for each in DF['cleantext']]
    vectorizer = buildVectorizer(1,3,Corpus)
    model  = BuildModel(vectorizer, data)
    buildPipeline(vectorizer, model, data)
    