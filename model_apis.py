import numpy as np
import en_core_web_sm
import string
import pickle
from joblib import load
from flask import Flask,request, jsonify, render_template
application = Flask(__name__)

@application.route('/')
def home():
    return render_template("index.html")

@application.route('/healthcheck', methods=['GET'])
def get_healthcheck():
    return "true"

def analyzeSentiment(text):
    #print("read request is working")
    sentimentModel = load("saved_models/sentiment_Model.sav")
    label = 0
    dictionary = {}
    label  = sentimentModel.predict(text)[0]
    if label == 0:
        dictionary.update({"Sentiment of tweet is": "Positive"})
    else:
        dictionary.update({"Sentiment of tweet is": "Negative"})
    return dictionary
    
#http://localhost:9052/SentimentAnalysis?input=this is first tweet
@application.route('/SentimentAnalysis', methods=['POST', 'GET'])
def SentimentAnalysis():
    #param = request.json
    param=(request.args.get('input',None))
    text = list(param)
    print(text)
    #text = 'i am done'  
    #text = list(text)
    rt = analyzeSentiment(text)
    ##js=json.dumps(rt)
    return jsonify(rt)

@application.route('/irispredict', methods=['POST'])
def iris_prediction():
    # Works only for a single sample
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    predict_request=[[data['sl'],data['sw'],data['pl'],data['pw']]]
    predict_request=np.array(predict_request)
    print(predict_request)
    model = pickle.load(open('saved_models/iris_Model.sav', 'rb'))
    prediction = model.predict(predict_request)  # runs globally loaded model on the data
    print(prediction)
    # Take the first value of prediction
    output = prediction[0]
    print(output)
    return jsonify(int(output))

@application.route('/ticket/assign', methods=['POST'])
def ticket_assignment():
    # Works only for a single sample
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    api_input=[[data['short_desc'],data['desc'],data['caller']]]
    print("api_input",api_input)
    # Extract only short_desc and desc and merge as per model building
    feature_request=data['short_desc']+' '+data['desc']
    #print("Before::",feature_request)
    feature_request = preprocess_ticket_data(feature_request)
    print("After::",feature_request)
    loaded_model = load('saved_models/Model_KNN.sav')
    #print('Model loaded successfully')
    # load the vectorizer
    loaded_vector = load('saved_models/vectorizer.sav')
    #print('feature_vector::',loaded_vector)
    # make a prediction
    prediction = loaded_model.predict(loaded_vector.transform([feature_request]))
    #prediction = process_ticket_data(feature_request, model)  # runs globally loaded model on the data
    print('Predicted Group::',prediction)
    # Take the first value of prediction
    output = prediction[0]
    #output = 'GRP_0'
    print('Output::',output)
    return jsonify(output)

def preprocess_ticket_data(ticket_text):
    #vectorizing the tweet by the pre-fitted tokenizer instance    
    #print('Inside preprocess_ticket_data')
    print('Input ticket:',ticket_text)
    # Perform following preprocessing on the text data as per model
    # 1. lower case 
    ticket_text = ticket_text.lower()
    #print('After Lower Case::',ticket_text)
    # 2. clean dataset - remove special, unwanted characters
    ticket_text = clean_data(ticket_text)
    # 3. Remove punctuation
    # 4. Remove Stopwords
    # 5. Remove accented characters
    # 6. Lemmatize
    ticket_text = lemmatize(ticket_text)
    #print('After lemmatize::',ticket_text)
    
    return ticket_text

def getList():
    rmvList = []
    rmvList += ['received from:(.*)']  # received data line
    rmvList += ['From:(.*)']  # from line
    rmvList += ['Sent:(.*)']  # sent to line
    rmvList += ['To:(.*)']  # to line
    rmvList += ['CC:(.*)']  # cc line
    rmvList += ['https?:[^\]\n\r]+']  # https & http
    rmvList += ['[\r\n]']  # for \r\n
    rmvList += ['[^a-zA-Z\s]']
    rmvList += ['sid_']
    rmvList += ['erp ']
    return rmvList

def lemmatize(text):
    nlp = en_core_web_sm.load()
    spacy_doc = nlp(text) # Parse the sentence using the loaded 'en' model object `nlp`
    return " ".join([token.lemma_ for token in spacy_doc if token.lemma_ !='-PRON-'])

def clean_data(text):
    rmvList = getList()
    for ex in rmvList:
        text = text.replace(ex.lower(), '')
    #print('After removing Special Chars::',text)    
    text = str(text).translate(str.maketrans('', '', string.punctuation))
    #print('After Punctuation::',text)
    return text

if __name__ == '__main__':
   application.run(host="0.0.0.0",port=9052,debug=False)