import numpy as np
import en_core_web_sm
import string
import flask
import pickle
from joblib import load
from flask import Flask,request, jsonify, render_template
application = Flask(__name__)

# Method to APP mode (GUI or API).
def get_app_mode(request):
    app_mode = request.values.get('mode')
    print('app_mode::',app_mode)
    return app_mode

@application.route('/')
def home():
    return render_template('index.html')
    
@application.route('/healthcheck', methods=['GET'])
def get_healthcheck():
    if get_app_mode(flask.request) == 'gui':
        return render_template('index.html', health_text='The API services are up and running!!')
    else:
        return jsonify('The API services are up and running!!')

@application.route('/ticket/assign', methods=['POST'])
def ticket_assignment():
    request_method = flask.request.method
    print('ticket_assignment request_method ::',request_method)
    # Works only for a single sample
    if get_app_mode(flask.request) == 'gui':
        input_desc = flask.request.values.get('desc')
        input_short_desc = flask.request.values.get('short_desc')
        feature_request = input_short_desc+' '+input_desc
    else:
        data = request.get_json(force=True)
        # Make prediction using model loaded from disk as per the data.
        api_input=[[data['short_desc'],data['desc'],data['caller']]]
        print("api_input",api_input)
        # Extract only short_desc and desc and merge as per model building
        feature_request = data['short_desc']+' '+data['desc']

    print("Before::",feature_request)
    feature_request = preprocess_ticket_data(feature_request)
    print("After::",feature_request)
    loaded_model = load('models/Model_KNN.sav')
    #print('Model loaded successfully')
    # load the vectorizer
    loaded_vector = load('models/vectorizer.sav')
    #print('feature_vector::',loaded_vector)
    # make a prediction
    prediction = loaded_model.predict(loaded_vector.transform([feature_request]))
    #prediction = process_ticket_data(feature_request, model)  # runs globally loaded model on the data
    print('Predicted Group::',prediction)
    # Take the first value of prediction
    output = prediction[0]
    print('Output::',output)
    if get_app_mode(flask.request) == 'gui':
        return render_template('index.html', prediction_text='Your ticket is assigned to Group {}'.format(output))
    else:
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