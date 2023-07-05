import numpy as np
import pandas as pd
import string
import random
import json
import nltk
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# load the model
model = load_model('chat_model.h5', compile=False)

# load the dataset
with open('intents.json', 'r', encoding='utf-8') as content:
    data1 = json.load(content)

tags = []
inputs = []
responses = {}
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

data = pd.DataFrame({"patterns": inputs, "tags": tags})

# load the tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['patterns'])

# initialize Label Encoder
le = LabelEncoder()
le.fit_transform(data['tags'])


def preprocess_message(message):
    prediction_input = message.lower()
    prediction_input = ''.join(
        [char for char in prediction_input if char not in string.punctuation])
    text_p = [prediction_input]
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input = pad_sequences(
        prediction_input, maxlen=model.input_shape[1])
    return prediction_input


def postprocess_response(response):
    response_tag = np.argmax(response)
    response_tag = le.inverse_transform([response_tag])[0]
    processed_response = random.choice(responses[response_tag])
    return processed_response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    processed_message = preprocess_message(message)

    # make prediction using the model
    predicted_output = model.predict(processed_message)[0]

    # convert the predicted output into desired format
    processed_response = postprocess_response(predicted_output)

    return processed_response


if __name__ == '__main__':
    app.run()
