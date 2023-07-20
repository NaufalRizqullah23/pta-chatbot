import numpy as np
import pandas as pd
import string
import random
import json
import nltk
from flask import Flask, render_template, request
from keras.models import load_model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
import pickle


app = Flask(__name__)

#load the tokenizer and label encoder
with open("tokenizer.pkl","rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open("label_encoder.pkl","rb") as le_file:
    le = pickle.load(le_file)


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


#removing punctuation
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))


#lemmatization
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#class sorting
classes = sorted(list(set(classes)))

# load the tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

#padding
x_train = pad_sequences(train)

# initialize Label Encoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])


# #input shape
input_shape = x_train.shape[1]

# #define vocabulary
vocabulary = len(tokenizer.word_index)

# #output length
output_length = le.classes_.shape[0]

# # load the model
# model = load_model('model_rnn_fix.h5')

#load model architecture
with open("model_rnn.json", "r") as json_file:
    loaded_model_json = json_file.read()

#load model weight
model = model_from_json(loaded_model_json)
model.load_weights("model_rnn.h5")


threshold = 0.65


def preprocess_message(message):
    text_p = []
    prediction_input = message
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    text_p.append(prediction_input)
    # prediction_input = message.lower()
    # prediction_input = ''.join(
    #     [char for char in prediction_input if char not in string.punctuation])
    # text_p = [prediction_input]
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    print("input shape: ",prediction_input.shape)
    return prediction_input

def postprocess_response(response):
    response_max = np.max(response)
    response_class = np.argmax(response)

    if response_max >= threshold:
        response_tag = le.inverse_transform([response_class])[0]
        processed_response = random.choice(responses[response_tag])
    else:
        processed_response = "Maaf, saya belum tahu.\nUntuk info lebih lanjut, kamu bisa mengunjungi situs berikut : https://www.pta-bandung.go.id/"
    
    return processed_response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    processed_message = preprocess_message(message)

    #debuggin
    print("input message: ",message)
    print("processed message: ",processed_message)

    # make prediction using the model
    predicted_output = model.predict(processed_message)[0]

    # convert the predicted output into desired format
    processed_response = postprocess_response(predicted_output)

    return processed_response


if __name__ == '__main__':
    app.run()
