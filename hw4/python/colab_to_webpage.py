"""
This notebook will help you go straight from training a model in Colab to deploying it in a webpage with TensorFlow.js - without having to leave the browser.

Configure this notebook to work with your GitHub account by populating these fields.
"""

!pip install tensorflowjs

"""# Git Setup"""

# your github username
USER_NAME = "enter your user name" 

# the email associated with your commits
# (may not matter if you leave it as this)
USER_EMAIL = "enter your email address" 

# the user token you've created (see the lecture 8 slides for instructions)
TOKEN = "enter your token" 

# site name
# for example, if my user_name is "foo", then this notebook will create
# a site at https://foo.github.io/hw4/
SITE_NAME = "hw4"

"""Next, run this cell to configure git."""

!git config --global user.email {USER_NAME}
!git config --global user.name  {USER_EMAIL}

"""Clone your GitHub pages repo (see the lecture 8 slides for instructions on how to create one)."""

import os
repo_path = USER_NAME + '.github.io'
if not os.path.exists(os.path.join(os.getcwd(), repo_path)):
  !git clone https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{USER_NAME}.github.io

os.chdir(repo_path)
!git pull

"""Create a folder for your site."""

project_path = os.path.join(os.getcwd(), SITE_NAME)
if not os.path.exists(project_path): 
  os.mkdir(project_path)
os.chdir(project_path)

"""These paths will be used by the converter script."""

# DO NOT MODIFY
MODEL_DIR = os.path.join(project_path, "model_js")
if not os.path.exists(MODEL_DIR):
  os.mkdir(MODEL_DIR)

"""# Project Gutenberg"""

import nltk
nltk.download('gutenberg')
nltk.download('punkt')

from nltk.corpus import gutenberg
gutenberg.fileids()

books = ['austen-emma.txt', 'chesterton-brown.txt','milton-paradise.txt']

x = []
y = []

num_of_books = 3
num_of_sentences = 1000

for i in range(num_of_books):
  
  sentences = gutenberg.sents(books[i])
  x.extend(sentences[3:num_of_sentences+3])
  
  for j in range(num_of_sentences):
    y.append(i)

"""# Dense Model"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

"""Tokenize the documents, create a word index (word -> number)."""

max_len = 20
# num_words = 7000
num_words = 5000

# Fit the tokenizer on the training data
t = Tokenizer(num_words=num_words)
t.fit_on_texts(x_train)

"""Here's how we vectorize a document."""

x_train_vectorized = t.texts_to_sequences(x_train)
x_test_vectorized = t.texts_to_sequences(x_test)

"""Apply padding if necessary."""

x_train_padded = pad_sequences(x_train_vectorized, maxlen=max_len, padding='post')
x_test_padded = pad_sequences(x_test_vectorized, maxlen=max_len, padding='post')

"""We will save the word index in metadata. Later, we'll use it to convert words typed in the browser to numbers for prediction."""

metadata = {
  'word_index': t.word_index,
  'max_len': max_len,
  'vocabulary_size': num_words,
}

"""Define a model."""

embedding_size = 8
n_classes = 3
epochs = 20

model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(max_len,)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""Prepare some training data."""

model_history = model.fit(x_train_padded, y_train, epochs=epochs, validation_split=0.2)

"""Demo using the model to make predictions."""

loss, accuracy = model.evaluate(x_test_padded, y_test)
print('Dense Model Test Loss: {0:.4f}'.format(loss))
print('Dense Model Test Accuracy: {0:.3f}'.format(accuracy))

def plot_history(histories, key):
  plt.figure(figsize=(20,10))
    
  for name, history in histories:
    val = plt.plot([x+1 for x in history.epoch], history.history['val_'+key], '--', label='Val '+name.title())
    plt.plot([x+1 for x in history.epoch], history.history[key], color=val[0].get_color(), label='Train '+name.title())

  plt.xlabel('Epochs')
  plt.xticks([x+1 for x in history.epoch])
  plt.ylabel(key.replace('_',' ').title())
  plt.title('Plot of '+ name.title() +' at different Epochs', fontsize=18)
  plt.legend()

plot_history([('Loss', model_history)],  key='loss')

plot_history([('Accuracy', model_history)],  key='acc')

"""# LSTM Model"""

embedding_size = 128
n_classes = 3
epochs = 20

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(max_len,)))
lstm_model.add(keras.layers.LSTM(16, return_sequences=True))
lstm_model.add(keras.layers.LSTM(16))
lstm_model.add(keras.layers.Dense(128, activation='relu'))
lstm_model.add(keras.layers.Dense(3, activation='softmax'))
lstm_model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.summary()

lstm_model_history = lstm_model.fit(x_train_padded, y_train, epochs=epochs, validation_split=0.2)

loss, accuracy = lstm_model.evaluate(x_test_padded, y_test)
print('LSTM Model Test Loss: {0:.4f}'.format(loss))
print('LSTM Model Test Accuracy: {0:.3f}'.format(accuracy))

plot_history([('LSTM Loss', lstm_model_history)],  key='loss')

plot_history([('LSTM Accuracy', lstm_model_history)],  key='acc')

"""# Deploy In Browser

** LSTM model doesn't show improvement in test accuracy, so saving DENSE model for deploying in browser.**


*   DENSE Model Test Accuracy ~ 86%
*   LSTM   Model Test Accuracy ~ 84.8%
"""

import json
import tensorflowjs as tfjs

metadata_json_path = os.path.join(MODEL_DIR, 'metadata.json')
json.dump(metadata, open(metadata_json_path, 'wt'))
tfjs.converters.save_keras_model(model, MODEL_DIR)
print('\nSaved model artifcats in directory: %s' % MODEL_DIR)

"""Write an index.html and an index.js file configured to load our model."""

index_html = """
<!doctype html>

<body>
  <style>
    #textfield {
      font-size: 120%;
      width: 60%;
      height: 200px;
    }
  </style>
  <h1>
    Title
  </h1>
  <hr>
  <div class="create-model">
    <button id="load-model" style="display:none">Load model</button>
  </div>
  <div>
    <div>
      <span>Vocabulary size: </span>
      <span id="vocabularySize"></span>
    </div>
    <div>
      <span>Max length: </span>
      <span id="maxLen"></span>
    </div>
  </div>
  <hr>
  <div>
    <select id="example-select" class="form-control">
      <option value="example1">Emma</option>
      <option value="example2">The Wisdom of Father Brown</option>
      <option value="example3">Paradise Lost</option>
    </select>
  </div>
  <div>
    <textarea id="text-entry"></textarea>
  </div>
  <hr>
  <div>
    <span id="status">Standing by.</span>
  </div>

  <script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js'></script>
  <script src='index.js'></script>
</body>
"""

index_js = """
const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'Emma Woodhouse , handsome , clever , and rich , with a comfortable home and happy disposition , seemed to unite some of the best blessings of existence ; and had lived nearly twenty - one years in the world with very little to distress or vex her .',
  'example2':
      'Poets and priests , if you will pardon my simplicity of speech , never have any money .',
  'example3':
      'Thus repulsed , our final hope Is flat despair : we must exasperate Th Almighty Victor to spend all his rage ; And that must end us ; that must be our cure -- To be no more .'      
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + " ->  " + result.score[x].toFixed(3) + ", "
  }
  //console.log(score_string);
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, i);
      //console.log(word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    //console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();
"""

with open('index.html','w') as f:
  f.write(index_html)
  
with open('index.js','w') as f:
  f.write(index_js)

!ls -ltr

"""Commit and push everything. Note: we're storing large binary files in GitHub, this isn't ideal (if you want to deploy a model down the road, better to host it in a cloud storage bucket)."""

!git add . 
!git commit -m "colab -> github"
!git push https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{USER_NAME}.github.io/ master

"""All done! Hopefully everything worked. You may need to wait a few moments for the changes to appear in your site. If not working, check the JavaScript console for errors (in Chrome: View -> Developer -> JavaScript Console)."""

print("Now, visit https://%s.github.io/%s/" % (USER_NAME, SITE_NAME))

"""If you are debugging and Chrome is failing to pick up your changes, though you've verified they're present in your GitHub repo, see the second answer to: https://superuser.com/questions/89809/how-to-force-refresh-without-cache-in-google-chrome

## GITHUB PAGES REPO:

https://github.com/{USER_NAME}/{USER_NAME}.github.io
"""
