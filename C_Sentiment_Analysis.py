# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:19:41 2022

@author: User
"""
import os
import json
import pandas as pd
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Bidirectional,Embedding, Masking
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import TensorBoard
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
vocab_size = 1000
oov_token = 'OOV'
max_len = 333

#%% STATICS
SENTIMENT_OHE_PICKLE_PATH = os.path.join(os.getcwd(),'sentiment_ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_URL)

# Step 2) Data Inspection
df.head(10) # to visualise first 10 rows
df.tail(10) # to visualise last 10 rows
df.info()

df['category'].unique()
df['category'][5]
df['text'][5]

df.duplicated().sum() # to check for duplicated data
df[df.duplicated()] # to visualise the duplicated data

# Step 3) Data Cleaning
# to remove duplicated data
df = df.drop_duplicates() # to remove duplicates
print(df)

text = df['text'].values # Features : X
category = df['category'].values # Features : y

for index,rev in enumerate(text):
    # remove html tags
    # ?dont be greedy
    # * zero or more occurences
    # .Any character except new line (/n)    
    text[index] = re.sub('<.*?>',' ',rev) 

    # convert into lower case
    # remove numbers 
    # ^ means NOT
    text[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()

# Step 4) Features selection
# Nothing to select

#%% Data preprocessing
#           1) Convert into lower case
#           2) Tokenization 
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(text) # learn all of the words
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(text)

#           3) Padding & truncating
length_of_text = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_text)) # to get the number of max length for padding

padded_text = pad_sequences(train_sequences,maxlen=max_len,padding='post',truncating='post')


#           4) One Hot Encoding for the target

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))
# Need to save ohe model
with open(SENTIMENT_OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)

#               5) Train test split

X_train,X_test,y_train,y_test = train_test_split(padded_text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
#%% Model development
# Use LSTM layers, dropout, dense, input
# for bidirectional
embedding_dim = 128

model = Sequential()
model.add(Input(shape=(333))) #np.shape(X_train)[1:]
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(Masking(mask_value=0)) 
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(np.shape(category)[1], activation='softmax'))
model.summary()


plot_model(model)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# callbacks 
hist = model.fit(X_train,y_train,
                 epochs=100,
                 batch_size=128,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback])

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss') 
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation acc') 
plt.legend()
plt.show()

#%% model evaluation
y_true = y_test
y_pred = model.predict(X_test)

#%%
y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)

#%%
print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

#%% Model saving
model.save(MODEL_SAVE_PATH)

token_json = tokenizer.to_json()
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)                              

ONE_PATH = os.path.join(os.getcwd(),'one_path.pkl')
with open(ONE_PATH,'wb') as file:
    pickle.dump(ohe,file)