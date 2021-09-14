# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:04:56 2021
IMDB DATASET TEXT CLASSIFICATION
@author: Denno
"""

import tensorflow as tf
import tensorflow_datasets as tfds

imdb,info = tfds.load('imdb_reviews',with_info=True,as_supervised=True)

import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
    
    
for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())
    
train_labels_final = np.array(training_labels)
test_labels_final = np.array(testing_labels)

#PARAMETRIZATION

vocab_size=10000
embedding_dim=16
max_length=120
trunc_type='post'
oov_tok='<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(test_sequences,maxlen=max_length)    

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ''.join([reverse_word_index.get(i,'?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

#BUILD THE MODEL

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs=50
history=model.fit(
    padded,
    train_labels_final,
    epochs=num_epochs,
    validation_data=(test_padded,test_labels_final)
    )

import matplotlib.pyplot as plt

def graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()
    
graphs(history,'accuracy')
graphs(history,'loss')
    


#LSTM MODEL

model2=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.summary()

history=model2.fit(
    padded,
    train_labels_final,
    epochs=num_epochs,
    validation_data=(test_padded,test_labels_final)
    )

def graph_plot(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()
    
graph_plot(history, 'accuracy')
graph_plot(history, 'loss')    


