# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:18:17 2021




import tensorflow as tf

fashion = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels) = fashion.load_data()

training_images=training_images.reshape(60000,28,28,1)
training_images=training_images/255.0
test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()
model.fit(training_images,training_labels,epochs=5)
test_loss=model.evaluate(test_images,test_labels)

#VISUALIZATION

import matplotlib.pyplot as plt

f,axarr=plt.subplots(3,4)
first_image=0
second_image=9
third_image=32
conv_number=1

from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers]

activation_model = tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)

for x in range(0,4):
    f1=activation_model.predict(test_images[first_image].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,convolution_number],cmap='inferno')
    axarr[0,x].grid(False)
    f2=activation_model.predict(test_images[second_image].reshape(1,28,28,1))[x]
    axarr[1,x].imshow(f2[0,:,:,convolution_number],cmap='inferno')
    axarr[1,x].grid(False)
    f3=activation_model.predict(test_images[third_image].reshape(1,28,28,1))[x]
    axarr[2,x].imshow(f3[0,:,:,convolution_number],cmap='inferno')
    axarr[2,x].grid(False)
    



