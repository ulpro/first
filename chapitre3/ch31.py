import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
fashion_mist_data=keras.datasets.fashion_mnist
(all_x_train,all_y_train),(x_test,y_test)=fashion_mist_data.load_data()#permet de partionner les données
#la conversion
all_x_train=all_x_train.astype('float32')
x_test=x_test.astype('float32')
x_validation, x_train = all_x_train[:5000] / 255.0, all_x_train[5000:] / 255.0
y_validation, y_train = all_y_train[:5000], all_y_train[5000:]
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


#reseau de neutrone type mlp
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))#ajoute la premier couche aec un type precis
model.add(keras.layers.Dense(300,activation="relu"))#ajoute des couches avec la fonction d'activation relu
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(300,activation="softmax"))
model.summary()
hidden_2=model.get_layer("dense")
#accés aux poids
weights,blases=hidden_2.get_weights()
print("les poids:\n")
print(weights)
print(weights.shape)
print("les biais:\n")
print(blases)
print(blases.shape)