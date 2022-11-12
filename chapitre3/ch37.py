import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


housing = fetch_california_housing()
all_x_train, x_test, all_y_train, y_test = train_test_split(housing.data, housing.target)

'''
print(f"all_x_train.shape = {all_x_train.shape}" )
print(f"all_x_train.dtype = {all_x_train.dtype}" )
print(f"type(all_x_train) = {type(all_x_train)}" )
print(f"Features = {all_x_train[0]} ; variable cible = {all_y_train[0]}" )
'''
'''
#https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
'''
 
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)
 
''' 
print(f"Le MAX - MIN de y_train = [{np.max(y_train)}-{np.min(y_train)}]")
print(f"Le MAX - MIN de y_validation = [{np.max(y_validation)}-{np.min(y_train)}]")
print(f"Le MAX - MIN de y_test = [{np.max(y_test)}-{np.min(y_test)}]")
'''  
 
 
model = keras.models.Sequential() 
model.add(keras.layers.Dense(30, activation="relu", input_shape=x_train_scaled.shape[1:]))   
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dense(1))



#model.summary()
 

 
model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
ressults = model.fit(x_train_scaled, y_train, epochs=100, \
                     validation_data=(x_validation_scaled, y_validation))

mae_test = model.evaluate(x_test_scaled, y_test)

 


