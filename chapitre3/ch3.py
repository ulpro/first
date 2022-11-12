import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
fashion_mist_data = keras.datasets.fashion_mnist #permet de recuperer le dataset en ligne
(all_x_train,all_y_train),(x_test,y_test)=fashion_mist_data.load_data()#permet de partionner les données
#la conversion
all_x_train=all_x_train.astype('float32')
x_test=x_test.astype('float32')
'''print(f'all_x_train = {all_x_train.shape}')
print(f'all_x_train[0].shape = {all_x_train[0].shape}')
print(f'all_x_train[0] = {all_x_train[0].dtype}')'''
x_validation, x_train = all_x_train[:5000] / 255.0, all_x_train[5000:] / 255.0
y_validation, y_train = all_y_train[:5000], all_y_train[5000:]

print(f'x_train = {x_train.shape}')
print(f'x_train[0].shape = {x_train[0].shape}')
print(f'x_train[0] = {x_train[0].dtype}')
fashion_mist_class_names=["T-shirt/top","trouser","pullover","Dress","coat","sandal","shirt","Sneaker","bag","Ankle boot"]
for cls in range(10):
    print(cls,":",fashion_mist_class_names[y_train[cls]])
for i in range(5):
    my_img=x_train[i]
    my_img_class=y_train[i]
    my_img_class_name=fashion_mist_class_names[my_img_class]
    plt.imshow(my_img)
    plt.title(my_img_class_name)
    plt.show()
    