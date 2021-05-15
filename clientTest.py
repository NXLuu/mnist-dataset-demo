import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow import keras
import cv2 as cv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)

model = keras.models.load_model('./model/simple.h5');
img = cv.imread('./mnist/validated/0-20170518022735.png.png')
img = cv.resize(img, (28, 28))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
matrix = np.array(img)
y_predict = model.predict(matrix.reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))
print(model)
plt.imshow(matrix);
# print(X_train[0])
cv.waitKey(0)
plt.show();