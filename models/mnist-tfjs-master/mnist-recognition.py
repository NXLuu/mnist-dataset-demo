import keras
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,RMSprop #Stochastic gradient descent optimizer.5y 
import tensorflowjs as tfjs

batch_size = 128
#10 numbers 0 to 9
num_classes = 10
#iterations for training with the training set. 
epochs = 30

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_test[0])
# quit()


#Convert the image pixils 28X28 to a single vector 784 so a training set 
#becomes a matrix. This is using numpy.reshape
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#Casting the number into float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('The first label from the traing set: ', y_train[0])

#Compress the greyscale level from 0-225 to 0-1
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test[0].shape, 'test samples')

quit()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('The first label in the training set is converted to ', y_train[0])

#Create a model which contains mutliple layers
model = Sequential()

#Add a layer type Dense with 512 output units for the hidden layer
#Because this is the input layer, we need to tell Keras what 
#the input data looks like in dimension 
#in this case, it is just a single dimension array with 784 units mapped to all 
#pixils in a 28X28 greyscale
model.add(Dense(512, activation='relu', input_shape=(784,)))

#According to the doc, dropout is used for preventing overfitting so it is 
#a regularisation process. It is easier in Keras than in Matlab
model.add(Dropout(0.2))

#Sigmoid function is used here, but it is said to use Relu function to have a 
#better performance. Sigmoid is a bit classic and old school feels like. 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
# Setting up the model for traing by defining the cost function which for is the loss param
# optimiser which is how we use to find the minmal of the cost function 
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])#It looks like accuracy is the one we normally use

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
#Like what I learnt from the course, we use training set and test set for training and 
#evaluating the performance 

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Saving trained model...')
model.save('mnist_model.h5')

print('Saving keras model to json...')
tfjs.converters.save_keras_model(model, './')