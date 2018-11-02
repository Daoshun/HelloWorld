#Handwritten digit recognition
import tensorflow
import keras

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_mnist_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]

    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)

    x_train = x_train/255
    x_test = x_test/255
    return (x_train,y_train),(x_test,y_test)
(x_train,y_train),(x_test,y_test) = load_mnist_data()
model = Sequential()
model.add(Convolution2D(255,3,3,input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(50,3,3))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(x_train,y_train,nb_epoch=20,batch_size=100)

result_train = model.evaluate(x_train,y_train)
print('\nTrain CNN Acc',result_train[1])
result_test = model.evaluate(x_test,y_test)
print('\nTest CNN Acc',result_test[1])
