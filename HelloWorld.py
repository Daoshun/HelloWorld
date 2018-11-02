import tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # x_train = x_train
    # x_test = x_test
    #
    x_train = x_train / 255
    x_test = x_test / 255
    # x_test=np.random.normal(x_test)#增加噪声
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
model = Sequential()
model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
# model.add(Dropout(0.5))#dropout 降低train的正确率，但是会提高test的正确率
model.add(Dense(units=500, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20)
result = model.evaluate(x_train, y_train)
print('\nTrain Acc', result[1])
result = model.evaluate(x_test, y_test)
print('\nTest Acc', result[1])
