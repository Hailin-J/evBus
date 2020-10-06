import numpy
import matplotlib.pyplot as plt
import pandas
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import *
from keras.layers.wrappers import *
from keras.optimizers import RMSprop

dataframe = pandas.read_csv('/user/arch/jin/Evbus/evBus/1589.csv', usecols=[1, 2], engine='python', skipfooter=1)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataX.append(xset)
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i+look_back, j]
            xset.append(a)
        dataY.append(xset)
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 30
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainY = trainY.reshape((trainY.shape[0], trainY.shape[1], 1))
testY = testY.reshape((testY.shape[0], testY.shape[1], 1))

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))
trainY = numpy.reshape(trainY, (trainY.shape[0], trainY.shape[2], trainY.shape[1]))
testY = numpy.reshape(testY, (testY.shape[0], testY.shape[2], testY.shape[1]))

trainX = trainX.reshape((trainX.shape[0], 3, 10, 2, 1))
testX = testX.reshape((testX.shape[0], 3, 10, 2, 1))
trainY = trainY.reshape((trainY.shape[0], 1, trainY.shape[1], 2, 1))
testY = testY.reshape((testY.shape[0], 1, testY.shape[1], 2, 1))

n_timesteps = trainX.shape[1]
output_timesteps = 1
model = Sequential()
model.add(BatchNormalization(name='batch_norm_0', input_shape=(n_timesteps, trainX.shape[2], trainX.shape[3], 1)))
model.add(ConvLSTM2D(name='conv_lstm_1',
                     filters=64, kernel_size=(10, 1),
                     padding='same',
                     return_sequences=True))

model.add(Dropout(0.21, name='dropout_1'))
model.add(BatchNormalization(name='batch_norm_1'))

model.add(ConvLSTM2D(name='conv_lstm_2',
                     filters=64, kernel_size=(5, 1),
                     padding='same',
                     return_sequences=False))

model.add(Dropout(0.20, name='dropout_2'))
model.add(BatchNormalization(name='batch_norm_2'))

model.add(Flatten())
model.add(RepeatVector(output_timesteps))
model.add(Reshape((output_timesteps, 1, 2, 640)))

model.add(ConvLSTM2D(name='conv_lstm_3',
                     filters=64, kernel_size=(10, 1),
                     padding='same',
                     return_sequences=True))

model.add(Dropout(0.1, name='dropout_3'))
model.add(BatchNormalization(name='batch_norm_3'))

model.add(ConvLSTM2D(name='conv_lstm_4',
                     filters=64, kernel_size=(5, 1),
                     padding='same',
                     return_sequences=True))

model.add(TimeDistributed(Dense(units=1, name='dense_1', activation='relu')))
# model.add(Dense(units=1, name = 'dense_2'))

# optimizer = RMSprop() #lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)
# model.compile(loss = "mse", optimizer = optimizer)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape'])
model.fit(trainX, trainY, epochs=40, batch_size=16, verbose=1)

model.save('/user/arch/jin/Evbus/ConvLSTM.h5')
