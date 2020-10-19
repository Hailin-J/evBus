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
from keras.callbacks import CSVLogger, EarlyStopping

plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.labelcolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['ytick.color'] = 'k'
plt.rcParams['xtick.color'] = 'k'
plt.rcParams['grid.color'] = (.7, .7, .7, 0)
plt.rcParams['figure.figsize'] = (16, 10)
import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )
print('numpy ver.: ' + numpy.__version__)
print('pandas ver.: ' + pandas.__version__)
print('tensorflow ver.: ' + tf.__version__)
print('keras ver.: ' + keras.__version__)
#データ読み込み
dataframe = pandas.read_csv('/user/arch/jin/Evbus/evBus/1589-stop.csv', usecols=[1, 2], engine='python', skipfooter=1)
dataset = dataframe.values
dataset = dataset.astype('float32')
# print(len(dataset))

dataset_1 = dataset[25243:27804]
dataset_2 = dataset[31047:33493]
dataset_3 = dataset[38359:40785]
# print(len(dataset_1))
# print(len(dataset_2))
# print(len(dataset_3))


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_1 = scaler.fit_transform(dataset_1)
dataset_2 = scaler.fit_transform(dataset_2)
dataset_3 = scaler.fit_transform(dataset_3)

dataset3 = dataset_3[0:829]
test = dataset_3[829:]
# # split into train and test sets
# train_size = int(len(dataset_real) * 0.67)
# test_size = len(dataset_real) - train_size
# train, test = dataset_real[0:train_size,:], dataset_real[train_size:len(dataset_real),:]
# print(len(test))


def create_dataset(dataset, look_back=1, look_forward=1, t=1):
    dataX, dataY = [], []
    for i in range(0, (len(dataset)-look_back-look_forward-t-1), 1):
        xset, yset = [], []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            b = dataset[(i+look_back+t):(look_forward+look_back+t+i), j]
            xset.append(a)
            yset.append(b)
        dataX.append(xset)
        dataY.append(yset)
    return numpy.array(dataX), numpy.array(dataY)

# use look_back --> predict look_forward after t seconds
look_back = 60
look_forward = 1
t = 30

trainX_1, trainY_1 = create_dataset(dataset_1, look_back, look_forward, t)
trainX_2, trainY_2 = create_dataset(dataset_2, look_back, look_forward, t)
trainX_3, trainY_3 = create_dataset(dataset3, look_back, look_forward, t)

testX, testY = create_dataset(test, look_back, look_forward)

trainX_12 = numpy.concatenate((trainX_1, trainX_2), axis=0)
trainY_12 = numpy.concatenate((trainY_1, trainY_2), axis=0)
trainX = numpy.concatenate((trainX_12,trainX_3), axis=0)
trainY = numpy.concatenate((trainY_12,trainY_3), axis=0)

trainX_trans = trainX.transpose((0, 2, 1))
trainY_trans = trainY.transpose((0, 2, 1))
testX_trans = testX.transpose((0, 2, 1))
testY_trans = testY.transpose((0, 2, 1))

n_outtime = 1
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(trainX_trans.shape[1], trainX_trans.shape[2])))
model.add(RepeatVector(n_outtime))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(2)))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape'])
model.summary()

model.fit(trainX_trans, trainY_trans, epochs=70, batch_size=16, verbose=1)

model.save('/user/arch/jin/Evbus/LSTM-withStopInfo-ver01-Y-U.h5')
