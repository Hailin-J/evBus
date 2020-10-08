import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#データ読み込み
dataframe = pandas.read_csv('/user/arch/jin/Evbus/evBus/1589.csv', usecols=[1, 2], engine='python', skipfooter=1)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset_real = dataset[24270:]
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_real = scaler.fit_transform(dataset_real)

# split into train and test sets
train_size = int(len(dataset_real) * 0.67)
test_size = len(dataset_real) - train_size
train, test = dataset_real[0:train_size,:], dataset_real[train_size:len(dataset_real),:]

# convert an array of values into a dataset matrix
# if you give look_back 3, a part of the array will be like this: Jan, Feb, Mar
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_forward-1):
        xset, yset = [], []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            b = dataset[(i+look_back):(look_forward+look_back+i), j]
            xset.append(a)
            yset.append(b)
        dataX.append(xset)
        dataY.append(yset)
    return numpy.array(dataX), numpy.array(dataY)

# use 60 --> predict 15
look_back = 60
look_forward = 15

trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)

trainX_trans = trainX.transpose((0, 2, 1))
trainY_trans = trainY.transpose((0, 2, 1))
testX_trans = testX.transpose((0, 2, 1))
testY_trans = testY.transpose((0, 2, 1))

n_outtime = 15
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(trainX_trans.shape[1], trainX_trans.shape[2])))
model.add(RepeatVector(n_outtime))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(2)))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape'])
model.summary()

model.fit(trainX_trans, trainY_trans, epochs=70, batch_size=16, verbose=1)

model.save('/user/arch/jin/Evbus/LSTM_train70.h5')
