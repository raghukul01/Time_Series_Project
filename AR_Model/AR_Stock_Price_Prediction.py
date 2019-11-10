# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
dataset = pd.read_csv('../Data/apple_share_price.csv', usecols=[1,2,3,4])
# print(dataset.head())
# print(dataset.index[::-1])
dataset = dataset.reindex(index = dataset.index[::-1])
# print(dataset.head())


# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
# OHLC_avg = dataset
# print(OHLC_avg)

# HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
# print(HLC_avg)
# close_val = dataset[['Close']]
# print(close_val.head())
# exit(0)


# PLOTTING ALL INDICATORS IN ONE PLOT
# plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
# plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
# plt.plot(obs, close_val, 'g', label = 'Closing price')
# plt.legend(loc = 'upper right')
# plt.show()

#Augmented Dickey Fuller Test for Stationarity

fig,axes = plt.subplots(2,2,sharex=True)
OHLC_avg
axes[0, 0].plot(obs,OHLC_avg); axes[0, 0].set_title('Original Series')
plot_acf(OHLC_avg, ax=axes[0,1],lags=100)
# result = adfuller(OHLC_avg.values)
# print('ADF Statistics: %f' % result[0])
# print('p-value: %f' % result[1])

b = obs[1:]

axes[1, 0].plot(obs,OHLC_avg.diff()); axes[1, 0].set_title('Diff Series')
plot_acf(OHLC_avg.diff(), ax=axes[1,1], lags=100)
plt.show()

# First order difference
# print(type(OHLC_avg))
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
diff = []
for i in range(OHLC_avg.shape[0]-1):
    diff.append(OHLC_avg[i+1]-OHLC_avg[i])

diff = np.asarray(diff)
diff = diff.reshape((diff.shape[0],))
# print(diff.shape)
result = adfuller(diff)
print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])
exit(0)

# plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(b, diff, 'b', label = 'diff')
plt.show()
exit(0)
# print(OHLC_avg)
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)
# print(OHLC_avg)

