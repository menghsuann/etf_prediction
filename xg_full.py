from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import h5py
import requests
import os
import numpy as np
import xgboost as xgb
import sys
from xgboost import cv, DMatrix
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot as plt

path_of_taetfp = 'input_data/taetfp.csv'
df = pd.read_csv(path_of_taetfp, header=0, engine='python', thousands=r',' , encoding='cp950')
df = df.rename(index=str, columns={"代碼": "symbol", "日期": "date","中文簡稱":"chinese_name","開盤價(元)":"open",\
                                   "最高價(元)":"high","最低價(元)":"low","收盤價(元)":"close","成交張數(張)":"volume","漲跌值":"change"})


df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
df = df.drop('chinese_name', axis = 1)
df = df.drop('change', axis = 1)

path_of_allData = 'input_data/allData.csv'
three_buy_sell = pd.read_csv(path_of_allData, header=0, engine='python', thousands=r',' , encoding='cp950')
three_buy_sell = three_buy_sell.rename(index=str, columns={"日期": "date", "自營商買進金額":"self_buy",\
                                                           "自營商賣出金額":"self_sell",\
                                                           "自營商買賣總額":"self_total","自營商買賣差額":"self_diff",\
                                                           "投信買進金額":"trust_buy","投信賣出金額":"trust_sell",\
                                                           "投信買賣總額":"trust_total","投信買賣差額":"trust_diff",\
                                                           "外資及陸資買進金額":"out_buy","外資及陸資賣出金額":"out_sell",\
                                                           "外資及陸資買賣總額":"out_total","外資及陸資買賣差額":"out_diff"})

three_buy_sell['date'] = three_buy_sell['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
#three_buy_sell.head()



path_of_options = 'input_data/options.csv'
options = pd.read_csv(path_of_options, header=0, engine='python', thousands=r',')
options['date'] = options['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y/%m/%d'))

#options.head()

path_of_futures = 'input_data/futures.csv'
futures = pd.read_csv(path_of_futures, header=0, engine='python', thousands=r',')
futures['date'] = futures['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y/%m/%d'))

#futures.head()



path_of_putcall = 'input_data/putcall.csv'
putcall = pd.read_csv(path_of_putcall, header=0, engine='python', thousands=r',' , encoding='cp950')
putcall = putcall.rename(index=str, columns={"日期": "date", \
                                            "賣權成交量":"sell_vol","買權成交量":"buy_vol", \
                                            "買賣權成交量比率%":"buy_sell_rate","買賣權未平倉量比率%":"no_flat_rate"})
putcall['date'] = putcall['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y/%m/%d'))

#putcall.head()


all_merge = pd.merge(putcall , futures)
all_merge = pd.merge(all_merge, options)
all_merge = pd.merge(all_merge, three_buy_sell)
all_merge = pd.merge(all_merge, df)
all_merge = all_merge.drop('date' , axis=1)
all_merge = all_merge.fillna(0)
# print(all_merge.head())


sz = all_merge.shape
# print(sz[1])#(19053,76)


data = all_merge.values.tolist()
for cnt in range(len(data)):
	if(str(data[cnt][75]) == '-1.0'):
		data[cnt][75] = int(0)
	elif(str(data[cnt][75]) == '0.0'):
		data[cnt][75] = int(1)
	elif(str(data[cnt][75]) == '1.0'):
		data[cnt][75] = int(2)

train = []
for cnt in range(int(sz[0]*0.9)):
	train.append(data[cnt])

test = []
for cnt in range(int(sz[0]*0.9) , len(data) , 1):
	test.append(data[cnt])


#train_X = train[:, :74]
train_X = []
for cnt in range(len(train)):
	train_X.append([])
	for cnt2 in range(75):
		train_X[cnt].append(train[cnt][cnt2])

train_Y =[]
for cnt in range(len(train)):
	train_Y.append(train[cnt][75])


# test_X = test[:, :74]
test_X = []
for cnt in range(len(test)):
	test_X.append([])
	for cnt2 in range(75):
		test_X[cnt].append(test[cnt][cnt2])

test_Y =[]
for cnt in range(len(test)):
	test_Y.append(test[cnt][75])

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / len(test_Y)
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
'''
bst.dump_model('dump.raw.txt')
bst.dump_model('dump.raw.txt','featmap.txt')
sys.exit()
'''
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(len(test_Y), 3)
print(pred_prob)
print(len(pred_prob))
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / len(test_Y)
print('Test error using softprob = {}'.format(error_rate))
plot_importance(bst)
plt.show()
