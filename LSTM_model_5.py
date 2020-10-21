
# 6/15:::creat ETF59_GRU_test_1.h5 which is better than ETF59_GRU.h5 model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras import backend as K
from sklearn import preprocessing
np.random.seed(2017)
num_of_features = 5
features_mean = 0
features_std = 0

def read_and_preprocess(path_to_dataset=None, sequence_length=20, ratio = 1):
    global features_mean
    global features_std
    max_values = ratio*2656
    # buf = [21,23,24,25,26]
    buf = [21,23,24,25,26]

    features = []
    for i in range(len(buf)):
        features.append([])
    with open(path_to_dataset, mode='r', encoding='ascii', errors='ignore') as f:
        data = csv.reader(f, delimiter=",")
        next(data)
        nb_of_values = 0
        for line in data:
            for i in range(len(buf)):
                try:
                    features[i].append(float(line[buf[i]]))
                except ValueError:
                    pass
            nb_of_values += 1
            if nb_of_values >= max_values:
                break
    print ("Data loaded from csv. Formatting...")
    features_normalized = []
    for i in range(max_values):
        features_normalized.append([])
    for i in range(max_values):
        # print(i)
        for j in range(len(features)):
            # print(features[j][i])
            features_normalized[i].append(features[j][i])
    features_normalized = np.array(features_normalized)
    features_mean = features_normalized.mean(axis=0)
    features_std = features_normalized.std(axis=0)
    features_normalized = preprocessing.scale(features_normalized)
    for i in range(max_values):
        for j in range(num_of_features):
            features[j][i] = features_normalized[i][j]
    print("Data being normalized...")
    result_features = []
    for i in range(len(buf)):
        result_features.append([])
    for i in range(len(buf)):
        for index in range(0, len(features[0]) - sequence_length, 5):
            result_features[i].append(features[i][index: index + sequence_length])
    result_features_combine = []
    for i in range(len(result_features[0])):
        result_features_combine.append([])

    for i in range(len(result_features[0])):
        for j in range(len(result_features)):
            result_features_combine[i].append(result_features[j][i])
    result_features_combine = np.array(result_features_combine)
    
    row = int(round(0.99 * result_features_combine.shape[0]))
    
    train = result_features_combine[:row, :]
    test = result_features_combine[row:, :]
    np.random.shuffle(train)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(train.shape[0]):
        X_train.append([])    
    for i in range(test.shape[0]):
        X_test.append([])
    
    for i in range(train.shape[0]):
        for j in range(len(result_features_combine[0])):
            temp_train = train[i][j]
            X_train[i].append(temp_train[:sequence_length - 5])
            if j == 0:
                temp_y_train = train[i][j]
                y_train.append(temp_y_train[15:])
    
    for i in range(test.shape[0]):
        for j in range(len(result_features_combine[0])):
            temp_test = test[i][j]
            X_test[i].append(temp_test[:sequence_length - 5])
            if j == 0:
                temp_y_test = test[i][j]
                y_test.append(temp_y_test[15:])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train_reshape_temp = []
    for i in range(X_train.shape[1]):
        X_train_reshape_temp.append([])
    
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            temp_train_reshape = X_train[i][j]
            X_train_reshape_temp[j].append(temp_train_reshape)
    for i in range(X_train.shape[1]):
        X_train_reshape_temp[i] = np.array(X_train_reshape_temp[i])
        X_train_reshape_temp[i] = np.reshape(X_train_reshape_temp[i], (X_train_reshape_temp[i].shape[0], X_train_reshape_temp[i].shape[1], 1))
    X_train_reshape = []
    for i in range(X_train.shape[0]):
        X_train_reshape.append([])
        for j in range(15):
            X_train_reshape[i].append([])
    for i in range(X_train.shape[1]):
        for j in range(X_train.shape[0]):
            for k in range(15):
                X_train_reshape[j][k].append(X_train_reshape_temp[i][j][k][0])
    X_train_reshape = np.array(X_train_reshape)

    X_test_reshape_temp = []
    for i in range(X_test.shape[1]):
        X_test_reshape_temp.append([])
    
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            temp_test_reshape = X_test[i][j]
            X_test_reshape_temp[j].append(temp_test_reshape)
    for i in range(X_test.shape[1]):
        X_test_reshape_temp[i] = np.array(X_test_reshape_temp[i])
        X_test_reshape_temp[i] = np.reshape(X_test_reshape_temp[i], (X_test_reshape_temp[i].shape[0], X_test_reshape_temp[i].shape[1], 1))
    X_test_reshape = []
    for i in range(X_test.shape[0]):
        X_test_reshape.append([])
        for j in range(15):
            X_test_reshape[i].append([])
    for i in range(X_test.shape[1]):
        for j in range(X_test.shape[0]):
            for k in range(15):
                X_test_reshape[j][k].append(X_test_reshape_temp[i][j][k][0])
    X_test_reshape = np.array(X_test_reshape)    
    print(X_train_reshape.shape)
    print(X_test_reshape.shape)
    print(y_train.shape)
    print(y_test.shape)
    return [X_train_reshape, y_train, X_test_reshape, y_test]

def build_model():
    model = Sequential()
    neurons = [num_of_features, 50, 100, 150, 200, 250, 5]

    model.add(GRU(
        neurons[1],
        input_shape=(15, num_of_features),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(
        neurons[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(Dense(neurons[4],kernel_initializer="uniform",activation='relu'))
    model.add(Dense(neurons[6],kernel_initializer="uniform",activation='relu'))
    # model.add(Dense(neurons[6],kernel_initializer="uniform",activation='linear'))  
    start = time.time()
    model.compile(loss='mse', optimizer="adam", metrics=['mse'])
    print ("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):
    global features_mean
    global features_std
    global_start_time = time.time()
    epochs = 400
    ratio = 1
    sequence_length = 20  
    path_to_dataset = 'merge_with_futures/0050merge_plus_futures.csv'

    if data is None:
        print ('Loading data... ')
        X_train, y_train, X_test, y_test = read_and_preprocess(
            path_to_dataset, sequence_length, ratio)
        print ('\nData Loaded. Compiling...\n')

    # model = load_model('model_with_futures_better/ETF50_GRU_test_2.h5')
    if model is None:
        model = build_model()
        print('model built')
        history = model.fit(
            X_train, y_train,
            batch_size=100, epochs=epochs, validation_split=0.25)
        fig0 = plt.figure(0)
        ax = fig0.add_subplot(111)
        ax.plot(history.history['mean_squared_error'], label="mean_squared_error")
        ax.legend(loc='upper left')
        plt.plot(history.history['val_mean_squared_error'], label="val_mean_squared_error")
        plt.legend(loc='upper left')
        plt.show()

        model.save('model_with_futures_better/ETF50_GRU_test_2.h5')
    try:
        print('Get predicted......')
        predicted = model.predict(X_test)
        for i in range(y_test.shape[0]):
            for j in range(y_test.shape[1]):
                y_test[i][j] = y_test[i][j] * features_std[0]
                y_test[i][j] += features_mean[0]
                predicted[i][j] = predicted[i][j] * features_std[0]
                predicted[i][j] += features_mean[0]
        print_start = predicted.shape[0]-5
        print('showing last 5 arrays\npredicted: ')
        for i in range(print_start, predicted.shape[0]):
            print(predicted[i])
        print('real: ')
        for i in range(print_start, predicted.shape[0]):
            print(y_test[i])
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_true, 0
    try:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(111)
        y_test_flatten = y_test.flatten()
        ax.plot(y_test_flatten,label="Real")
        ax.legend(loc='upper left')
        predicted_flatten = predicted.flatten()
        plt.plot(predicted_flatten,label="Prediction")
        plt.legend(loc='upper left')
        plt.show()
    except Exception as e:
        print (str(e))
    print ('Training duration (s) : ', time.time() - global_start_time)
    return model, y_test, predicted
if __name__ == '__main__':
    run_network()

