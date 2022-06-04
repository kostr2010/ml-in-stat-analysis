#!/usr/local/bin/python3
from base64 import encode
from xmlrpc.client import boolean
from torch import float32, float64
import xgboost as xgb
import pandas
import sys
import re
import csv
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing

TRAIN_TEST_RATIO = 0.7

###############################
# USE:
# python3 tree.py dataset.csv [HYPERPARAMS]
###############################

assert(len(sys.argv) == 2)

data_csv = pandas.read_csv(sys.argv[1], header=None)
temp_cols = data_csv.columns.tolist()
new_cols = temp_cols[1:-1] + temp_cols[:1]
data_csv = data_csv[new_cols]

cwe_list = list(set(data_csv[1].tolist()))

train = pandas.DataFrame()
test = pandas.DataFrame()

for cwe in cwe_list:
    mask = (data_csv[1] == cwe)
    data_cwe = data_csv[mask]

    data_cwe_t = data_cwe[data_cwe[0] == True]
    data_cwe_f = data_cwe[data_cwe[0] == False]

    data_cwe_t_train = data_cwe_t.sample(frac=TRAIN_TEST_RATIO)
    data_cwe_t_test = data_cwe_t.drop(data_cwe_t_train.index)

    data_cwe_f_train = data_cwe_f.sample(frac=TRAIN_TEST_RATIO)
    data_cwe_f_test = data_cwe_f.drop(data_cwe_f_train.index)

    train = pandas.concat(
        [train, data_cwe_t_train, data_cwe_f_train], ignore_index=True)
    test = pandas.concat(
        [test, data_cwe_t_test, data_cwe_f_test], ignore_index=True)

train = train.fillna(0)
test = test.fillna(0)

X_train = train.to_numpy()[:, :-1]
Y_train = train.iloc[:, -1:].transpose().to_numpy()[0]

X_test = test.to_numpy()[:, :-1]
Y_test = test.iloc[:, -1:].transpose().to_numpy()[0]

# print(X_train)
# print(Y_train)

# print(X_test)
# print(Y_test)
Y_train_enc = []
Y_test_enc = []
for i in range(len(Y_train)):
    if Y_train[i]:
        Y_train_enc.append(1)
    else:
        Y_train_enc.append(0)

for i in range(len(Y_test)):
    if Y_test[i]:
        Y_test_enc.append(1)
    else:
        Y_test_enc.append(0)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train_enc)
Y_test_pred = clf.predict(X_test)

print("Accuracy Score: {}".format(accuracy_score(Y_test_enc, Y_test_pred)))

tree.plot_tree(clf)

# print(train.columns[len(train.columns) - 1])
# clf.fit(train.to_numpy()[0:-1], train.to_numpy()[-1])
# clf.predict(test.to_numpy()[:-1])

# print(train.to_numpy()[0:-1])
# print(test.to_numpy()[:-1])

# import xgboost as xgb
# # read in data
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)
