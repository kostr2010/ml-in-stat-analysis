#!/usr/local/bin/python3
from xmlrpc.client import boolean
import xgboost as xgb
import pandas
import sys
import re
import csv
import os
import numpy as np

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

print(train)
print(test)

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
