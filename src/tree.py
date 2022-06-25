#!/usr/local/bin/python3
import xgboost as xgb
import pandas
import sys
import re
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

TRAIN_TEST_RATIO = 0.8
TOKEN_WINDOW = 25

TOKENS = []
for i in range(2 * TOKEN_WINDOW):
    TOKENS.append("token[{}]".format(i - TOKEN_WINDOW))

METRICS_NAMES_UNEDITED = ["FILE_CNT", "FILE_SIZE", "FUNC_CNT", "KW_IF_CNT", "RAW_KW_IF_CNT", "KW_ELSE_CNT", "RAW_KW_ELSE_CNT", "KW_FOR_CNT", "RAW_KW_FOR_CNT", "KW_RETURN_CNT", "RAW_KW_RETURN_CNT", "KW_DO_CNT", "RAW_KW_DO_CNT", "KW_WHILE_CNT", "RAW_KW_WHILE_CNT", "KW_SWITCH_CNT", "RAW_KW_SWITCH_CNT", "KW_CASE_CNT", "RAW_KW_CASE_CNT", "KW_BREAK_CNT", "RAW_KW_BREAK_CNT", "KW_DEFAULT_CNT", "RAW_KW_DEFAULT_CNT", "KW_GOTO_CNT", "RAW_KW_GOTO_CNT", "KW_AUTO_CNT", "RAW_KW_AUTO_CNT", "KW_VOLATILE_CNT(local)", "KW_VOLATILE_CNT(cumulative)", "RAW_KW_VOLATILE_CNT(local)", "RAW_KW_VOLATILE_CNT(cumulative)", "KW_CONST_CNT(local)", "KW_CONST_CNT(cumulative)", "RAW_KW_CONST_CNT(local)", "RAW_KW_CONST_CNT(cumulative)", "KW_BODY_CONST_CNT", "RAW_KW_BODY_CONST_CNT", "KW_TYPEDEF_CNT(local)", "KW_TYPEDEF_CNT(cumulative)", "RAW_KW_TYPEDEF_CNT(local)", "RAW_KW_TYPEDEF_CNT(cumulative)", "KW_CONTINUE_CNT", "RAW_KW_CONTINUE_CNT", "KW_UNION_CNT(local)", "KW_UNION_CNT(cumulative)", "KW_BODY_UNION_CNT", "RAW_KW_UNION_CNT(local)", "RAW_KW_UNION_CNT(cumulative)", "RAW_KW_BODY_UNION_CNT", "KW_STRUCT_CNT(local)", "KW_STRUCT_CNT(cumulative)", "KW_BODY_STRUCT_CNT", "RAW_KW_STRUCT_CNT(local)", "RAW_KW_STRUCT_CNT(cumulative)", "RAW_KW_BODY_STRUCT_CNT", "KW_ENUM_CNT(local)", "KW_ENUM_CNT(cumulative)", "KW_BODY_ENUM_CNT", "RAW_KW_ENUM_CNT(local)", "RAW_KW_ENUM_CNT(cumulative)", "RAW_KW_BODY_ENUM_CNT", "KW_CHAR_CNT(local)", "KW_CHAR_CNT(cumulative)", "KW_BODY_CHAR_CNT", "RAW_KW_CHAR_CNT(local)", "RAW_KW_CHAR_CNT(cumulative)", "RAW_KW_BODY_CHAR_CNT", "KW_UNSIGNED_CNT(local)", "KW_UNSIGNED_CNT(cumulative)", "KW_BODY_UNSIGNED_CNT", "RAW_KW_UNSIGNED_CNT(local)", "RAW_KW_UNSIGNED_CNT(cumulative)", "RAW_KW_BODY_UNSIGNED_CNT", "KW_SIGNED_CNT(local)", "KW_SIGNED_CNT(cumulative)", "KW_BODY_SIGNED_CNT", "RAW_KW_SIGNED_CNT(local)", "RAW_KW_SIGNED_CNT(cumulative)", "RAW_KW_BODY_SIGNED_CNT", "KW_DOUBLE_CNT(local)", "KW_DOUBLE_CNT(cumulative)", "KW_BODY_DOUBLE_CNT", "RAW_KW_DOUBLE_CNT(local)", "RAW_KW_DOUBLE_CNT(cumulative)", "RAW_KW_BODY_DOUBLE_CNT", "KW_FLOAT_CNT(local)", "KW_FLOAT_CNT(cumulative)", "KW_BODY_FLOAT_CNT", "RAW_KW_FLOAT_CNT(local)", "RAW_KW_FLOAT_CNT(cumulative)", "RAW_KW_BODY_FLOAT_CNT", "KW_INT_CNT(local)", "KW_INT_CNT(cumulative)", "RAW_KW_INT_CNT(local)", "RAW_KW_INT_CNT(cumulative)", "KW_BODY_INT_CNT", "RAW_KW_BODY_INT_CNT", "KW_LONG_CNT(local)", "KW_LONG_CNT(cumulative)", "KW_BODY_LONG_CNT", "RAW_KW_LONG_CNT(local)", "RAW_KW_LONG_CNT(cumulative)", "RAW_KW_BODY_LONG_CNT", "KW_SHORT_CNT(local)", "KW_SHORT_CNT(cumulative)", "KW_BODY_SHORT_CNT", "RAW_KW_SHORT_CNT(local)", "RAW_KW_SHORT_CNT(cumulative)", "RAW_KW_BODY_SHORT_CNT", "KW_STATIC_CNT(local)", "KW_STATIC_CNT(cumulative)", "KW_BODY_STATIC_CNT", "RAW_KW_STATIC_CNT(local)", "RAW_KW_STATIC_CNT(cumulative)", "RAW_KW_BODY_STATIC_CNT", "KW_EXTERN_CNT(local)", "KW_EXTERN_CNT(cumulative)", "RAW_KW_EXTERN_CNT(local)", "RAW_KW_EXTERN_CNT(cumulative)", "KW_REGISTER_CNT(local)", "KW_REGISTER_CNT(cumulative)", "RAW_KW_REGISTER_CNT(local)", "RAW_KW_REGISTER_CNT(cumulative)", "KW_VOID_CNT(local)", "KW_VOID_CNT(cumulative)", "KW_BODY_VOID_CNT", "RAW_KW_VOID_CNT(local)", "RAW_KW_VOID_CNT(cumulative)", "RAW_KW_BODY_VOID_CNT", "OP_SIZEOF_CNT(local)", "OP_SIZEOF_CNT(cumulative)", "RAW_OP_SIZEOF_CNT(local)", "RAW_OP_SIZEOF_CNT(cumulative)", "KW_CNT(local)", "KW_CNT(cumulative)", "KW_TYPES_CNT(local)", "KW_TYPES_CNT(cumulative)", "IDENT_LABEL_CNT", "RAW_IDENT_LABEL_CNT", "NUMERIC_CONST_CNT(local)", "NUMERIC_CONST_CNT(cumulative)", "NUMERIC_CONST_UNIQ(local)", "NUMERIC_CONST_UNIQ(cumulative)", "RAW_NUMERIC_CONST_CNT(local)", "RAW_NUMERIC_CONST_CNT(cumulative)", "RAW_NUMERIC_CONST_UNIQ(local)", "RAW_NUMERIC_CONST_UNIQ(cumulative)", "STRING_LITERALS(local)", "STRING_LITERALS(cumulative)", "STRING_LITERALS_UNIQ(local)",
                          "STRING_LITERALS_UNIQ(cumulative)", "RAW_STRING_LITERALS(local)", "RAW_STRING_LITERALS(cumulative)", "RAW_STRING_LITERALS_UNIQ(local)", "RAW_STRING_LITERALS_UNIQ(cumulative)", "CHAR_CONSTS", "CHAR_CONSTS_UNIQ", "RAW_CHAR_CONSTS", "RAW_CHAR_CONSTS_UNIQ", "UNRESERVED_IDENTIFIERS", "UNRESERVED_IDENTIFIERS_UNIQ", "BODY_UNRESERVED_IDENTIFIERS", "BODY_UNRESERVED_IDENTIFIERS_UNIQ", "RAW_UNRESERVED_IDENTIFIERS", "RAW_UNRESERVED_IDENTIFIERS_UNIQ", "VAR_FILE_LOC_CNT", "VAR_FILE_LOC_STATIC_CNT", "VAR_FILE_EXT_CNT", "VAR_FILE_VOLATILE_CNT", "VAR_FILE_CONST_CNT", "VAR_FN_LOC_CNT", "VAR_FN_LOC_STATIC_CNT", "VAR_FN_LOC_CONST_CNT", "VAR_FN_LOC_VOLATILE_CNT", "VAR_FN_LOC_REG_CNT", "VAR_FN_LOC_AUTO_CNT", "VAR_FN_EXT_CNT", "RETURN_POINT_CNT", "STMT_CNT(local)", "STMT_CNT(cumulative)", "RAW_STMT_CNT(local)", "RAW_STMT_CNT(cumulative)", "COMMENT_HIS_COMF", "COMMENT_BYTE_CNT", "COMMENT_CNT", "MCCABE", "MCCABE_MOD", "RAW_MCCABE", "RAW_MCCABE_MOD", "FUNC_LOCAL_CNT", "FUNC_EXTERN_EXPL_CNT(local)", "FUNC_EXTERN_EXPL_CNT(cumulative)", "FUNC_EXTERN_IMPL_CNT(local)", "FUNC_EXTERN_IMPL_CNT(cumulative)", "FUNC_INLINE_CNT", "FUNC_CALLED_BY_LOCAL", "FUNC_CALLED_BY_EXTERN", "OP_FN_CALL_CNT", "FUNC_PATHS", "OP_FN_CALL_UNIQUE_CNT", "LOCAL_FN_CALL_CNT", "FILE_LINE_CNT", "FUNC_DEFINITION_LINE_CNT", "FUNC_BODY_LINE_CNT", "STMT_HIS_PARAM", "TODO", "TODO", "FUNC_NESTING", "HIS_VOCF", "OP_ASSIGN_CNT", "OP_ADD_CNT", "OP_ADD_ASSIGN_CNT", "OP_SUB_CNT", "OP_SUB_ASSIGN_CNT", "OP_UNARY_PLUS_CNT", "OP_UNARY_MINUS_CNT", "OP_MULT_CNT", "OP_MULT_ASSIGN_CNT", "OP_DIV_CNT", "OP_DIV_ASSIGN_CNT", "OP_MOD_CNT", "OP_MOD_ASSIGN_CNT", "OP_INC_PRE_CNT", "OP_INC_POST_CNT", "OP_DEC_PRE_CNT", "OP_DEC_POST_CNT", "OP_SHFT_LEFT_CNT", "OP_SHFT_LEFT_ASSIGN_CNT", "OP_SHFT_RGHT_CNT", "OP_SHFT_RGHT_ASSIGN_CNT", "OP_CMP_LT_CNT", "OP_CMP_GT_CNT", "OP_CMP_LT_EQ_CNT", "OP_CMP_GT_EQ_CNT", "OP_CMP_EQ_CNT", "OP_CMP_NEQ_CNT", "OP_COMMA_CNT", "OP_TERNARY_CNT", "OP_LOG_AND_CNT", "OP_LOG_OR_CNT", "OP_LOG_NIT_CNT", "OP_BITWISE_AND_CNT", "OP_BITWISE_AND_ASSIGN_CNT", "OP_BITWISE_OR_CNT", "OP_BITWISE_OR_ASSIGN_CNT", "OP_BITWISE_XOR_CNT", "OP_BITWISE_XOR_ASSIGN_CNT", "OP_BITWISE_NOT_CNT", "OP_PTR_TO_MEMBER_DIRECT_CNT", "OP_PTR_TO_MEMBER_INDIRECT_CNT", "OP_ADDR_OF_CNT", "OP_DEREF_CNT", "OP_ARRAY_SUBSCRIPT_CNT", "OP_MEMBER_ACCESS_DIRECT_CNT", "OP_MEMBER_ACCESS_POINTER_CNT", "OP_ALIGNOF_CNT", "OP_CAST_CNT", "OP_TYPES_CNT(local)", "OP_TYPES_CNT(cumulative)", "OP_CNT(local)", "OP_CNT(cumulative)", "HALSTEAD_OPERATOR_UNIQUE_CNT", "HALSTEAD_OPERATOR_CNT", "HALSTEAD_OPERAND_UNIQUE_CNT", "HALSTEAD_OPERAND_CNT", "HALSTEAD_VOCABULARY", "HALSTEAD_LENGTH", "HALSTEAD_CALC_LENGTH", "HALSTEAD_VOLUME", "HALSTEAD_DIFFICULTY", "TOK_BOOL", "TOK_INLINE", "TOK_VIRTUAL", "TOK_MUTABLE", "TOK_FRIEND", "TOK_ASM", "TOK_CLASS", "TOK_DELETE", "TOK_NEW", "TOK_OPERATOR", "TOK_PRIVATE", "TOK_PROTECTED", "TOK_PUBLIC", "TOK_THIS", "TOK_NAMESPACE", "TOK_USING", "TOK_TRY", "TOK_CATCH", "TOK_THROW", "TOK_TYPEID", "TOK_TEMPLATE", "TOK_EXPLICIT", "TOK_TRUE", "TOK_FALSE", "TOK_TYPENAME", "TOK_NOT", "TOK_NOT_EQUAL", "TOK_MODULO", "TOK_MODULO_ASSIGN", "TOK_AMP", "TOK_AMPAMP", "TOK_PIPEPIPE", "TOK_AND_ASSIGN", "TOK_LPAREN", "TOK_RPAREN", "TOK_ASTERISK", "TOK_ASTERISK_ASSIGN", "TOK_PLUS", "TOK_PLUSPLUS", "TOK_PLUS_ASSIGN", "TOK_COMMA", "TOK_MINUS", "TOK_MINUSMINUS", "TOK_MINUS_ASSIGN", "TOK_MEMBER_PTR", "TOK_MEMBER_REF", "TOK_ELLIPSIS", "TOK_SLASH", "TOK_SLASH_ASSIGN", "TOK_COLON", "TOK_COLONCOLON", "TOK_LESS", "TOK_LESSLESS", "TOK_LESSLESS_ASSIGN", "TOK_LESS_EQUAL", "TOK_ASSIGN", "TOK_COMPARISON", "TOK_MORE", "TOK_MOREMORE", "TOK_MOREMORE_ASSIGN", "TOK_MORE_EQUAL", "TOK_LSQUARE", "TOK_RSQUARE", "TOK_LBRACE", "TOK_RBRACE", "TOK_QUESTION", "TOK_CARET", "TOK_CARET_ASSIGN", "TOK_PIPE", "TOK_PIPE_ASSIGN", "TOK_TILDE", "HIS_CALLING"]

##################################################################################################################
# only select needed metrics
METRICS_NAMES = METRICS_NAMES_UNEDITED[3:25:2] + METRICS_NAMES_UNEDITED[41:42] + METRICS_NAMES_UNEDITED[129:130] + METRICS_NAMES_UNEDITED[185:187] + \
    METRICS_NAMES_UNEDITED[198:199] + METRICS_NAMES_UNEDITED[207:208] + \
    METRICS_NAMES_UNEDITED[248:257] + METRICS_NAMES_UNEDITED[341:342]
# do nothing
# METRICS_NAMES = METRICS_NAMES_UNEDITED
##################################################################################################################

################################################################
# normal dataset
FEATURE_LABELS = list(["cwe"] + TOKENS + METRICS_NAMES)
#
# no tokens
# FEATURE_LABELS = list(["cwe"] + METRICS_NAMES)
#
# no metrics
# FEATURE_LABELS = list(["cwe"] + TOKENS)
################################################################

###################
# generate fmap.tsv
###################

desc = ['q' for i in range(len(FEATURE_LABELS))]
zipped = list(zip(FEATURE_LABELS, desc))
df = pandas.DataFrame(zipped)
df.to_csv('fmap.tsv', sep="\t", header=None)

#############################################
# USE:
# python3 tree.py dataset.csv [test-data.csv]
#############################################

data_csv = pandas.read_csv(sys.argv[1], header=None)

################################################################################################################
# select only needed metrics
select_metrics = list(range(3, 25, 2)) + list(range(41, 42)) + list(range(129, 130)) + list(range(185, 187)) + \
    list(range(198, 199)) + list(range(207, 208)) + \
    list(range(248, 257)) + list(range(341, 342))
select_metrics = list(
    set(range(len(METRICS_NAMES_UNEDITED))).difference(select_metrics))
select_metrics = [i + (2+2*TOKEN_WINDOW) for i in select_metrics]
data_csv.drop(columns=select_metrics, inplace=True)
################################################################################################################

temp_cols = data_csv.columns.tolist()

##############################################################################
# normal dataset
new_cols = temp_cols[1:-1] + temp_cols[:1]
#
# no tokens
# new_cols = temp_cols[1:2] + temp_cols[(2+2*TOKEN_WINDOW):-1] + temp_cols[:1]
#
# no metrics
# new_cols = temp_cols[1:(2+2*TOKEN_WINDOW)] + temp_cols[:1]
##############################################################################

data_csv = data_csv[new_cols]

cwe_list = list(set(data_csv[1].tolist()))

# print(cwe_list)
# exit(0)

train = pandas.DataFrame()
test = pandas.DataFrame()

for cwe in cwe_list:
    mask = (data_csv[1] == cwe)
    data_cwe = data_csv[mask]

    data_cwe_t = data_cwe[data_cwe[0] == True]
    data_cwe_f = data_cwe[data_cwe[0] == False]

    # ignore all CWEs without FP
    if (data_cwe_f.size == 0):
        continue

    data_cwe_t_train = data_cwe_t.sample(frac=TRAIN_TEST_RATIO)
    data_cwe_t_test = data_cwe_t.drop(data_cwe_t_train.index)

    data_cwe_f_train = data_cwe_f.sample(frac=TRAIN_TEST_RATIO)
    data_cwe_f_test = data_cwe_f.drop(data_cwe_f_train.index)

    train = pandas.concat(
        [train, data_cwe_t_train, data_cwe_f_train], ignore_index=True)
    test = pandas.concat(
        [test, data_cwe_t_test, data_cwe_f_test], ignore_index=True)

####################################
# preprocess train and test datasets
####################################

train = train.fillna(0)
test = test.fillna(0)

X_train = train.to_numpy()[:, :-1]
X_train = preprocessing.normalize(X_train)
Y_train = train.iloc[:, -1:].transpose().to_numpy()[0]
Y_train = Y_train * 1  # convert True, False to 1, 0

X_test = test.to_numpy()[:, :-1]
X_test = preprocessing.normalize(X_test)
Y_test = test.iloc[:, -1:].transpose().to_numpy()[0]
Y_test = Y_test * 1  # convert True, False to 1, 0

######################
# simple decision tree
######################

clf = tree.DecisionTreeClassifier(max_depth=4, criterion="entropy")
clf.fit(X_train, Y_train)
Y_test_pred = clf.predict(X_test)

print("Accuracy Score Tree: {}".format(accuracy_score(Y_test, Y_test_pred)))
print("Precision Tree: {}".format(
    precision_score(Y_test, Y_test_pred, pos_label=0)))
print("Recall Score Tree: {}".format(
    recall_score(Y_test, Y_test_pred,  pos_label=0)))


plt.figure()
tree.plot_tree(clf, filled=True, feature_names=FEATURE_LABELS)
plt.savefig('tree.pdf', format='pdf', bbox_inches="tight", dpi=1200)

#######################
# xgboost tree boosting
#######################

estimator = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=4,
    seed=42,
    use_label_encoder=False
)

parameters = {
    'max_depth': range(2, 5, 1),
    'n_estimators': range(5, 55, 10),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring='roc_auc',  # try f1
    n_jobs=10,
    cv=2
)

grid_search.fit(X_train, Y_train)
Y_predict = grid_search.best_estimator_.predict(X_test)

plt.figure()
xgb.plot_tree(grid_search.best_estimator_, filled=True, fmap='fmap.tsv')
plt.savefig('xgboost-tree.pdf', format='pdf', bbox_inches="tight", dpi=1200)

print("Accuracy Score XGBoost: {}".format(accuracy_score(Y_test, Y_predict)))
print("Precision Score XGBoost: {}".format(
    precision_score(Y_test, Y_predict, pos_label=0)))
print("Recall Score XGBoost: {}".format(
    recall_score(Y_test, Y_predict, pos_label=0)))

###################################
# if input given, predict for input
###################################

if (len(sys.argv) == 3):
    test_data = pandas.read_csv(sys.argv[2], header=None)

    ################################################################################################################
    # select only needed metrics
    select_metrics = list(range(3, 25, 2)) + list(range(41, 42)) + list(range(129, 130)) + list(range(185, 187)) + \
        list(range(198, 199)) + list(range(207, 208)) + \
        list(range(248, 257)) + list(range(341, 342))
    select_metrics = list(
        set(range(len(METRICS_NAMES_UNEDITED))).difference(select_metrics))
    select_metrics = [i + (1+2*TOKEN_WINDOW) for i in select_metrics]
    test_data.drop(columns=select_metrics, inplace=True)
    ################################################################################################################

    cols = test_data.columns.tolist()

    ################################################
    # normal dataset
    cols = cols[:-1]
    #
    # no tokens
    # cols = cols[0:1] + cols[(1+2*TOKEN_WINDOW):-1]
    #
    # no metrics
    # cols = cols[0:(1+2*TOKEN_WINDOW)]
    ################################################

    test_data = test_data[cols]

    test = test_data.fillna(0).to_numpy()
    test = preprocessing.normalize(test)
    predict = clf.predict(test)
    proba = clf.predict_proba(test)
    print("predict tree")
    print(predict)
    print(proba)
    predict = grid_search.best_estimator_.predict(test)
    proba = grid_search.best_estimator_.predict_proba(test)
    print("predict xgboost")
    print(predict)
    print(proba)
