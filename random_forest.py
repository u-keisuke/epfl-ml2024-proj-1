import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

from utils_copy.decision_tree import entropy, gini, variance, mad_median, DecisionTree
from utils_copy.decision_tree import RandomForest
from utils_copy.boost_tree import BoostForest


x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
y_train[y_train==-1] = 0

x_train = np.array(x_train)




x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

y_train = np.array(y_train)
x_train = x_train[:100000, :]
y_train = y_train[:100000]


############################################### Oversampling
x_train_1 = x_train[y_train == 1]
y_train_1 = y_train[y_train == 1]
x_train_0 = x_train[y_train == 0]
y_train_0 = y_train[y_train == 0]
x_train = np.concatenate([x_train] + [x_train_1] * 2)
y_train = np.concatenate([y_train] + [y_train_1] * 2)

# # x_val_0 = x_val[y_val == 1]
# y_val_0 = y_val[y_val == 1]
# x_val = np.concatenate([x_val] + [x_val_0] * 9)
# y_val = np.concatenate([y_val] + [y_val_0] * 9)
pos = sum(y_train == 1)
neg = len(y_train) - pos
print(pos, neg)

classifier_forest = BoostForest(max_depth=6, learning_rate=0.2, random_state=None, num_trees=20,
                                lambda_regularizer=1, gamma_regularizer=1)
print("starting" )
start = time.time()
# sampleweight = ..
classifier_forest.fit(x_train, y_train)
print("It took us", time.time()-start, "seconds")

y_val_pred = classifier_forest.predict(x_val)
accuracy_forest = accuracy_score(y_val, y_val_pred.round())
print(accuracy_forest)

f1_scoree = f1_score(y_val, y_val_pred.round())
print(f1_scoree)