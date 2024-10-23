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
y_train = np.array(y_train)
print(x_train.shape)
x_train = x_train[:5000, :]
y_train = y_train[:5000, :]
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

classifier_forest = BoostForest(max_depth=6, learning_rate=0.2, random_state=None, num_trees=10)
print("starting" )
start = time.time()
# sampleweight = ..
classifier_forest.fit(x_train, y_train)
print("finished", time.time()-start)

y_val_pred = classifier_forest.predict(x_val)
print(y_val_pred)
accuracy_forest = accuracy_score(y_val, y_val_pred.round())
print(accuracy_forest)

f1_scoree = f1_score(y_val, y_val_pred.round())
print(f1_scoree)