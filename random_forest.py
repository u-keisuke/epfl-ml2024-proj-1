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
print(x_train.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

y_train = np.array(y_train)
# x_train = x_train[:10000, :]
# y_train = y_train[:10000]


############################################### Oversampling
# x_train_1 = x_train[y_train == 1]
# y_train_1 = y_train[y_train == 1]
# x_train_0 = x_train[y_train == 0]
# y_train_0 = y_train[y_train == 0]
# x_train = np.concatenate([x_train] + [x_train_1] * 2)
# y_train = np.concatenate([y_train] + [y_train_1] * 2)

pos = sum(y_train == 1)
neg = len(y_train) - pos
print(pos, neg)

classifier_forest = BoostForest(max_depth=6, learning_rate=0.1, random_state=None, num_trees=100, #160
                                lambda_regularizer=1, gamma_regularizer=1) #regularization to choose

# need to implement lr decay
print("starting" )
start = time.time()

sample_weights = np.where(y_train == 1, 3.5, 1) # make y=1 more important for loss function
classifier_forest.fit(x_train, y_train, sample_weights)
print("It took us", time.time()-start, "seconds")

y_val_pred = classifier_forest.predict(x_val)
accuracy_forest = accuracy_score(y_val, y_val_pred.round())
val_f1 = f1_score(y_val, y_val_pred.round())
val_recall = recall_score(y_val, y_val_pred.round())
val_precision = precision_score(y_val, y_val_pred.round())
print(f'Validation Accuracy: {accuracy_forest * 100:.2f}%')
print(f'Validation F1 Score: {val_f1:.4f}')
print(f'Validation Recall: {val_recall:.3f}')
print(f'Validation Precision: {val_precision:.3f}') #low: many of the predicted 1s are false