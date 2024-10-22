# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score  # Add f1_score to the imports

import numpy as np

x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
# x_test = pd.read_csv('dataset/x_test.csv', index_col=0)

y_train[y_train==-1] = 0
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

x_train.fillna(0, inplace=True)
x_val.fillna(0, inplace=True)

x_train = np.array(x_train)
x_train_1 = x_train[y_train == 1]
y_train_1 = y_train[y_train == 1]
x_train_0 = x_train[y_train == 0]
y_train_0 = y_train[y_train == 0]

####################################### Balancing the data #########################

# x_train = np.concatenate([x_train] + [x_train_1] * 9)
# y_train = np.concatenate([y_train] + [y_train_1] * 9)
# x_val_0 = x_val[y_val == 1]
# y_val_0 = y_val[y_val == 1]
# x_val = np.concatenate([x_val] + [x_val_0] * 9)
# y_val = np.concatenate([y_val] + [y_val_0] * 9)
# Validation Accuracy: 73.59%
# Validation F1 Score: 0.69

num_samples_to_select = int(len(x_train_1) * 2.5)
random_indices = np.random.choice(len(x_train_0), size=num_samples_to_select, replace=False)
x_train_0_undersampled = x_train_0[random_indices]
y_train_0_undersampled = y_train_0[random_indices]
x_train = np.concatenate([x_train_0_undersampled, x_train_1])
y_train = np.concatenate([y_train_0_undersampled, y_train_1])
shuffle_indices = np.random.permutation(len(y_train))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]
pos = sum(y_train == 1)
neg = len(y_train) - pos
print(pos, neg)
###########################################################################################

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
params = {
    'objective': 'binary:logistic',  # Binary classification
    'max_depth': 12,  # Maximum depth of the tree
    'eta': 0.10,  # Learning rate
}

bst = xgb.train(params, dtrain, num_boost_round=45) #, evals=[(dval, 'eval')], early_stopping_rounds=30)

y_train_pred_proba = bst.predict(dtrain)  # Get probability predictions
y_train_pred = [1 if pred > 0.5 else 0 for pred in y_train_pred_proba]  # Convert to 1 or 0
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)  # Calculate F1 score for training
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Train F1 Score: {train_f1:.2f}')

y_val_pred_proba = bst.predict(dval)  # Get probability predictions
y_val_pred = np.array([1 if pred > 0.5 else 0 for pred in y_val_pred_proba])  # Convert to 1 or 0
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')


val_f1 = f1_score(y_val, y_val_pred)
print(f'Validation F1 Score of 1-0: {val_f1:.4f}')

pos = sum(y_val_pred)
neg = len(y_val_pred) - pos
print(pos, neg)

pos = sum(y_val)
neg = len(y_val) - pos
print(pos, neg)

# y_val_pred = 1 - y_val_pred
# y_val = 1 - y_val
# val_f1 = f1_score(y_val, y_val_pred)  # Calculate F1 score for validation
# print(f'Validation F1 Score of 0-1: {val_f1:.4f}')