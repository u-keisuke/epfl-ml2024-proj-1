# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_selection import VarianceThreshold

from utils.helpers import create_csv_submission

import numpy as np

x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
y_train[y_train==-1] = 0

# x_test = pd.read_csv('dataset/x_test.csv', index_col=0)
# print(x_train.describe())
# print(x_train.head())

####################################################### drop nan values
# threshold = 0.5 * len(x_train)
# columns_to_drop = x_train.columns[x_train.isnull().sum() > threshold]
# print(columns_to_drop)

# x_train = x_train.dropna(axis=1, thresh=threshold)
# print(x_train.head())

x_train.fillna(0, inplace=True)
############################################################ drop low variance columns
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# selector = VarianceThreshold(threshold=0.01)
# x_train = selector.fit_transform(x_train)

############################### High correlated features
# corr_matrix = x_train.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# x_train = x_train.drop(columns=to_drop)
################################################## PCA
# pca = PCA(n_components=150)  
# x_train = pca.fit_transform(x_train)

x_train = np.array(x_train)
print(x_train.shape)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()
# y_val = y_val.values.ravel()
####################################### Balancing the data #########################
# x_train_1 = x_train[y_train == 1]
# y_train_1 = y_train[y_train == 1]
# x_train_0 = x_train[y_train == 0]
# y_train_0 = y_train[y_train == 0]

# x_train = np.concatenate([x_train] + [x_train_1] * 9)
# y_train = np.concatenate([y_train] + [y_train_1] * 9)
# x_val_0 = x_val[y_val == 1]
# y_val_0 = y_val[y_val == 1]
# x_val = np.concatenate([x_val] + [x_val_0] * 9)
# y_val = np.concatenate([y_val] + [y_val_0] * 9)
# Validation Accuracy: 73.59%
# Validation F1 Score: 0.69

# num_samples_to_select = int(len(x_train_1) * 2.5)
# random_indices = np.random.choice(len(x_train_0), size=num_samples_to_select, replace=False)
# x_train_0_undersampled = x_train_0[random_indices]
# y_train_0_undersampled = y_train_0[random_indices]
# x_train = np.concatenate([x_train_0_undersampled, x_train_1])
# y_train = np.concatenate([y_train_0_undersampled, y_train_1])
# shuffle_indices = np.random.permutation(len(y_train))
# x_train = x_train[shuffle_indices]
# y_train = y_train[shuffle_indices]
pos = sum(y_train == 1)
neg = len(y_train) - pos
print(pos, neg)
###########################################################################################
sample_weights=np.where(y_train == 1, 3.5, 1)

x_train = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
# x_val = xgb.DMatrix(x_val, label=y_val)
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.10,  # Initial learning rate
    "gamma": 0,
    "lambda": 1,
    # "colsample_bytree": 0.8
    # "min_child_weight": 5
}
num_boost_round = 160
eta_decay = 0.95  # Factor to decrease eta
decay_interval = 40  # Decrease eta every 50 rounds
model = None
for _ in range(0, num_boost_round, decay_interval):
    params['eta'] *= eta_decay
    model = xgb.train(
        params,
        x_train,
        num_boost_round=decay_interval,
        xgb_model=model,  
        # evals=[(x_val, 'eval')],
        # early_stopping_rounds=20  
    )

"""
Validation F1 Score: 0.4382
Validation Recall: 0.513
Validation Precision: 0.382
"""
# model = xgb.XGBClassifier(objective='binary:logistic', max_depth=7, eta=0.10, n_estimators=300) #7, 0.1, 200-> 0.4323
# model.fit(x_train, y_train, sample_weight=sample_weights)


y_train_pred_proba = model.predict(x_train)  # Get probability predictions
y_train_pred = np.array([1 if pred > 0.5 else 0 for pred in y_train_pred_proba])
print(np.all(y_train_pred==1))

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)  # Calculate F1 score for training
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Train F1 Score: {train_f1:.2f}')


###################################################### Validation set
# y_val_pred_proba = model.predict(x_val)  # Get probability predictions
# y_val_pred = np.array([1 if pred > 0.5 else 0 for pred in y_val_pred_proba])  # Convert to 1 or 0
# val_accuracy = accuracy_score(y_val, y_val_pred)
# val_f1 = f1_score(y_val, y_val_pred)
# val_recall = recall_score(y_val, y_val_pred)
# val_precision = precision_score(y_val, y_val_pred)
# print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
# print(f'Validation F1 Score: {val_f1:.4f}')
# print(f'Validation Recall: {val_recall:.3f}')
# print(f'Validation Precision: {val_precision:.3f}') #low: many of the predicted 1s are false
# pos = sum(y_val_pred)
# neg = len(y_val_pred) - pos
# print(pos, neg)
# pos = sum(y_val)
# neg = len(y_val) - pos
# print(pos, neg)
# y_val_pred = 1 - y_val_pred
# y_val = 1 - y_val
# val_f1 = f1_score(y_val, y_val_pred)  # Calculate F1 score for validation
# print(f'Validation F1 Score of 0-1: {val_f1:.4f}')

####################################################### Test set
x_test = pd.read_csv('dataset/x_test.csv')
x_test.fillna(0, inplace=True)

x_test = np.array(x_test)
test_ids = x_test[:, 0].astype(dtype=int)
x_test = x_test[:, 1:] #we don't need ids

print(x_test.shape)
x_test = xgb.DMatrix(x_test)
y_pred = model.predict(x_test)

y_pred = np.array([1 if pred > 0.5 else 0 for pred in y_pred])

pos = sum(y_pred.round())
neg = len(y_pred) - pos
print(pos, neg)

print(np.all(y_pred==1))
y_pred = 2 * (y_pred.round()) - 1
create_csv_submission(test_ids, y_pred, "xgboost_meme")