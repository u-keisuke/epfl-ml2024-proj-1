import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

from boost_tree import BoostForest

from utils.helpers import create_csv_submission

x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
x_train.fillna(0, inplace=True)
y_train[y_train==-1] = 0


x_train = np.array(x_train)
y_train = np.array(y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train = x_train[:100000, :]
y_train = y_train[:100000, 0]
assert len(y_train.shape)==1, "y should be one dimensional array"

pos = sum(y_train == 1)
neg = len(y_train) - pos
print(pos, neg)

classifier_forest = BoostForest(max_depth=6, lr=0.1, decay_rate=0.95, decay_interval=None, #40
                                random_state=42, num_trees=16, cover=1, max_features=0.8,
                                lambda_regularizer=1, gamma_regularizer=0) # regularization to choose

start = time.time()
sample_weights = np.where(y_train == 1, 3.5, 1) # make y=1 more important for loss function
file_path = "/home/dimgor/Рабочий стол/projects/epfl-ml2024-proj-1/model1.pkl"
classifier_forest.load_model(file_path)
print(len(classifier_forest.Forest))
print("loaded the prev model's", len(classifier_forest.Forest), "trees")
classifier_forest.fit(x_train, y_train, sample_weights, file_path)
classifier_forest.save_model(file_path)
print("It took us", time.time()-start, "seconds")

y_val_pred = classifier_forest.predict(x_val)
accuracy_forest = accuracy_score(y_val, y_val_pred.round())
val_f1 = f1_score(y_val, y_val_pred.round())
val_recall = recall_score(y_val, y_val_pred.round())
val_precision = precision_score(y_val, y_val_pred.round())
print(f'Validation Accuracy: {accuracy_forest * 100:.2f}%')
print(f'Validation F1 Score: {val_f1:.4f}')
print(f'Validation Recall: {val_recall:.3f}')
print(f'Validation Precision: {val_precision:.3f}') # low: many of the predicted 1s are false

####################################################### Test part
# x_test = pd.read_csv('dataset/x_test.csv')
# x_test.fillna(0, inplace=True)
# x_test = np.array(x_test)
# test_ids = x_test[:, 0].astype(dtype=int)
# x_test = x_test[:, 1:] # we don't need ids
# y_pred = classifier_forest.predict(x_test)
# pos = sum(y_pred.round())
# neg = len(y_pred) - pos
# print(pos, neg)
# y_pred = 2 * (y_pred.round()) - 1
# create_csv_submission(test_ids, y_pred, "meme2")