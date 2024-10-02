# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# Step 1: Load the datasets
x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
x_test = pd.read_csv('dataset/x_test.csv', index_col=0)

# Step 2: Split x_train into x_train and x_val, and also y_train into y_train and y_val
y_train[y_train==-1] = 0

print(y_train.head())
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# print(y_train)

y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

# Step 3: Preprocess data (if needed)
# print(f'Missing values in x_train:\n{x_train.isnull().sum()}\n')
# print(f'Missing values in x_val:\n{x_val.isnull().sum()}\n')

x_val.fillna(x_val.mean(), inplace=True)
x_test.fillna(x_test.mean(), inplace=True)

# Step 4: Train the XGBoost model
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)


xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # learning rate or 'eta'
    'max_depth': [3, 4, 5],  # maximum depth of the trees
    'n_estimators': [100, 200, 300],  # number of boosting rounds
    'subsample': [0.7, 0.8, 1.0],  # fraction of data to use for training each tree
    'colsample_bytree': [0.7, 0.8, 1.0],  # fraction of features to consider at each split
    'gamma': [0, 0.1, 0.2],  # minimum loss reduction required to make a further partition
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term
    'reg_lambda': [1, 1.5, 2]  # L2 regularization term
}
# Step 5: Set up the GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

# Step 6: Fit GridSearchCV on the training data
grid_search.fit(x_train, y_train)

# Step 7: Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters found: {best_params}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Step 8: Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy with the best model: {val_accuracy * 100:.2f}%")


