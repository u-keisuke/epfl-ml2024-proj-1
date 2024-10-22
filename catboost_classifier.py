# Import necessary libraries
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the datasets
x_train = pd.read_csv('dataset/x_train.csv', index_col=0)
y_train = pd.read_csv('dataset/y_train.csv', index_col=0)  # Assuming y_train is a separate file
x_test = pd.read_csv('dataset/x_test.csv', index_col=0)

# Step 2: Preprocess labels and split data into training and validation sets
y_train[y_train == -1] = 0  # Convert -1 labels to 0 for binary classification

# Split x_train into x_train and x_val, and also y_train into y_train and y_val
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Flatten the labels if needed
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

# Step 3: Handle missing values (CatBoost can handle NaNs natively, but we'll fill them with 0 here)
x_train.fillna(0, inplace=True)
x_val.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)


# Step 3: Standardize the data (PCA requires standardization)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Step 4: Apply PCA to reduce dimensionality (choose the number of components, e.g., 50)
pca = PCA(n_components=50)  # Adjust the number of components as needed
x_train_pca = pca.fit_transform(x_train_scaled)
x_val_pca = pca.transform(x_val_scaled)
x_test_pca = pca.transform(x_test_scaled)


# Step 4: Train the CatBoost model
catboost_model = CatBoostClassifier(
    iterations=100,           # Number of boosting rounds
    depth=8,                 # Maximum tree depth
    learning_rate=0.30,       # Learning rate (equivalent to eta in XGBoost)
    loss_function='Logloss',  # Loss function for binary classification
    verbose=False             # Disable verbose output
)

# Fit the model
catboost_model.fit(x_train, y_train)

# Step 5: Make predictions and evaluate the model on the training set
y_train_pred_proba = catboost_model.predict_proba(x_train)[:, 1]  # Get probability predictions for the positive class
y_train_pred = [1 if pred > 0.5 else 0 for pred in y_train_pred_proba]  # Convert probabilities to binary (1 or 0)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

# Step 6: Make predictions and evaluate the model on the validation set
y_val_pred_proba = catboost_model.predict_proba(x_val)[:, 1]  # Get probability predictions for the positive class
y_val_pred = [1 if pred > 0.5 else 0 for pred in y_val_pred_proba]  # Convert probabilities to binary (1 or 0)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
