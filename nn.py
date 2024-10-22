import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

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

# Step 3: Standardize the data (PCA requires standardization)
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_val_scaled = scaler.transform(x_val)
# x_test_scaled = scaler.transform(x_test)

# # Step 4: Apply PCA to reduce dimensionality (e.g., reduce to 50 components)
# pca = PCA(n_components=50)
# x_train_pca = pca.fit_transform(x_train_scaled)
# x_val_pca = pca.transform(x_val_scaled)
# x_test_pca = pca.transform(x_test_scaled)

# Step 5: Convert the data into PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Unsqueeze to match the output shape
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Step 6: Define the feedforward neural network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 32)          # Third hidden layer with 32 units
        self.fc4 = nn.Linear(32, 1)           # Output layer (1 output for binary classification)
        self.sigmoid = nn.Sigmoid()           # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # Sigmoid applied at the output layer
        return x

# Step 7: Initialize the model, define the loss function and optimizer
input_size = x_train.shape[1]  # Number of PCA components
model = FeedforwardNN(input_size)

# Binary cross-entropy loss for binary classification
criterion = nn.BCELoss()

# Stochastic Gradient Descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 8: Train the model
epochs = 50  # Number of epochs to train
for epoch in range(epochs):
    # Forward pass: Compute predictions and loss
    y_train_pred = model(x_train_tensor)
    loss = criterion(y_train_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 9: Evaluate the model on the validation set
with torch.no_grad():  # Disable gradient calculation for evaluation
    y_val_pred = model(x_val_tensor)
    y_val_pred = y_val_pred.numpy()
    y_val_pred = [1 if pred > 0.5 else 0 for pred in y_val_pred]  # Convert probabilities to binary (0 or 1)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

