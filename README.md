# Predicting Cardiovascular Disease Risk with Boosting Trees and Logistic Regression


## Overview
This project implements a Boosted Decision Tree model and Logistic Regression for binary classification task, in particular, predicting Cardiovascular Disease Risk. 


1. **Boosted Decision Tree Model**: A highly adaptable model — custom-built variant inspired by XGBoost — designed with advanced techniques to enhance performance and efficiency:
    - **Handling Imbalanced Classes**: Adjusts sample weights to prioritize underrepresented classes, making it suitable for skewed datasets, like ours.
    - **Learning Rate Decay**: Gradually decreases learning rate to stabilize training, reducing the risk of overfitting and improving convergence.
    - **Tree Pruning with Regularization**: Leverages `lambda`, `gamma`, and `cover` parameters to streamline the tree structure.
    - **Randomized Feature Selection**: Selects a subset of features for each tree to speed up training.
    - **Save and load functionalities**:  Enables efficient model reuse and iterative training.

2. **Logistic Regression**: A robust baseline model, complementing the Boosted Decision Tree to provide interpretable insights into Cardiovascular Disease Risk prediction:
    - **Handling Imbalanced Classes**: Adjusts weights for binary cross entropy to prioritize underrepresented classes, making it suitable for skewed datasets, like ours.
    - **Missing value handling**: Changing NaN values to zeros in order to run the logistic regression
    - **Normalizing the dataset**: MinMax normalization to make features robuster
    - **Removing correlated features**: Increases convergence of the logistic regression


## Repository Structure

- `run.py`: Main script for training and testing the Boosted Decision Tree model. 
- `implementations.py`: Contains helper functions for linear regression, logistic regression.
- `boost_tree.py`: Defines the core classes (`BoostTree` and `BoostForest`) implementing the boosting algorithm (self-implemented xgboost) and the decision tree structure.
- `helper.py`: contains the functions to read dataset, calculate metrics, and write the submission file.
- `logreg.ipynb`: contains code for running ablation study for logistic regression.


## Installation

Download the code:
```
git clone git@github.com:u-keisuke/epfl-ml2024-proj-1.git
cd epfl-ml2024-proj-1
```

Install required packages (for python 3.10):
```
pip install numpy
```

## Usage

### 0. Download dataset

Download the dataset from [here](https://github.com/epfml/ML_course/tree/main/projects/project1/data), and place them under the `dataset` folder.


### 1. Training Boost trees

To train the model, run the `train_model` function within `run.py`. This function loads the training dataset, initializes the Boosted Forest model, and saves the trained model to the specified file.
```
python3 run.py --re-train --model-path models/model_new.pkl
```

### 2. Testing Boosting trees

To test the model, execute the `run.py` script using the `test_model` function, which requires the path to a pretrained model. This function loads the test dataset, initializes the Boosted Forest model, loads the specified pretrained model, and saves the predictions for the test set in a CSV file.
```
python3 run.py --model-path models/model7.pkl
```





