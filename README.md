# Predicting Cardiovascular Disease Risk with Boosting Trees and Logistic Regression


### Overview
This project implements a Boosted Decision Tree model and Logistic Regression for binary classification task, in particular, predicting Cardiovascular Disease Risk. 


1. **Boosted Decision Tree Model**: A highly adaptable model — custom-built variant inspired by XGBoost — designed with advanced techniques to enhance performance and efficiency:
   - **Handling Imbalanced Classes**: Adjusts sample weights to prioritize underrepresented classes, making it suitable for skewed datasets, like ours.
   - **Learning Rate Decay**: Gradually decreases learning rate to stabilize training, reducing the risk of overfitting and improving convergence.
   - **Tree Pruning with Regularization**: Leverages `lambda`, `gamma`, and `cover` parameters to streamline the tree structure.
   - **Randomized Feature Selection**: Selects a subset of features for each tree to speed up training.
   - **Save and load functionalities**:  Enables efficient model reuse and iterative training.

2. **Logistic Regression**: A robust baseline model, complementing the Boosted Decision Tree to provide interpretable insights into Cardiovascular Disease Risk prediction. ?????????????????????????

### Repository Structure

- `run.py`: Main script for training and testing the Boosted Decision Tree model. 
- `implementations.py`: Contains helper functions for linear regression, logistic regression.
- `boost_tree.py`: Defines the core classes (`BoostTree` and `BoostForest`) implementing the boosting algorithm (self-implemented xgboost) and the decision tree structure.
- `helper.py`: contains the functions to read dataset, calculate metrics, and write the submission file.
- `logreg.ipynb`: ??????????????????
### Installation

Download the code:
```
git clone git@github.com:u-keisuke/epfl-ml2024-proj-1.git
cd epfl-ml2024-proj-1
```

Install uv:
```
pip install uv
```

A uv environment can be created:
```
uv python pin 3.10
uv sync
```

Install required packages:
```
pip install numpy
```

### Usage

1. **Training Boost trees**:
   To train the model, use the `train_model` function in `run.py`. This function loads training data, initializes the Boosted Forest, and saves the trained model to a file.

2. **Testing Boosting trees**:
   To test the model, just run the file `run.py`, that uses function `test_model`, specifying the link to pretrained model. This function loads testing data, initializes the Boosted Forest, loads the pretrained model, and saves the prediction for test set.

    ```
    python3 run.py
    ```

### Dataset

Download the dataset from [here](https://github.com/epfml/ML_course/tree/main/projects/project1/data).



