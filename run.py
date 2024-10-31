import time

import numpy as np 

from boost_tree import BoostForest
from utils.helpers import create_csv_submission, load_train_data, train_test_split, load_test_data, accuracy, f1_score


####################################################### Training part
def train_model():
    x_train, y_train = load_train_data('dataset')

    x_train = np.nan_to_num(x_train)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train[y_train==-1] = 0
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = x_train[:100, :]
    y_train = y_train[:100]
    assert len(y_train.shape)==1, "y should be one dimensional array"

    classifier_forest = BoostForest(max_depth=6, lr=0.1, decay_rate=0.95, decay_interval=None, #40
                                    random_state=42, num_trees=16, cover=1, max_features=0.8,
                                    lambda_regularizer=1, gamma_regularizer=0) # regularization to choose
    start = time.time()
    sample_weights = np.where(y_train == 1, 3.5, 1) # make y=1 more important for loss function
    file_path = ".../model.pkl"
    print("loaded the previous model's", len(classifier_forest.Forest), "trees")
    classifier_forest.fit(x_train, y_train, sample_weights, file_path)
    classifier_forest.save_model(file_path)
    print("It took us", time.time()-start, "seconds")

    y_val_pred = classifier_forest.predict(x_val)
    accuracy_forest = accuracy(y_val, y_val_pred.round())
    val_f1 = f1_score(y_val, y_val_pred.round())
    print(f'Validation Accuracy: {accuracy_forest * 100:.2f}%')
    print(f'Validation F1 Score: {val_f1:.4f}')


####################################################### Test part
def test_model(file_path="model7.pkl"):

    classifier_forest = BoostForest() 
    classifier_forest.load_model(file_path, num_trees=None)

    x_test, test_ids= load_test_data('dataset')
    x_test = np.nan_to_num(x_test)
    x_test = np.array(x_test)
    y_pred = classifier_forest.predict(x_test)
    # pos = sum(y_pred.round())
    # neg = len(y_pred) - pos
    # print(pos, neg)
    y_pred = 2 * (y_pred.round()) - 1
    create_csv_submission(test_ids, y_pred, "prediction")


if __name__ == "__main__":
    test_model(file_path="model7.pkl")
    # train_model(file_path="...")