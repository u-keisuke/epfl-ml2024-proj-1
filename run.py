import argparse
import time

import numpy as np

from boost_tree import BoostForest
from helpers import (
    accuracy,
    create_csv_submission,
    f1_score,
    load_test_data,
    load_train_data,
    train_test_split,
)


def train_model(file_path):
    x_train, y_train = load_train_data("dataset")

    x_train = np.nan_to_num(x_train)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train[y_train == -1] = 0
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )
    x_train = x_train[:100, :]
    y_train = y_train[:100]
    assert len(y_train.shape) == 1, "y should be one dimensional array"

    classifier_forest = BoostForest(
        max_depth=6,
        lr=0.1,
        decay_rate=0.95,
        decay_interval=None, # 40
        random_state=42,
        num_trees=16,
        cover=1,
        max_features=0.8,
        lambda_regularizer=1,
        gamma_regularizer=0,
    )  # regularization to choose
    
    print("loaded the previous model's", len(classifier_forest.Forest), "trees")

    print("Training the model")
    start = time.time()
    sample_weights = np.where(
        y_train == 1, 3.5, 1
    )  # make y=1 more important for loss function
    classifier_forest.fit(x_train, y_train, sample_weights, file_path)
    print("It took us", time.time() - start, "seconds")

    # save the model
    classifier_forest.save_model(file_path)

    # predict on the validation set
    y_val_pred = classifier_forest.predict(x_val)
    
    # calculate the accuracy and f1 score
    accuracy_forest = accuracy(y_val, y_val_pred.round())
    val_f1 = f1_score(y_val, y_val_pred.round())
    
    # log
    print(f"Validation Accuracy: {accuracy_forest * 100:.2f}%")
    print(f"Validation F1 Score: {val_f1:.4f}")


def test_model(file_path, output_path):
    # load the model
    print("Loading the model")
    classifier_forest = BoostForest()
    classifier_forest.load_model(file_path, num_trees=None)

    # load the test data
    print("Loading the test data")
    x_test, test_ids = load_test_data("dataset")
    x_test = np.nan_to_num(x_test)
    x_test = np.array(x_test)

    # predict on the test set
    print("Predicting on the test data")
    y_pred = classifier_forest.predict(x_test)
    ## pos = sum(y_pred.round())
    ## neg = len(y_pred) - pos
    ## print(pos, neg)
    y_pred = 2 * (y_pred.round()) - 1

    # create the submission file
    print("Creating the submission file")
    create_csv_submission(test_ids, y_pred, output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--re-train", action="store_true")
    args.add_argument("--model-path", type=str, default="models/model7.pkl")
    args.add_argument("--output-path", type=str, default="prediction.csv")
    args = args.parse_args()

    # training from scratch
    if args.re_train:
        train_model(file_path=args.model_path)

    # testing the model
    test_model(file_path=args.model_path, output_path=args.output_path)
