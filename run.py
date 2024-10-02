import argparse

import numpy as np

from utils.helpers import create_csv_submission, load_csv_data
from utils.random import set_random_seed


def main(data_path, output_path, model_name):
    set_random_seed()

    # 1. Load the data
    data_path = "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    # 2. Predict
    if model_name == "monkey":
        from models.monkey import make_prediction
        y_pred = make_prediction(model_name, x_train, x_test, y_train, train_ids, test_ids)

    elif model_name == "lightgbm":
        from models.light_gbm import make_prediction
        y_pred = make_prediction(model_name, x_train, x_test, y_train, train_ids, test_ids)
        
    else:
        raise ValueError(f"Model {model_name} not found")

    # 3. Create the submission file
    create_csv_submission(test_ids, y_pred, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--data_path", type=str, default="dataset", help="Path to the data folder")
    parser.add_argument("--output_path", type=str, default="submission.csv", help="Path to the output file")
    parser.add_argument("--model", type=str, default="monkey", help="Model to use for prediction")
    args = parser.parse_args()

    main(args.data_path, args.output_path, args.model)
