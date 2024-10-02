import numpy as np


def make_prediction(model_name, x_train, x_test, y_train, train_ids, test_ids):
    """Generate predictions"""
    
    # monkey prediction
    y_pred = np.random.choice([-1, 1], size=x_test.shape[0])

    return y_pred