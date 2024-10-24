import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder


# def one_hot_encode(n_classes, y):
#     y_one_hot = np.zeros((len(y), n_classes), dtype=float)
#     y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
#     return y_one_hot

# def one_hot_decode(y_one_hot):
#     return y_one_hot.argmax(axis=1)[:, None]

def prob2logodds(p):
    return np.log(p/(1-p))

def logodss2prob(log_odds):
    return np.exp(log_odds)/(1+np.exp(log_odds))


class Node:
    def __init__(self, feature_index, threshold, logodds=None):
        self.feature_index = feature_index
        self.value = threshold
        self.logodds = logodds
        self.left_child = None
        self.right_child = None 
       
        
class BoostTree():
    """
    change the VALUES in the leaves: in case of 
    """
    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 max_features=None, 
                 random_state=None, replace=False, 
                 gamma_regularizer=0, lambda_regularizer=0):
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.max_features = max_features
        self.random_state = random_state
        self.replace = replace
        self.gamma_regularizer = gamma_regularizer
        self.lambda_regularizer = lambda_regularizer

    def calculate_similarity(self, y, prev_predictions):
        residuals = y - prev_predictions
        similarity = (residuals.sum())**2 
        similarity /= (np.sum(prev_predictions*(1-prev_predictions)) + self.lambda_regularizer)

        # print("similarity2", similarity)
        return similarity

    def calculate_logodds(self, y, prev_predictions):
        residuals = y - prev_predictions
        logodds = (residuals.sum())
        logodds /= (np.sum(prev_predictions*(1-prev_predictions)) + self.lambda_regularizer)
        return logodds

    def make_split(self, feature_index, threshold, X_subset, y_subset, prev_predictions):
        index_left = X_subset[:, feature_index] < threshold
        index_right = X_subset[:, feature_index] >= threshold
        X_left, y_left, prev_predictions_left = X_subset[index_left, :], y_subset[index_left], prev_predictions[index_left]
        X_right, y_right, prev_predictions_right = X_subset[index_right, :], y_subset[index_right], prev_predictions[index_right]
        return (X_left, y_left, prev_predictions_left), (X_right, y_right, prev_predictions_right)
    

    def choose_best_split(self, X_subset, y_subset, prev_predictions):
        # Bruteforce selection. Can we do better?
        if self.random_state:
            np.random.seed(self.random_state)
        feat_idxs = np.random.choice(len(X_subset[0]), self.max_features, replace=self.replace)
        X_subset = X_subset[:, feat_idxs]

        best_gain = -np.inf
        feature_index = 0
        threshold = 0
        similarity = self.calculate_similarity(y_subset, prev_predictions)
        for i in range(len(X_subset[0])):
            X_featured = X_subset[:, i]
            thresholds = np.unique(X_featured)
            for thread in thresholds:
                pred_L = prev_predictions[X_featured < thread]
                pred_R = prev_predictions[X_featured >= thread] 
                y_L = y_subset[X_featured < thread]
                y_R = y_subset[X_featured >= thread]
                if not (y_L.any() and y_L.any()):
                    continue
                gain = self.calculate_similarity(y_L, pred_L) +  self.calculate_similarity(y_R, pred_R) - similarity

                if gain > best_gain and gain>=self.gamma_regularizer:
                    threshold = thread
                    feature_index = i
                    best_gain = gain
        if best_gain<0:
            return None, None # to unpack            
        return feat_idxs[feature_index], threshold
    
    def make_tree(self, X_subset, y_subset, depth, prev_predictions):
        n_labels = len(np.unique(y_subset)) 
        if (depth == 1 or n_labels == 1):
            # value = np.sum(y_subset, axis=0)
            logodds = self.calculate_logodds(y_subset, prev_predictions)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax()
            
        best_feat, best_thresh = self.choose_best_split(X_subset, y_subset, prev_predictions)
        if not best_feat: # gain is negative: stop expanding the tree
            # value = np.sum(y_subset, axis=0)
            logodds = self.calculate_logodds(y_subset, prev_predictions)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax()
        
        new_node = Node(best_feat, best_thresh)
        (X_left, y_left, prev_predictions_Left), (X_right, y_right, prev_predictions_Right) = \
            self.make_split(best_feat, best_thresh, X_subset, y_subset, prev_predictions)

        if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
            # value = np.sum(y_subset, axis=0)
            logodds = self.calculate_logodds(y_subset, prev_predictions)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax(),

        new_node.left_child = self.make_tree(X_left, y_left, depth-1, prev_predictions_Left)
        new_node.right_child = self.make_tree(X_right, y_right, depth-1, prev_predictions_Right)

        return new_node


    def fit(self, X, y, prev_predictions):

        # y = one_hot_encode(2, y)
        if not self.max_features:
            self.max_features = len(X[0])
        self.root = self.make_tree(X, y, self.max_depth, prev_predictions)
        
    def predict_logodds(self, X):
        pred_logodds = np.zeros((len(X)))
        for i in range(len(X)): 
            pred_logodds[i] = self.pass_tree(X[i, :], self.root)
        return pred_logodds
    
    def pass_tree(self, X, node):
        if node.feature_index is None:
            return node.logodds

        if X[node.feature_index] < node.value:
            return self.pass_tree(X, node.left_child)
        else:
            return self.pass_tree(X, node.right_child)
    

class BoostForest():
    """
    add regularizer???
    """

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 reduce_features=True,
                 random_state=None, classif_type="common",
                 depth=3, num_trees=10,
                 learning_rate=0.3,
                 lambda_regularizer=0, gamma_regularizer=0):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.depth = depth
        self.root = None # Use the Node class to initialize it later
        self.reduce_features = reduce_features
        self.random_state = random_state
        self.num_trees = num_trees
        assert classif_type in {"common", "proba", "both"}
        self.classif_type = classif_type
        self.lambda_regularizer = lambda_regularizer # regularize the scores
        self.gamma_regularizer = gamma_regularizer # regularize the number of leaves
        self.learning_rate = learning_rate

    def fit(self, X, y):
        Forest = []

        if self.reduce_features:
            max_features = np.around(np.sqrt(len(X[0]))).astype(int)
        
        # prev_prediction = np.random.rand(len(X)) # any better initializations?
        prev_prediction = np.zeros(len(X)) + 0.5
        logodds = prob2logodds(prev_prediction)
        
        for _ in range(self.num_trees):
            # print("prev_prediction", prev_prediction, np.all(prev_prediction>=0))
            class_estimator = BoostTree(max_depth=self.max_depth, 
                                            min_samples_split=self.min_samples_split, 
                                            max_features=max_features, 
                                            gamma_regularizer=self.gamma_regularizer, # regularize the number of leaves
                                            lambda_regularizer=self.lambda_regularizer,
                                            replace=False) 

            class_estimator.fit(X, y, prev_prediction)
            logodds += self.learning_rate * class_estimator.predict_logodds(X)

            logodds = np.clip(logodds, -100, 100)
            prev_prediction = logodss2prob(logodds) # for training the next tree
            # print(np.all(prev_prediction>=0) * np.all(prev_prediction<=1))
            
            Forest.append((class_estimator, self.learning_rate))
        self.Forest = Forest

    def predict(self, X):
        logodds = np.zeros(len(X)) + 0.5
        for Tree, lr in self.Forest:
            logodds += lr * Tree.predict_logodds(X)
        logodds = np.clip(logodds, -100, 100)
        return logodss2prob(logodds)
         
