import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot

def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

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

    def calculate_similarity(self, residuals):
        similarity = residuals.sum()**2 
        similarity /= len(residuals) + self.lambda_regularizer
        return similarity

    def make_split(self, feature_index, threshold, X_subset, y_subset, residuals):
        index_left = X_subset[:, feature_index] < threshold
        index_right = X_subset[:, feature_index] >= threshold
        X_left, y_left, residualsleft = X_subset[index_left, :], y_subset[index_left], residuals[index_left]
        X_right, y_right, residuals_right = X_subset[index_right, :], y_subset[index_right], residuals[index_right]
        return (X_left, y_left, residualsleft), (X_right, y_right, residuals_right)
    

    def choose_best_split(self, X_subset, y_subset, residuals):
        # Bruteforce selection. Can we do better?
        if self.random_state:
            np.random.seed(self.random_state)
        feat_idxs = np.random.choice(len(X_subset[0]), self.max_features, replace=self.replace)
        X_subset = X_subset[:, feat_idxs]

        best_gain = -np.inf
        feature_index = 0
        threshold = 0
        
        similarity = self.calculate_similarity(residuals)
        for i in range(len(X_subset[0])):
            X_featured = X_subset[:, i]
            thresholds = np.unique(X_featured)
            for thread in thresholds:
                L = residuals[X_featured < thread]
                R = residuals[X_featured >= thread] 
                if not (R.any() and L.any()):
                    continue
                gain = self.calculate_similarity(L) +  self.calculate_similarity(R) - similarity

                if gain > best_gain and gain>=self.gamma_regularizer:
                    threshold = thread
                    feature_index = i
                    best_gain = gain
        if best_gain<0:
            return None, None #to unpack            
        return feat_idxs[feature_index], threshold
    
    def make_tree(self, X_subset, y_subset, depth, residuals):
        n_labels = len(np.unique(y_subset)) 
        if (depth == 1 or n_labels == 1):
            value = np.sum(y_subset, axis=0)
            similarity = self.calculate_similarity(residuals)
            return Node(feature_index=None, threshold=value.argmax(), logodds=similarity)
            
        best_feat, best_thresh = self.choose_best_split(X_subset, y_subset, residuals)
        if not best_feat: # gain is negative: stop expanding the tree
            value = np.sum(y_subset, axis=0)
            similarity = self.calculate_similarity(residuals)
            return Node(feature_index=None, threshold=value.argmax(), logodds=similarity)
        
        new_node = Node(best_feat, best_thresh)
        (X_left, y_left, residuals_Left), (X_right, y_right, residuals_Right) = \
            self.make_split(best_feat, best_thresh, X_subset, y_subset, residuals)

        if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
            value = np.sum(y_subset, axis=0)
            similarity = self.calculate_similarity(residuals)
            return Node(feature_index=None, threshold=value.argmax(), logodds=similarity)

        new_node.left_child = self.make_tree(X_left, y_left, depth-1, residuals_Left)
        new_node.right_child = self.make_tree(X_right, y_right, depth-1, residuals_Right)

        return new_node


    def fit(self, X, y, previous_predictions):
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'

        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        y = one_hot_encode(self.n_classes, y)
        if not self.max_features:
            self.max_features = len(X[0])
        residuals = np.sum(y, axis=1) - previous_predictions
        self.root = self.make_tree(X, y, self.max_depth, residuals)
    
    def predict(self, X):
        pred = np.zeros(len(X))
        for i in range(len(X)): 
            pred[i] = self.pass_tree(X[i, :], self.root)
        return pred
        
    def predict_logodds(self, X):
        pred_logodds = np.zeros((len(X)))
        for i in range(len(X)): 
            pred_logodds[i] = self.pass_tree(X[i, :], self.root, logodds_include=True)
        return pred_logodds
    
    def pass_tree(self, X, node, logodds_include=False):
        if node.feature_index is None:
            if not logodds_include:
                return node.value
            else:
                return node.logodds

        if X[node.feature_index] < node.value:
            return self.pass_tree(X, node.left_child, logodds_include=logodds_include)
        else:
            return self.pass_tree(X, node.right_child, logodds_include=logodds_include)
    

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
        
        self.n_classes = n_classes
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
        self.n_classes = len(np.unique(y))

        if self.reduce_features:
            max_features = np.around(np.sqrt(len(X[0]))).astype(int)
        
        # previous_prediction = np.random.rand(len(X)) # any better initializations?
        previous_prediction = np.zeros(len(X)) + 0.5
        logodds = prob2logodds(previous_prediction)
        
        for _ in range(self.num_trees):
            class_estimator = BoostTree(n_classes=self.n_classes, max_depth=self.max_depth, 
                                            min_samples_split=self.min_samples_split, 
                                            max_features=max_features, 
                                            replace=False) 



            class_estimator.fit(X, y, previous_prediction)
            logodds += self.learning_rate * class_estimator.predict_logodds(X)
            previous_prediction = logodss2prob(logodds) # for training the next tree
            Forest.append((class_estimator, self.learning_rate))
        self.Forest = Forest

    def predict(self, X):
        logodds = np.zeros(len(X)) + 0.5
        for Tree, lr in self.Forest:
            logodds = + lr * Tree.predict_logodds(X)
            print(logodds)
        return logodss2prob(logodds)
         
