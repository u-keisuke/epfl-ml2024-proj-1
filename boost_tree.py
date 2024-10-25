import numpy as np
import pickle

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

    def calculate_similarity(self, y, prev_predictions, sample_weights):
        residuals = sample_weights * (y - prev_predictions)
        similarity = (residuals.sum())**2 
        similarity /= (np.sum(sample_weights * prev_predictions*(1-prev_predictions)) + self.lambda_regularizer)
        return similarity
    
    def calculate_logodds(self, y, prev_predictions, sample_weights):
        residuals = sample_weights * (y - prev_predictions)
        logodds = (residuals.sum())
        logodds /= (np.sum(sample_weights * prev_predictions*(1-prev_predictions)) + self.lambda_regularizer)
        return logodds

    def make_split(self, feature_index, threshold, X_subset, y_subset, prev_predictions, sample_weights):
        index_left = X_subset[:, feature_index] < threshold
        index_right = X_subset[:, feature_index] >= threshold
        X_left, y_left, prev_predictions_left, sample_weights_L = \
            X_subset[index_left, :], y_subset[index_left], prev_predictions[index_left], sample_weights[index_left]
        X_right, y_right, prev_predictions_right, sample_weights_R = \
            X_subset[index_right, :], y_subset[index_right], prev_predictions[index_right], sample_weights[index_right]
        return (X_left, y_left, prev_predictions_left, sample_weights_L), (X_right, y_right, prev_predictions_right, sample_weights_R)
    

    def choose_best_split(self, X_subset, y_subset, prev_predictions, sample_weights):

        if self.random_state:
            np.random.seed(self.random_state)
        feat_idxs = np.random.choice(len(X_subset[0]), self.max_features, replace=self.replace)
        X_subset = X_subset[:, feat_idxs] #to consider only part of the features

        best_gain = -np.inf
        feature_index = 0
        threshold = 0
        similarity = self.calculate_similarity(y_subset, prev_predictions, sample_weights)
        for i in range(len(X_subset[0])):
            X_featured = X_subset[:, i]
            thresholds = np.unique(X_featured)
            for thread in thresholds:
                pred_L = prev_predictions[X_featured < thread]
                pred_R = prev_predictions[X_featured >= thread] 
                y_L = y_subset[X_featured < thread]
                y_R = y_subset[X_featured >= thread]

                sample_weights_L = sample_weights[X_featured < thread]
                sample_weights_R = sample_weights[X_featured >= thread]
                if not (y_L.any() and y_L.any()):
                    continue
                gain = self.calculate_similarity(y_L, pred_L, sample_weights_L) +  self.calculate_similarity(y_R, pred_R, sample_weights_R) - similarity

                if gain > best_gain and gain>=self.gamma_regularizer:
                    threshold = thread
                    feature_index = i
                    best_gain = gain
        if best_gain<0:
            return None, None # to unpack            
        return feat_idxs[feature_index], threshold
    
    def make_tree(self, X_subset, y_subset, depth, prev_predictions, sample_weights):
        n_labels = len(np.unique(y_subset)) 
        if (depth == 1 or n_labels == 1):
            logodds = self.calculate_logodds(y_subset, prev_predictions, sample_weights)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax()
            
        best_feat, best_thresh = self.choose_best_split(X_subset, y_subset, prev_predictions, sample_weights)
        if best_feat is None: # gain is negative: stop expanding the tree
            logodds = self.calculate_logodds(y_subset, prev_predictions, sample_weights)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax()
        
        new_node = Node(best_feat, best_thresh)
        (X_left, y_left, prev_predictions_Left, sample_weights_L), (X_right, y_right, prev_predictions_Right, sample_weights_R) = \
            self.make_split(best_feat, best_thresh, X_subset, y_subset, prev_predictions, sample_weights)

        if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
            logodds = self.calculate_logodds(y_subset, prev_predictions, sample_weights)
            return Node(feature_index=None, threshold=None, logodds=logodds) #value.argmax(),

        new_node.left_child = self.make_tree(X_left, y_left, depth-1, prev_predictions_Left, sample_weights_L)
        new_node.right_child = self.make_tree(X_right, y_right, depth-1, prev_predictions_Right, sample_weights_R)

        return new_node


    def fit(self, X, y, prev_predictions, sample_weights):

        # y = one_hot_encode(2, y)
        if not self.max_features:
            self.max_features = len(X[0])
        self.root = self.make_tree(X, y, self.max_depth, prev_predictions, sample_weights)
        
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
    def __init__(self,  max_depth=np.inf, min_samples_split=2, 
                 reduce_features=True,
                 random_state=None, 
                 depth=3, num_trees=10,
                 lr=0.3, decay_rate=1, decay_interval=None,
                 lambda_regularizer=0, gamma_regularizer=0):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.depth = depth
        self.root = None 
        self.reduce_features = reduce_features
        self.random_state = random_state
        self.num_trees = num_trees
        self.lambda_regularizer = lambda_regularizer # regularize the scores
        self.gamma_regularizer = gamma_regularizer # regularize the number of leaves
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval

    def fit(self, X, y, sample_weights=None):
        """Training part"""
        Forest = []
        if self.reduce_features:
            max_features = np.around(np.sqrt(len(X[0]))).astype(int)
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        if self.decay_interval is None:
            self.decay_interval = self.num_trees

        prev_prediction = np.random.rand(len(X)) 
        # prev_prediction = np.zeros(len(X)) + 0.5 # any better initializations?
        logodds = prob2logodds(prev_prediction)
        
        lr = self.lr
        for _ in range(0, self.num_trees, self.decay_interval):
            lr = lr * self.decay_rate
            for _ in range(self.decay_interval):
                class_estimator = BoostTree(max_depth=self.max_depth, 
                                                min_samples_split=self.min_samples_split, 
                                                max_features=max_features, 
                                                gamma_regularizer=self.gamma_regularizer, # regularize the number of leaves
                                                lambda_regularizer=self.lambda_regularizer,
                                                replace=False) 

                class_estimator.fit(X, y, prev_prediction, sample_weights=sample_weights)
                logodds += lr * class_estimator.predict_logodds(X)
                logodds = np.clip(logodds, -100, 100)
                prev_prediction = logodss2prob(logodds) # for training the next tree            
                Forest.append((class_estimator, lr))
            print("finished first round of rate decay; numtrees:", self.decay_interval)
        self.Forest = Forest

    def predict(self, X):
        """Test the model"""
        logodds = np.zeros(len(X)) + 0.5
        for Tree, lr in self.Forest:
            logodds += lr * Tree.predict_logodds(X)
        logodds = np.clip(logodds, -100, 100)
        return logodss2prob(logodds)

    def save_model(self, file_path):
        """Save the trained model"""
        with open(file_path, 'wb') as file:
            pickle.dump(self.Forest, file)

    def load_model(self, file_path, num_trees):
        """Load the trained model"""
        with open(file_path, 'rb') as file:
            self.Forest = pickle.load(file)
            assert num_trees <= len(self.Forest), "The trained boost contains fewer trees than you want to unpack"
            self.Forest = self.Forest[:num_trees]
         
