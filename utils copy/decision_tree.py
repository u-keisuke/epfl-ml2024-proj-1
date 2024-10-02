import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder


def entropy(y):  
    EPS = 0.0005  
    return - np.sum(y * np.log(y + EPS))
    
def gini(y):
    return 1 - np.sum(y * y)
    
def variance(y):
    mean = np.mean(y)
    return np.mean((y - mean) * (y - mean))

def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot

def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False, max_features=None, 
                 random_state=None, replace=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
        self.max_features = max_features
        self.random_state = random_state
        self.replace = replace

        
    def make_split(self, feature_index, threshold, X_subset, y_subset):

        index_left = X_subset[:, feature_index] < threshold
        index_right = X_subset[:, feature_index] >= threshold
        X_left, y_left = X_subset[index_left, :], y_subset[index_left]
        X_right, y_right = X_subset[index_right, :], y_subset[index_right]
        
        return (X_left, y_left), (X_right, y_right)
    

    def choose_best_split(self, X_subset, y_subset):
        H = self.criterion
        
        # Bruteforce selection. Can we do better?
        if self.random_state:
            np.random.seed(self.random_state)
        feat_idxs = np.random.choice(len(X_subset[0]), self.max_features, replace=self.replace)
        X_subset = X_subset[:, feat_idxs]

        best_loss = np.inf
        feature_index = 0
        threshold = 0
        for i in range(len(X_subset[0])):
            X_featured = X_subset[:, i]
            thresholds = np.unique(X_featured)
            for thread in thresholds:
                L = y_subset[X_featured < thread]
                R = y_subset[X_featured >= thread]               
                if not (R.any() and L.any()):
                    continue
                if self.classification:
                    hist_R = np.sum(R, axis=0)
                    hist_L = np.sum(L, axis=0)
                    loss = len(R) / len(y_subset) * H( hist_R / len(R)) +  len(L) / len(y_subset) * H(hist_L / len(L))
                else:
                    loss = len(R) / len(y_subset) * H(R) + len(L) / len(y_subset) * H(L) 
                if loss < best_loss:
                    threshold = thread
                    feature_index = i
                    best_loss = loss
        return feat_idxs[feature_index], threshold
    
    def make_tree(self, X_subset, y_subset, depth):
        n_labels = len(np.unique(y_subset)) 

        if (depth == 1 or n_labels == 1):
            if self.classification:
                value = np.sum(y_subset, axis=0)
                proba = value / len(y_subset)
                return Node(feature_index=None, threshold=value.argmax(), proba=proba)
            elif self.criterion_name == "variance":
                return Node(feature_index=None, threshold=np.mean(y_subset))
            elif self.criterion_name == "mad_median":
                return Node(feature_index=None, threshold=np.median(y_subset))  
            

        best_feat, best_thresh = self.choose_best_split(X_subset, y_subset)
        new_node = Node(best_feat, best_thresh)
        (X_left, y_left), (X_right, y_right) = self.make_split(best_feat, best_thresh, X_subset, y_subset)

        if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
            if self.classification:
                value = np.sum(y_subset, axis=0)
                proba = value / len(y_subset)
                return Node(feature_index=None, threshold=value.argmax(), proba=proba)
            elif self.criterion_name == "variance":
                return Node(feature_index=None, threshold=np.mean(y_subset))
            elif self.criterion_name == "mad_median":
                return Node(feature_index=None, threshold=np.median(y_subset))
            
        new_node.left_child = self.make_tree(X_left, y_left, depth-1)
        new_node.right_child = self.make_tree(X_right, y_right, depth-1)

        return new_node
            

    def fit(self, X, y):
        
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        if not self.max_features:
            self.max_features = len(X[0])

        self.root = self.make_tree(X, y, self.max_depth)
    
    def predict(self, X):
        pred = np.zeros(len(X))
        for i in range(len(X)): 
            pred[i] = self.pass_tree(X[i, :], self.root)
        return pred
        
    def predict_proba(self, X):
        assert self.classification, 'Available only for classification problem'
        pred_prob = np.zeros((len(X), self.n_classes))
        for i in range(len(X)): 
            pred_prob[i] = self.pass_tree(X[i, :], self.root, prob_include=True)
        return pred_prob
    
    def pass_tree(self, X, node, prob_include=False):
        if node.feature_index is None:
            if not prob_include:
                return node.value
            else:
                return node.proba

        if X[node.feature_index] < node.value:
            return self.pass_tree(X, node.left_child, prob_include=prob_include)
        else:
            return self.pass_tree(X, node.right_child, prob_include=prob_include)


class RandomForest(BaseEstimator):
    clasification_loss = {
        'gini': True, # (criterion, classification flag)
        'entropy': True,
        'variance': False,
        'mad_median': False
    }
    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False, reduce_features=True,
                 random_state=None, num_trees=10, classif_type="common" ):
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
        self.reduce_features = reduce_features
        self.random_state = random_state
        self.num_trees = num_trees
        assert classif_type in {"common", "proba", "both"}
        self.classif_type = classif_type

    def fit(self, X,y):
        Forest = []
        self.classification = self.clasification_loss[self.criterion_name]
        if self.classification and self.n_classes is None:
                self.n_classes = len(np.unique(y))

        if self.reduce_features:
            max_features = np.around(np.sqrt(len(X[0]))).astype(int)
        
        for _ in range(self.num_trees):
            class_estimator = DecisionTree(n_classes=self.n_classes, max_depth=self.max_depth, 
                                            min_samples_split=self.min_samples_split, 
                                            criterion_name=self.criterion_name, 
                                            max_features=max_features, 
                                            replace=False) # random_state=None, 

            objects_idxs = np.random.choice(len(X[:, 0]), len(X[:, 0]), replace=True)
            X_subset = X[objects_idxs, :]
            y_subset = y[objects_idxs]

            class_estimator.fit(X_subset, y_subset)
            Forest.append(class_estimator)
        self.Forest = Forest

    def predict(self, X):
        if self.classification:
            prediction = np.zeros((len(X), self.n_classes))
            # #by probabilities
            if self.classif_type == "proba":
                for Tree in self.Forest:
                    prediction = + Tree.predict_proba(X)
            # by most common label
            elif self.classif_type == "common":
                for Tree in self.Forest:
                    labels = Tree.predict(X).astype(int)
                    prediction[np.arange(len(X[:,0])), labels] += 1
            elif self.classif_type == "both":
                for Tree in self.Forest:
                    prediction = + Tree.predict_proba(X)

                prediction1 = np.zeros((len(X), self.n_classes))
                for Tree in self.Forest:
                    labels = Tree.predict(X).astype(int)
                    prediction1[np.arange(len(X[:,0])), labels] += 1
                pred1 = pred = np.argmax(prediction1, axis=1)
            pred = np.argmax(prediction, axis=1)
        else:    
            pred = np.zeros(len(X))
            for Tree in self.Forest:
                pred += Tree.predict(X)
            pred = pred / self.num_trees   
        if self.classif_type == "both":
            return pred, pred1
        return pred
         
