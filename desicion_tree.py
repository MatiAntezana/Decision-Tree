import numpy as np
from metrics import *

class Node:
    """
    Decision tree node.

    Args:
        feature (int): Index of the feature used to split this node.
        threshold (float): Threshold used for the feature split.
        right (Node): Right child node.
        left (Node): Left child node.
        value (float): Value assigned to the node.
    """
    def __init__(self, feature=None, threshold=None, right=None, left=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
    
    def exist_node(self):
        """
        Checks if the node has a value assigned.

        Returns:
            bool: True if the node has a value assigned, False otherwise.
        """
        return self.value is not None
    
class DecisionTree:
    """
    Decision tree.

    Args:
        min_num_split (int): Minimum number of samples required to split a node.
        max_depth (int): Maximum depth of the tree.
        n_features (int): Number of features to consider when looking for the best split.
        root (Node): Root of the decision tree.
    """

    def __init__(self, max_depth = 10, min_num_split = 2, n_features=None):
        self.min_num_split = min_num_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def most_common_label(self, Y):
        """
        Returns the most common label in the label array.

        Args:
            Y (numpy.array): Label array.

        Returns:
            float: The most common label in the label array.
        """
        return np.sum(Y == 1) / len(Y)

    def entropy(self, Y):
        """
        Computes the entropy of the label array.

        Args:
            Y (numpy.array): Label array.

        Returns:
            float: Entropy of the label array.
        """
        p_class = np.bincount(Y.astype(int)) / len(Y)
        return -np.sum([prob * np.log(prob) for prob in p_class if prob > 0])
    
    def split(self, X_col, split_threshold):
        """
        Performs the split of a feature column based on a threshold.

        Args:
            X_col (numpy.array): Feature column.
            split_threshold (float): Split threshold.

        Returns:
            tuple: Indices of samples that are less than or equal to the threshold (left),
                   and greater than the threshold (right).
        """
        index_left = np.argwhere(X_col<=split_threshold).flatten()
        index_right = np.argwhere(X_col>split_threshold).flatten()
        return index_left, index_right

    def calc_gain(self, Y, X_col, threshold):
        """
        Calculates the information gain for a node split.

        Args:
            Y (numpy.array): Label array.
            X_col (numpy.array): Feature column.
            threshold (float): Split threshold.

        Returns:
            float: Information gain for the node split.
        """
        parent_entropy = self.entropy(Y)

        index_left, index_right = self.split(X_col, threshold)

        num_left, num_right = len(index_left), len(index_right)

        if num_left == 0 or num_right == 0:
            return 0

        num_y = len(Y)

        entropy_left = self.entropy(Y[index_left])
        entropy_right = self.entropy(Y[index_right])

        son_entropy = (num_left / num_y) * entropy_left + (num_right/num_y) * entropy_right

        return parent_entropy - son_entropy

    def find_best_split(self, X, Y, feat_idxs):
        """
        Finds the best split for a set of features.

        Args:
            X (array_like): Feature matrix.
            Y (array_like): Label matrix.
            feat_idxs (array_like): Indices of the features to consider.

        Returns:
            tuple: Index of the feature for the best split and split threshold.
        """
        best_gain = -1
        idx_split = None
        split_threshold = None

        for idx in feat_idxs:
            thresholds = np.percentile(X[:, idx], q=np.arange(25, 100, 25))

            for th in thresholds:

                gain = self.calc_gain(Y, X[:, idx], th)

                if gain > best_gain:
                    best_gain = gain
                    idx_split = idx
                    split_threshold = th
                
        return idx_split, split_threshold

    def predict(self, X):
        """
        Makes predictions for input samples.

        Args:
            X (array_like): Feature matrix of input samples.

        Returns:
            array_like: Vector of predicted labels for the input samples.
        """
        return np.array([self.initial_tree(x, self.root) for x in X])
    
    def initial_tree(self, X, node):
        """
        Performs the recursive prediction stage of the tree.

        Args:
            X (array_like): Feature vector of an input sample.
            node (Node): Current node in the prediction stage.

        Returns:
            float: Predicted value for the input sample.
        """
        if node.exist_node():
            return node.value

        if X[node.feature] <= node.threshold:
            return self.initial_tree(X, node.left)

        else:
            return self.initial_tree(X, node.right)

    def expand_tree(self, X, Y, depth):
        """
        Expands the decision tree recursively.

        Args:
            X (array_like): Feature matrix.
            Y (array_like): Label matrix.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the expanded decision tree.
        """
        if depth > self.max_depth:
            return Node(value=self.most_common_label(Y))
        
        size_data, cant_features = X.shape
        cant_labels = len(np.unique(Y))

        if self.min_num_split > size_data or cant_labels == 1:
            return Node(value=self.most_common_label(Y))
        
        feat_idxs = np.random.choice(cant_features, self.n_features, replace=False)
        best_feature, best_thresh = self.find_best_split(X, Y, feat_idxs)

        index_left, index_right = self.split(X[:, best_feature], best_thresh)

        node_left = self.expand_tree(X[index_left, :], Y[index_left], depth+1)
        node_right = self.expand_tree(X[index_right, :], Y[index_right], depth+1)

        return Node(feature=best_feature,threshold=best_thresh, right=node_right, left=node_left)

    def fit(self, X, Y):
        """
        Train Desicion Tree.

        Args:
            X (numpy.array): Feature matrix.
            Y (numpy.array): Feature matrix.
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.expand_tree(X, Y, 0)
