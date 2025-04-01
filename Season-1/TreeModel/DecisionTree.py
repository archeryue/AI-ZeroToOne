import numpy as np
import pandas as pd
from typing import Union, Tuple

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        if task not in ['classification', 'regression']:
            raise ValueError("task must be either 'classification' or 'regression'")
        self.task = task
        self.classes_ = None

    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.is_leaf = False

    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error for regression"""
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for classification"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity (MSE for regression, Gini for classification)"""
        if self.task == 'regression':
            return self._calculate_mse(y)
        else:
            return self._calculate_gini(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split using appropriate criterion"""
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                # Weighted average of impurity
                current_score = (np.sum(left_mask) * left_impurity + np.sum(right_mask) * right_impurity) / n_samples
                
                if current_score < best_score:
                    best_score = current_score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_score

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree"""
        node = self.Node()
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            node.is_leaf = True
            if self.task == 'regression':
                node.value = np.mean(y)
            else:
                node.value = self.classes_[np.argmax(np.bincount(y))]
            return node

        # Find the best split
        feature, threshold, score = self._find_best_split(X, y)
        
        if feature is None:  # No valid split found
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Create child nodes
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the decision tree"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
            
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _predict_single(self, x: np.ndarray, node: Node) -> Union[float, str]:
        """Predict for a single sample"""
        if node.is_leaf:
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict for multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return np.array([self._predict_single(x, self.tree) for x in X])
