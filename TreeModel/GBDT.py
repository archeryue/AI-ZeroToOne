import numpy as np
from DecisionTree import DecisionTree

class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize the Gradient Boosting Decision Tree for regression.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages (trees) to perform
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of each decision tree
        min_samples_split : int
            Minimum number of samples required to split an internal node
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        # Initialize predictions with zeros
        F = np.zeros_like(y, dtype=np.float64)
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - F
            
            # Fit a tree on residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task='regression'
            )
            tree.fit(X, residuals)
            
            # Update predictions
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict regression target for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    def score(self, X, y):
        """
        Return the coefficient of determination R^2.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R^2 score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
