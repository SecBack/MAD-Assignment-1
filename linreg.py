import numpy as np

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        self.w: np.ndarray = np.empty(0)
            
    def fit(self, X: np.ndarray, t: np.ndarray) -> None:
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        
        # make sure X is 2D: shape (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # augment X with a column of 1s for w_0
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        X_aug = np.hstack((ones, X))

        # solve the equation for the weights
        self.w = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ t

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     
        # make sure X is 2D: shape (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1) 

        # augment X with a column of 1s for w_0
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        X_aug = np.hstack((ones, X))

        # simply multiplying the sample by the weights give the prediction
        return X_aug @ self.w