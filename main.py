import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the LinearRegression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        iterations (int): The number of training iterations. Default is 1000.

        Attributes:
        weights (numpy.ndarray): Model weights for each feature.
        bias (float): Model bias.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y) -> None:
        """
        Train the linear regression model using gradient descent.

        Parameters:
        X (numpy.ndarray): Input features of shape (num_samples, num_features).
        y (numpy.ndarray): Target values of shape (num_samples,).

        Returns:
        None
        """
        num_samples, num_features = X.shape

        # Initialize the weights and bias as zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            # Predict the output using the current weights and bias
            y_pred = self.predict(X)

            # Calculate the gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update the weights and bias using the gradients and learning rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained linear regression model.

        Parameters:
        X (numpy.ndarray): Input features of shape (num_samples, num_features).

        Returns:
        numpy.ndarray: Predicted values for each input sample.
        """
        return np.dot(X, self.weights) + self.bias
