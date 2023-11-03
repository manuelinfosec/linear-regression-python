# Linear Regression From Scratch with Python

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
```

The above piece defines LinearRegression class to encapsulate the functionality of linear regression. 

The class has 3 methods: `__init__`, `fit` and `predict`.

Initializing with the learning rate, number of iterations, the weights and bias attributes to store the model parameters in the `__init__` method.

Define the `fit` method that does the training of the model using gradient descent. 

The method takes input data `X` and the corresponding target values, `y`.

Inside the method, first determine the number of samples and features in the input data.


```python
def fit(self, X, y):
    num_samples, num_features = X.shape

    # Initialize the weights and bias as zeros
    self.weights = np.zeros(num_features)
    self.bias = 0
```

Also, initialize the weights and bias to zeros using `np.zeros()` against the number of features observed from `X`.

Iteration is done over the specified number of iterations performing the following steps:

1. Predict the output values `y_pred` using current weights and bias. This is done by calling the `predict` method.

```python
# Gradient descent
for _ in range(self.iterations):
    # Predict the output using the current weights and bias
    y_pred = self.predict(X)
```


2. Calculate the gradiens/slope of weights and bias using the following formulas:
    - `dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))`
    - `db = (1 / num_samples) * np.sum(y_pred - y)`

Here `(y_pred - y)` is the difference between the predicted and the actual values.

```python
# Calculate the gradients
dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
db = (1 / num_samples) * np.sum(y_pred - y)
```

3. Update the weights and bias using the gradients and the learning rate:
    - `self.weights -= learning_rate * dw`
    - `self.bias -= self.learning_rate * db`

```python
# Update the weights and bias using the gradients and learning rate
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```

Finally, the predict method:

This method takes the input data `X` and returns the predicted output by multiplying the input data with the weights and adding the bias term.

```python
def predict(self, X) -> np.ndarray:
    return np.dot(X, self.weights) + self.bias
```

Linear Regression implemented from scratch. If this helped, star the repository or reach out to me on [X/Twitter](www.twitter.com/manuelinfosec/).