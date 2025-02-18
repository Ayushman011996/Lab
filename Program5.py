import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression with Gradient Descent
def train_logistic_regression(X, y, lr=0.001, epochs=200):
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(epochs):
        weights -= lr * np.dot(X.T, (sigmoid(np.dot(X, weights)) - y)) / len(y)
    return weights

# Load and prepare data
iris = load_iris()
X, y = iris.data[:, :2], (iris.target != 0).astype(int)  # Binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)

# Standardize features
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Train model
weights = train_logistic_regression(X_train, y_train)

# Predictions
y_pred = sigmoid(np.dot(X_test, weights)) > 0.5
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')

# Decision boundary
x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
y_values = -(weights[0] * x_values) / weights[1]  # Equation of line

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.plot(x_values, y_values, color='black', linewidth=2)  # Decision boundary
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
