import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and target labels
class_names = iris.target_names  # Names of the classes

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

class KNN:
    """
    A simple k-Nearest Neighbors (k-NN) classifier.
    """
    def __init__(self, k=3):
        self.k = k  # Number of neighbors to consider 

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class labels for the given test data."""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """Predict the class label for a single test example."""
        # Compute the Euclidean distances to all training examples
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Determine the most common label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

# Create a k-NN classifier with k=3
knn = KNN(k=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = knn.predict(X_test)

# Calculate and display the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')

# Display the predictions in terms of class names
predicted_classes = [class_names[label] for label in y_pred]
print("Predictions:", predicted_classes)

# Generate and display the confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
