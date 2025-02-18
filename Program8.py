import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features (sepal length, sepal width)

def kmeans(X, k, max_iterations=10000):
    # Randomly initialize centroids by selecting k random points
    # centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    centroids = X[:k]
    
    for _ in range(max_iterations):
        # Calculate distances to centroids and assign labels
        labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)
        
        # Update centroids as the mean of points in each cluster
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    return centroids, labels

# Number of clusters
k = 3

# Perform K-means clustering
centroids, labels = kmeans(X, k)

# Plot clusters with different colors
plt.scatter(X[:, 0], X[:, 1], c=labels)

# Plot centroids as black 'x'
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Centroids')

plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
