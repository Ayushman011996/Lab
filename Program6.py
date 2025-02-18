import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target  
class_names = iris.target_names  

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.priors = np.array([np.mean(y == c) for c in self.classes])

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        scores = [np.log(self.priors[c]) + np.sum(np.log(self._pdf(c, x))) for c in range(len(self.classes))]
        return self.classes[np.argmax(scores)]

    def _pdf(self, class_idx, x):
        mean, var = self.means[class_idx], self.vars[class_idx]
        return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train and predict
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Print results
print(f'Accuracy: {np.mean(y_pred == y_test):.4f}')
print("\nPredictions:", class_names[y_pred])
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
