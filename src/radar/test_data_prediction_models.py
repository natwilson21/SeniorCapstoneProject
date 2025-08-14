import joblib
import numpy as np  # Make sure to import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestBinaryClassifier:
    def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)

    def fit(self, X, y):
        # Convert X to numpy array if it's a list
        X = np.array(X)  # Ensure X is a numpy array

        # Ensure X is 2D, reshape if necessary
        if len(X.shape) == 1:  # If X is a 1D array
            X = X.reshape(-1, 1)  # Reshape X into 2D (n_samples, n_features)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Convert labels to numerical values (0 for 'other', 1 for 'human')
        self.y_test = [1 if label == 'human' else 0 for label in y_test]

        # Train the model
        self.model.fit(X_train, [1 if label == 'human' else 0 for label in y_train])

        # Store test data for evaluation
        self.X_test = X_test

        # Store the original training data for comparison
        self.X_train = X_train

    def predict(self, X):
        # Convert X to numpy array if it's a list
        X = np.array(X)  # Ensure X is a numpy array

        # Ensure X is 2D, reshape if necessary
        if len(X.shape) == 1:  # If X is a 1D array
            X = X.reshape(-1, 1)  # Reshape X into 2D (n_samples, n_features)

        return self.model.predict(X)

    def evaluate(self, y_pred, y_true):
        # Evaluate the model
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def predict_labels(self, y_pred):
        # Convert predictions back to labels (0 for 'other', 1 for 'human')
        return ['human' if pred == 1 else 'other' for pred in y_pred]

    def compare_with_database(self, input_data):
        """
        Compare an input list with the training data (X_train) and return 'human' if any number in input_data matches
        any number in any entry of X_train. Otherwise, return '0'.
        """
        # We need to check if any of the rows in X_train contain any of the input_data values.
        for row in self.X_train:
            if any(num in row for num in input_data):  # If any input_data value is found in the row
                return "human"
        
        # If no match is found in any row, return 0
        return 0

    def save_model(self, filename='avgdist.pkl'):
        """Save the trained model and X_train to a file."""
        joblib.dump({'model': self.model, 'X_train': self.X_train}, filename)
        print(f"Model and training data saved to {filename}")

# Example usage:
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    X = [1, 2, .487717, .5252183]  # 1D array as input features
    y = ['other', 'other', 'human', 'human']  # Updated target labels for binary classification

    # Initialize classifier
    clf = RandomForestBinaryClassifier(n_estimators=100)

    # Fit the model
    clf.fit(X, y)

    # Save the model
    clf.save_model('avgdist.pkl')

    # Make predictions on the test data
    y_pred = clf.predict(clf.X_test)

    # Evaluate the model with the predicted values and actual values
    accuracy = clf.evaluate(y_pred, clf.y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Convert predictions to labels
    y_pred_labels = clf.predict_labels(y_pred)
    print("Predictions (labels):", y_pred_labels)
