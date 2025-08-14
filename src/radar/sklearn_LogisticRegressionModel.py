from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

class LogisticRegressionModel:
    def __init__(self, n_samples=1000, n_features=20, test_size=0.3, random_state=42):
        # Generate a synthetic dataset
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, 
                                              n_informative=15, n_redundant=5, random_state=random_state)
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        # Initialize the model
        self.model = LogisticRegression()
    
    def train_model(self):
        """Train the logistic regression model."""
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate the model's performance on the test set."""
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report

    def get_performance(self):
        """Train and evaluate the model, then return the results."""
        self.train_model()
        accuracy, report = self.evaluate_model()
        return accuracy, report

    def classify_input(self, input_data):
        """Classify a new input into one of the binary classes."""
        # Make prediction for the input data
        prediction = self.model.predict([input_data])
        
        # Return the predicted class (0 or 1)
        return prediction[0]

# Example of using the class
if __name__ == "__main__":
    # Create an instance of the model
    lr_model = LogisticRegressionModel()

    # Train and evaluate the model, then print results
    accuracy, report = lr_model.get_performance()
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Example input data to classify (a single sample with 20 features)
    new_input = [0.1, -0.3, 0.5, 0.8, -0.7, 0.4, -0.2, 0.9, 0.2, -0.5, 
                 0.6, -0.4, 0.3, -0.1, 0.7, -0.6, 0.1, -0.9, 0.8, 0.0, 0.4, -0.3]

    # Classify the new input
    predicted_class = lr_model.classify_input(new_input)
    print(f"The predicted class for the input is: {predicted_class}")
