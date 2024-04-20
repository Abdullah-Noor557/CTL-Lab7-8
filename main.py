from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating the   logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, predictions)
print(f"Initial Accuracy: {initial_accuracy * 100:.2f}%")

# Improving the model with Feature Scaling.

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model_scaled = LogisticRegression(max_iter=10000)
model_scaled.fit(X_train_scaled, y_train)

# Predict and evaluate the model
predictions_scaled = model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predictions_scaled)
print(f"Accuracy after Scaling: {accuracy_scaled * 100:.2f}%")


# Improving the model with HyperParameterTuning, Hyperparameter tuning involves finding the combination of hyperparameters for a model that provides the best performance as measured on a validation set.

# Define the model and parameters
model = LogisticRegression(max_iter=10000)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Algorithm to use in the optimization problem
}

# Create GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
predictions_tuned = best_model.predict(X_test_scaled)
accuracy_tuned = accuracy_score(y_test, predictions_tuned)
print(f"Accuracy after Hyperparameter Tuning: {accuracy_tuned * 100:.2f}%")
