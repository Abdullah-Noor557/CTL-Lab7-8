# CTL-Lab7-8
Computer Tools Lab 7-8 Task in which we try to improve a machine learning model by atleast 2%, this project mainly aims to get familiar with collaborative codebases, work with different branches etc.

Group Members:
 - Abdullah Noor    (2022029)
 - Muhammad Mustafa (2022405)

# Model Chosen
We are choosing the **LogisticRegression** model on the breast cancer dataset from **sklearn.datasets**, this model in our case had an accuracy of 97.66% after improvement our accuracy reached 99.42%.


# Results
 - Base Model Accuracy :                         97.66%
 - Improvement (Feature Scaling) Accuracy:       98.25%
 - Improvement (Hyperparameter Tuning) Accuracy: 99.42%


# 1. Feature Scaling
Reason for Use:

Normalization of Feature Magnitude: Machine learning models, particularly those that compute distances between data points or that use gradient descent as an optimization algorithm (like logistic regression), can be sensitive to the scale of the features. If features are on very different scales, it can lead to skewed or biased results and significantly longer training times.
Improved Convergence: Scaling features to a common scale can help the optimization algorithm converge more quickly, which improves the efficiency of training the model.
Reasoning:

Logistic regression can perform better and converge faster when features are on similar scales because the weight updates in gradient descent are more uniform across all features.
This was essential for ensuring that no feature unduly influences the model's predictions due to its scale.

# 2. Hyperparameter Tuning
Reason for Use:

Optimizing Model Performance: Hyperparameters control the behavior of the training algorithm and can significantly affect the performance of a machine learning model. Unlike model parameters, hyperparameters are not learned from the data automatically and must be set prior to training.
Customization to Specific Data: Tuning hyperparameters helps tailor the model to the specific characteristics of the dataset, potentially enhancing model accuracy and preventing overfitting.
Reasoning:

For logistic regression, the regularization strength (C) and the choice of solver can impact how well the model generalizes. Different values of C can control the trade-off between achieving lower training error and a lower test error, which is crucial for avoiding overfitting.
Grid search provides a systematic way to test combinations of multiple hyperparameter values, ensuring that the best possible model configuration is identified based on the performance metric defined (e.g., accuracy).
