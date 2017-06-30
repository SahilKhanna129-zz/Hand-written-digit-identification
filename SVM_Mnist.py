import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import tensorflow.examples.tutorials.mnist.input_data as data

# Data object with training data as 55000*784 images and 55000*1 labels
# and testing data as 10000*784 images and 10000*1 labels
mnist = data.read_data_sets("MNIST")

# Store training data in X and Y matrix
X_train, y_train = mnist.train.images, mnist.train.labels

# Split data into validation set and testing set
X_valid, X_test, y_valid, y_test = train_test_split(mnist.test.images, mnist.test.labels, test_size = 0.5, random_state = 42)

# Parameter selections for SVM
parameter_c, parameter_gamma = 5, 0.05

# Selecting RBF classifier for calculating similarity
classifier = svm.SVC(C = parameter_c, gamma = parameter_gamma)

#classifier = svm.LinearSVC()

#classifier = svm.SVC(C = parameter_c, kernel = 'poly')

# Train the classifier
classifier.fit(X_train, y_train)

# Predict the labels with test samples
y_test_predict = classifier.predict(X_test)

# Predict the labels with validation samples
y_valid_predict = classifier.predict(X_valid)

# Print both the accuracies
print("Accuracy with test samples: {0}".format(metrics.accuracy_score(y_test, y_test_predict)))
print("Accuracy with validation samples: {0}".format(metrics.accuracy_score(y_valid, y_valid_predict)))

