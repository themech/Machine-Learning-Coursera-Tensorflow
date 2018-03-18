# This classifies the same digit as the logistic regression classifiers from
# the first step. But here we're using a pre-trained neural network classifier
# (loaded from data/ex3weights.mat)
import numpy as np
from scipy import io
from sklearn import metrics

# Load the data.
filename = 'data/ex3data1.mat'
data = io.loadmat(filename)
X_data, Y_data = data['X'], data['y']

numSamples = X_data.shape[0]

# Add a 'constant' to each of the rows.
X_data = np.insert(X_data, 0, 1, axis=1)

print("X_data shape ", X_data.shape, ", Y_data shape", Y_data.shape)

# Load the pre-trained network.
weights = io.loadmat('data/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

print("Theta1 shape", theta1.shape, ", theta2 shape", theta2.shape)

# Classify the input data using the pre-trained network/
a1 = X_data
z2 = np.matmul(a1, np.transpose(theta1))  # (5000,401) @ (25,401).T = (5000,25)
print("z2 shape", z2.shape)

z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


a2 = sigmoid(z2)
print("a2 shape", a2.shape)  # (5000, 26)

z3 = np.matmul(a2, np.transpose(theta2))
print("z3 shape", z3.shape)  # (5000, 10)

a3 = sigmoid(z3)

# Numpy is 0 base index. We add +1 to make it compatible with matlab (so we can
# compare y_pred with the correct answers from Y_data).
y_pred = np.argmax(a3, axis=1) + 1
print("y_pred shape", y_pred.shape)  # (5000,)

# Print the report
print(metrics.classification_report(Y_data, y_pred))
