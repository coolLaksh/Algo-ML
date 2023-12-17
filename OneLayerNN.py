import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import math as mth
import tensorflow

def init_params(X, labels):
    # Need to define weights matrix for the two layers and bias matrix for the same
    W1 = np.random.randn(X.shape[0], 2)
    W2 = np.random.randn(2, 1)
    b1 = np.random.randn(2).reshape(2,1)
    b2 = np.random.randn(1).reshape(1,1)

    return W1, W2, b1, b2

def sigmoid(Y):
    return 1 / (1 + np.exp(-Y))


def forward_Propagation(X, labels):
    # Since we are calcualting forward propgation using one hidden layer
    # we have to compute both Z1, A1,  Z2, A2
    # Need to keep in the mind we are using exapmles as a column vector

    W1, W2, b1, b2 = init_params(X, labels)
    Z1 = np.dot(W1.T, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2.T, A1) + b2
    A2 = sigmoid(Z2)
    return A1, Z1, Z2, A2, W1, W2, b1, b2


def back_Propagation(A0, A1, Z1, A2, Z2, labels, W1, W2, b1, b2, alpha):
    cost_function = -labels/A2 + (1-labels)/(1-A2)

    dW2 = (A2 - labels) * A1
    db2 = (A2 - labels)

    dW1 = (A2 - labels) * np.dot(W2.T, A1.reshape(2,1)) * np.dot((1-A1), A0.reshape(1,2))
    db1 = (A2 - labels) * np.dot(W2.T, A1.reshape(2,1)) * (1-A1)

    # Update values
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    return W2, W1, b1, b2

X = np.array([3,5]).reshape(2,1)
labels = 0.75
A1, Z1, Z2, A2, W1, W2, b1, b2 = forward_Propagation(X,labels)

back_Propagation(X,A1, Z1=Z1, A2=A2, Z2=Z2,labels=labels, W1=W1, W2=W2, b1=b1, b2=b2,alpha=1)