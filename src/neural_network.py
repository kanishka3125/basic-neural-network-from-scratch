# Neural Network from Scratch using NumPy
# Implemented during my AI internship to gain hands-on understanding of forward propagation, loss computation, and backpropagation without relying on high-level machine learning libraries.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# -----------------------------
# STEP 1- Data Loading and Preprocessing
# -----------------------------

# Each digit image is represented as a flattened vector of pixel intensities
digits = load_digits()
X = digits.data
y = digits.target

# Normalize input features to improve numerical stability during training
X = X / 16.0


# -----------------------------
# STEP 2 - Utility Functions
# -----------------------------

# Converts class labels (0–9) into one-hot encoded vectors
# suitable for softmax classification
def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

y_one_hot = one_hot_encode(y)


# -----------------------------
# STEP 3 - Model Parameter Initialization
# -----------------------------

# Small random initialization helps break symmetry between neurons
np.random.seed(42)

W1 = np.random.randn(64, 32) * 0.01
b1 = np.zeros((1, 32))

W2 = np.random.randn(32, 10) * 0.01
b2 = np.zeros((1, 10))


# -----------------------------
# STEP 4 - Activation Functions
# -----------------------------

# ReLU introduces non-linearity, enabling the network to learn complex patterns
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0


# Softmax transforms raw scores into a probability distribution
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


# -----------------------------
# STEP 5 - Forward Propagation
# -----------------------------

# Computes network output given the current parameters
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


# -----------------------------
# STEP 6 - Loss Function
# -----------------------------

# Categorical cross-entropy quantifies prediction error
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m


# -----------------------------
# STEP 7 - Backpropagation
# -----------------------------

# Updates weights and biases using gradient descent
def backpropagation(X, y, Z1, A1, A2, learning_rate=0.01):
    global W1, b1, W2, b2
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# -----------------------------
# STEP 8 - Evaluation Metric
# -----------------------------

# Measures classification accuracy
def accuracy(y_true, y_pred):
    return np.mean(
        np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    )


# -----------------------------
# STEP 9 - Training Loop
# -----------------------------

epochs = 200
losses = []
accuracies = []

# The model iteratively improves by minimizing loss over multiple epochs
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_propagation(X)

    loss = compute_loss(y_one_hot, A2)
    acc = accuracy(y_one_hot, A2)

    backpropagation(X, y_one_hot, Z1, A1, A2)

    losses.append(loss)
    accuracies.append(acc)

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")


# -----------------------------
# STEP 10 - Visualization
# -----------------------------

# Plotting training trends to verify learning behavior
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("results/loss.png")
plt.show()

plt.plot(accuracies)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("results/accuracy.png")
plt.show()

print("Training complete.")
