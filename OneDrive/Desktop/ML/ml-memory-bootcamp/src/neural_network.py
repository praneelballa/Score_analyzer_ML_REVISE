import numpy as np
import pandas as pd

def sigmoid(z):
    """
    Computes the sigmoid function: squeezes any number between 0 and 1.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def dense(a_in, W, b, activation="sigmoid"):
    """
    Computes one layer of the neural network.
    Formula: Z = (input * weights) + bias
    Then passes Z through an activation function.
    """
    z = np.dot(a_in, W) + b
    if activation == "sigmoid":
        a_out = sigmoid(z)
    else:
        a_out = z  # No activation (linear)
    return a_out, z

def forward_propagation(X, parameters):
    """
    Passes data forward through all layers of our network.
    """
    # Unpack weights and biases from our dictionary
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    
    # Layer 1: Input to Hidden Layer 1 (25 neurons)
    A1, Z1 = dense(X, W1, b1, activation="sigmoid")
    
    # Layer 2: Hidden Layer 1 to Hidden Layer 2 (15 neurons)
    A2, Z2 = dense(A1, W2, b2, activation="sigmoid")
    
    # Output Layer: Hidden Layer 2 to Output (1 neuron)
    A3, Z3 = dense(A2, W3, b3, activation="sigmoid")
    
    # Store intermediate values in a cache (useful later when you learn backpropagation!)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    
    return A3, cache

# --- Execution Example ---
if __name__ == "__main__":
    print("Testing Neural Network Forward Propagation from Scratch...\n")
    
    # Create mock student data with 6 features for 4 students
    np.random.seed(42)
    m = 4  # 4 students
    n_features = 6
    X_mock = np.random.randn(m, n_features)
    
    # Initialize random weights and biases for our network architecture:
    # W1: 6 inputs into 25 neurons -> shape (6, 25)
    # W2: 25 inputs into 15 neurons -> shape (25, 15)
    # W3: 15 inputs into 1 neuron -> shape (15, 1)
    parameters = {
        'W1': np.random.randn(n_features, 25) * 0.01,
        'b1': np.zeros((1, 25)),
        'W2': np.random.randn(25, 15) * 0.01,
        'b2': np.zeros((1, 15)),
        'W3': np.random.randn(15, 1) * 0.01,
        'b3': np.zeros((1, 1))
    }
    
    # Run forward propagation
    pass_probabilities, cache = forward_propagation(X_mock, parameters)
    
    print("Pass Probabilities calculated by our NumPy neural network:")
    for i, prob in enumerate(pass_probabilities):
        status = "PASS" if prob[0] >= 0.5 else "FAIL"
        print(f"Student {i+1}: {prob[0]*100:.2f}% probability ({status})")