# main.py

import numpy as np

# 1. Initialize Parameters
def initialize_parameters(layer_dims):
    """
    Initializes parameters for a deep neural network.

    Arguments:
    layer_dims -- list containing the dimensions of each layer in our network

    Returns:
    parameters -- dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

# 2. Forward Propagation
def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data)
    W -- weights matrix
    b -- bias vector

    Returns:
    Z -- pre-activation parameter
    cache -- tuple containing "A", "W", and "b" for backpropagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    """
    Implements the sigmoid activation function.

    Arguments:
    Z -- pre-activation parameter

    Returns:
    A -- output of sigmoid(z)
    cache -- returns Z, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Implements the ReLU activation function.

    Arguments:
    Z -- pre-activation parameter

    Returns:
    A -- output of relu(z)
    cache -- returns Z, useful during backpropagation
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements forward propagation for a single layer with an activation function.

    Arguments:
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector
    activation -- "sigmoid" or "relu"

    Returns:
    A -- post-activation value
    cache -- tuple of caches for backpropagation
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    """
    Implements forward propagation for L-layer neural network.

    Arguments:
    X -- input data
    parameters -- initialized weights and biases

    Returns:
    AL -- output of the last layer
    caches -- list of caches for backpropagation
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers

    # Iterate over layers (LINEAR -> RELU) * (L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    # Last layer uses sigmoid activation
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y):
    """
    Implements the cross-entropy cost function.

    Arguments:
    AL -- predictions
    Y -- true labels

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost

# 4. Backward Propagation
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(dZ, cache):
    """
    Implements the linear part of backpropagation for a single layer.

    Arguments:
    dZ -- gradient of the cost with respect to the linear output
    cache -- tuple of values (A_prev, W, b)

    Returns:
    dA_prev -- gradient of cost with respect to activation of the previous layer
    dW -- gradient of cost with respect to W
    db -- gradient of cost with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for a single layer with activation function.

    Arguments:
    dA -- gradient of cost with respect to activation
    cache -- tuple containing linear and activation caches
    activation -- "relu" or "sigmoid"

    Returns:
    dA_prev -- gradient of cost with respect to previous layer activation
    dW -- gradient of cost with respect to W
    db -- gradient of cost with respect to b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implements backpropagation for an L-layer neural network.

    Arguments:
    AL -- output from forward propagation
    Y -- true labels
    caches -- list of caches from forward propagation

    Returns:
    grads -- dictionary with gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initialize backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Last layer (sigmoid)
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    # Loop through hidden layers (ReLU)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

# 5. Update Parameters
def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.

    Arguments:
    parameters -- dictionary containing parameters
    grads -- dictionary containing gradients
    learning_rate -- learning rate for update

    Returns:
    parameters -- dictionary containing updated parameters
    """
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    
    return parameters

# 6. Model Training Function
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements an L-layer neural network.

    Arguments:
    X -- input data
    Y -- true labels
    layers_dims -- list of layer sizes
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations
    print_cost -- if True, prints cost every 100 iterations

    Returns:
    parameters -- trained parameters
    costs -- list of costs
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Log cost
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
        if i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs
