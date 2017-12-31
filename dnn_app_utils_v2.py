import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):

    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def initialize_parameters(n_x, n_h, n_y):
   
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


def initialize_parameters_deep(layer_dims):
  
    np.random.seed(2)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
   
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
   
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):

    ''' The original cost function was meant for binary classification.
        I changed it to one that works better for reinforcement learning'''

    m = Y.shape[1]
    # Compute loss from aL and y.
    #cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))  <- Binary classification
    
    cost =  np.sqrt((1./m) * np.sum((AL - Y)**2))  # Means squared error (variance) for reinforcement learning
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
   
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    '''
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    '''
        
    return probas

def print_mislabeled_images(classes, X, y, p):

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        
def linear_forward_test_case():
    np.random.seed(1)

    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    
    return A, W, b

def linear_activation_forward_test_case():

    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():

    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])
    
    return Y, aL

def linear_backward_test_case():

    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_activation_backward_test_case():

    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    
    return dA, linear_activation_cache

def L_model_backward_test_case():

    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches

def update_parameters_test_case():

    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return parameters, grads


def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

def print_grads(grads):
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA2"])) # this is done on purpose to be consistent with lecture where we normally start with A0
                                        # in this implementation we started with A1, hence we bump it up by 1. 


def testCase1():
    parameters = initialize_parameters(3,2,1)
    W1, b1, W2, b2 = [parameters[x] for x in ["W1", "b1", "W2", "b2"]]
    
    
    assert str(W1) == str(np.array([[ 0.01624345, -0.00611756, -0.00528172], [-0.01072969,  0.00865408, -0.02301539]]))
    assert str(b1) == str(np.array([[0.],[0.]]))
    assert str(W2) == str(np.array([[ 0.01744812, -0.00761207]]))
    assert str(b2) == str(np.array([[ 0.]]))
    
def testCase2():
    parameters = initialize_parameters_deep([5,4,3])
    W1, b1, W2, b2 = [parameters[x] for x in ["W1", "b1", "W2", "b2"]]
    
    assert np.sum(np.abs(W1 - [[-0.18637978, -0.02516329, -0.95533594,  0.73355141, -0.80204878],
                 [-0.37644087,  0.22489541, -0.55690976, -0.47313062, -0.40652056],
                 [ 0.24661775,  1.02510659,  0.01857698, -0.49995146,  0.24107421],
                 [-0.26661072, -0.00855542,  0.52547652, -0.33445806,  0.00403621]])) < .1
    assert str(b1) == str(np.array([[0.],[0.],[0.],[0.]]))
    
    
    assert np.sum(np.abs(W2 - [[-0.43905395, -0.07821709,  0.12828523, -0.49438952],
                                 [-0.16941098, -0.11809202, -0.31882751, -0.59380614],
                                 [-0.71060861, -0.0767476,  -0.13452848,  1.11568339]])) < .1
    assert str(b2) == str(np.array([[0.],[0.],[0.]]))
    
def testCase3():
    A, W, b = linear_forward_test_case()
    Z, linear_cache = linear_forward(A, W, b)

    assert str(Z) == str(np.array([[ 3.26295337, -1.23429987]]))
    
def testCase4():
    A_prev, W, b = linear_activation_forward_test_case()

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    assert str(A) == str(np.array([[ 0.96890023, 0.11013289]]))
    
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    assert str(A) == str(np.array([[ 3.43896131, 0.]]))
    
def testCase5():
    X, parameters = L_model_forward_test_case_2hidden()
    AL, caches = L_model_forward(X, parameters)
    
    assert str(AL) == str(np.array([[ 0.03921668, 0.70498921, 0.19734387, 0.04728177]]))
    assert str(len(caches)) == str(3)
def testCase6():
    Y, AL = compute_cost_test_case()
    cost = compute_cost(AL, Y)
    
    
    assert np.abs(cost - 0.369684550214) <= .01
    
def testCase7():
    # Set up some test inputs
    dZ, linear_cache = linear_backward_test_case()
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    assert str(dA_prev) == str(np.array([[ 0.51822968, -0.19517421],[-0.40506361, 0.15255393],[ 2.37496825, -0.89445391]]))
    assert str(dW) == str(np.array([[-0.10076895, 1.40685096, 1.64992505]]))
    assert str(db) == str(np.array([[ 0.50629448]]))
    
def testCase8():
    AL, linear_activation_cache = linear_activation_backward_test_case()

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")

    assert str(dA_prev) == str(np.array([[ 0.11017994, 0.01105339],[ 0.09466817, 0.00949723],[-0.05743092, -0.00576154]]))
    assert str(dW) == str(np.array([[ 0.10266786, 0.09778551, -0.01968084]]))
    assert str(db) == str(np.array([[-0.05729622]]))
    
    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
    
    assert str(dA_prev) == str(np.array([[ 0.44090989, -0.],[ 0.37883606, -0.],[-0.2298228, 0.]]))
    assert str(dW) == str(np.array([[ 0.44513824, 0.37371418, -0.10478989]]))
    assert str(db) == str(np.array([[-0.20837892]]))
    
def testCase9():
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    dW1, db1, dA2 = [grads[x] for x in ["dW1", "db1", "dA2"]]
    #print_grads(grads)
    
    assert str(dW1) == str(np.array([[ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                     [0., 0., 0., 0.],[ 0.05283652, 0.01005865, 0.01777766, 0.0135308]]))
    assert str(db1) == str(np.array([[-0.22007063], [ 0.], [-0.02835349]]))
    assert str(dA2) == str(np.array([[ 0.12913162, -0.44014127],[-0.14175655, 0.48317296], [ 0.01663708, -0.05670698]]))
    
def testCase10():
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)
    W1, b1, W2, b2 = [str(parameters[x]) for x in ["W1", "b1", "W2", "b2"]]
    
    assert str(W1) == str(np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
                                    [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
                                    [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]]))
    assert str(b1) == str(np.array([[-0.04659241], [-1.28888275], [ 0.53405496]]))
    assert str(W2) == str(np.array([[-0.55569196, 0.0354055, 1.32964895]]))
    assert str(b2) == str(np.array([[-0.84610769]]))

def runTests():
    testCase1()
    testCase2()
    testCase3()
    testCase4()
    testCase5()
    testCase6()
    testCase7()
    testCase8()
    testCase9()
    testCase10()
    print("Tests Comleted")

