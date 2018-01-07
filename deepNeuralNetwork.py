import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
#from tempfile import TemporaryFile

def generateStartingHands(startingHands, test = True):    
    #startingHands = np.load(outfile)  #No longer using tempfile

    pokerHands, results, testPokerHands, testResults = [],[],[],[]

    np.random.seed(1)

    for y in range(13):
        for x in range(13):
            hand = [0, 0, 0, 0]
            ev = startingHands[y][x]

            if x > y: 
                hand[2] = 1
                lowest = 12 - x
                highest = 12 - y
            else:
                lowest = 12 - y
                highest = 12 - x

            if highest == lowest: 
                hand[3] = 1

            hand[0] = highest/ 12.
            hand[1] = lowest/ 12.

            if np.random.randint(10) == 0 and test:   
                testPokerHands.append(hand)
                testResults.append([ev])
            else:
                pokerHands.append(hand)
                results.append([ev])
                
    pokerHands = np.array(pokerHands).T
    results = np.array(results).T
    testPokerHands = np.array(testPokerHands).T
    testResults = np.array(testResults).T

    return pokerHands, results, testPokerHands, testResults

def testPokerStartingHands():

    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'neare|st'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)
    train_x, train_y, test_x, test_y = generateStartingHands(np.loadtxt("data", unpack=True))

    m_train = train_x.shape[1]

    print ("Number of training examples: " + str(m_train))
    print ("train_x_orig shape: " + str(train_x.shape))
    print ("train_y shape: " + str(train_y.shape))

    n_x = train_x.shape[0]
    n_y = train_y.shape[0]
    layers_dims = [n_x, 50, 12, 5, n_y] #  This determines how many hidden layers the network will use
    
    runTests()
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    # This shows the final error rate for the training set
    AL, caches = L_model_forward(train_x, parameters)
    cost = compute_cost(AL, train_y)
    print("Training set error: " + '{:06.3f}'.format(cost*100) + "%")
    
    # This shows the error rate for your testing set
    AL, caches = L_model_forward(test_x, parameters)
    cost = compute_cost(AL, test_y)
    print("Testing set error:  " + '{:06.3f}'.format(cost*100) + "%")

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, plotGraph = False, params = None):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    if params == None:
        parameters = initialize_parameters_deep(layers_dims)
    else:
        parameters = params
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.        
        AL, caches = L_model_forward(X, parameters)
        
        
        # Compute cost.
        cost = compute_cost(AL, Y)
        
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % (num_iterations//25) == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
        
    if plotGraph:
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters