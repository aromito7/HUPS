import numpy as np

def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    
    return A, W, b

def linear_activation_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
        [-1.07296862,  0.86540763, -2.3015387 ]]),
 'W2': np.array([[ 1.74481176, -0.7612069 ]]),
 'b1': np.array([[ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.]])}
    """
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
    """
    z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
       [-1.62328545,  0.64667545],
       [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
    """
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_activation_backward_test_case():
    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
    """
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
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
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
    """
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
        [-1.8634927 , -0.2773882 , -0.35475898],
        [-0.08274148, -0.62700068, -0.04381817],
        [-0.47721803, -1.31386475,  0.88462238]]),
 'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
        [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
        [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
 'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
        [-0.16051336, -0.76883635, -0.23003072]]),
 'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.],
        [ 0.],
        [ 0.]]),
 'b3': np.array([[ 0.],
        [ 0.]])}
    grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]),
 'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]]),
 'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
 'da1': np.array([[ 0.70760786,  0.65063504],
        [ 0.17268975,  0.15878569],
        [ 0.03817582,  0.03510211]]),
 'da2': np.array([[ 0.39561478,  0.36376198],
        [ 0.7674101 ,  0.70562233],
        [ 0.0224596 ,  0.02065127],
        [-0.18165561, -0.16702967]]),
 'da3': np.array([[ 0.44888991,  0.41274769],
        [ 0.31261975,  0.28744927],
        [-0.27414557, -0.25207283]]),
 'db1': 0.75937676204411464,
 'db2': 0.86163759922811056,
 'db3': -0.84161956022334572}
    """
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
    
    assert str(W1) == str(np.array([[ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                                    [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                                    [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                                    [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]]))
    assert str(b1) == str(np.array([[0.],[0.],[0.],[0.]]))
    assert str(W2) == str(np.array([[-0.01185047, -0.0020565 , 0.01486148,  0.00236716],
                                    [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                                    [-0.00768836, -0.00230031, 0.00745056,  0.01976111]]))
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
    
    assert str(cost) == str(0.414931599615)
    
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
    
runTests()

  
