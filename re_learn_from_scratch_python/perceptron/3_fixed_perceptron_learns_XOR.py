'''
X (4,2)  4 data x 2 feature

layer 1:
    2 neuron
    W1 (2,2)  each column is a neuron, 2 rows are its two incoming weights
    b1 (2,)  two bias for the 2 neurons
    
    Z1 = W@X + b1   , even though b1 isn't of correct shape, it broadcasts
    
    A1 = tanh(Z1)  , activation tanh instead of previously used ReLU
    

layer 2: output layer:
    1 neuron
    W2 (2,1)
    b2 (1,)
    Z2 = A1@W2 + b2
    A2 = sigmoid(Z2)


Loss = BCE = Binary Cross-Entropy
     = -[Y.ln(A) + (1-Y).ln(1-A)]
     
    
interesting thing:
    dL/dZ simplifies to = A-Y   (in this case dL/dZ2 = A2-Y )

also:
    A1 = tanh(Z1)
    d tanh(Z) / dZ =  1 - tanh^2(Z)
    so dA1/dZ1 = 1 - A1^2


'''


'''
IMPORTANT NOTE:

    this also sometimes give wrong output,
    actually the main limitation is not theory, theoritically its a correct implementation 
    but only 2 neuron in hidden layer sometimes struggles to find global minima if initialization is not in favor 

    so I turned those number of neurons into variable/parameters
    just setting the hidden layer with 3 neuron instead of 2 fixes the issue 

    I tried doing that in the previous version, it also worked great
'''



import numpy as np


# np.random.seed(42)  # fix a seed if you want same output in each run 


# each row is a datapoint, each column are features 
X_train = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)  

y_train = np.array([0,1,1,0]).reshape(-1, 1)    # XOR 



NUM_FEATURES =  X_train.shape[1]
NUM_HIDDEN_LAYER_1_NEURONS = 3  # use 3 neuron in hidden layer instead of 2 and get bette result
NUM_OUTPUT_LAYER_NEURONS = 1

# each column is for a neuron, where each row is the weights of that  neuron

W1 = np.random.randn(NUM_FEATURES, NUM_HIDDEN_LAYER_1_NEURONS)*(np.sqrt(2/NUM_FEATURES))

# each column is for a neuron, 
b1 = np.zeros(NUM_HIDDEN_LAYER_1_NEURONS)




W2 = np.random.randn(NUM_HIDDEN_LAYER_1_NEURONS, NUM_OUTPUT_LAYER_NEURONS )*(np.sqrt(2/NUM_HIDDEN_LAYER_1_NEURONS))
b2 = np.zeros(NUM_OUTPUT_LAYER_NEURONS)



def sigmoid(Z):

    return 1/(1+ np.exp(-Z))



epochs = 20000
LR = 0.1

for _ in range(epochs):

    Z1 = X_train@W1 + b1 
    A1 = np.tanh(Z1)

    Z2 = A1@W2 + b2
    A2 = sigmoid(Z2)


    # dL/dW2 = (dL/dA2)(dA2/dZ2)(dZ2/dW2)  chain rule
    #        = ( dL/dZ ) . A1
    #        = (A2-Y) . A1

    dL_dZ2= (A2-y_train) / NUM_FEATURES
    # dL_dZ2 = A2 - y_train

    dL_dW2 = A1.T @ dL_dZ2

    dL_db2 = np.sum(dL_dZ2, axis=0)  # Shape: (1,)  columnwise sum


    # dL/dW1 = dL/dA2 . dA2/dZ2 . dZ2/dA1 .dA1/dZ1 . dZ1/dW1
    #        = [dL_dZ2]         . W2      . (1-A1**2) . X


    dL_dZ1 = (dL_dZ2@W2.T)*(1-A1**2)

    dL_dW1 = X_train.T @ dL_dZ1 

    dL_db1 = np.sum(dL_dZ1, axis=0)  # Shape: (2,)
    
    
    W2 -=  LR*dL_dW2
    b2 -=  LR*dL_db2
    W1 -=  LR*dL_dW1
    b1 -=  LR*dL_db1
    
    


## TEST 

Z1 = X_train@W1 + b1 
A1 = np.tanh(Z1)

Z2 = A1@W2 + b2
A2 = sigmoid(Z2)



def predict(A):    
    return (A>=0.5).astype(int)

print(X_train)
print(predict(A2))

