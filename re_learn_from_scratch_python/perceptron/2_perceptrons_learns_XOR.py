'''
the given code is theoritically correct but doesn't produce correct output always,
majority of the time it produce correct output, and sometimes it doesn't

---------------------------------------------------

the network used the input layer (4 data, two feature)  (4,2) 

a hidden layer of 2 neuron
    its weights W1 (2,2), bias b1 (2,)
    activation function ReLU
    
output layer of 1 neuron
    weights W2(2,1), bias b2(1,)
    
MSE loss function

--------------------------------------------

* the issue is that loss is getting stuck at some local minima sometimes
* also, as ReLU is used, if a neuron's Z value is <0 , activation =0,  its dead neuron, 
* a network with only 3 neuron is sensitive if any neuron is randomly dead
* so correct initialization matters, but as initialization is random, it doesn't always work
* a seed = 42 gives correct output btw


------------------------------------------------

* use tanh instead of ReLU, also try Binary Cross-Entropy derivative as loss function instead of MSE 

'''

'''
IMPORTANT NOTE:
    actually you can make it do correct otuput just by adding more neuron in the hidden layer,
    even setting 3 neuron in W1 insted of 2 gives correct output.
    and that even works if bias is initialized at zero
    
    check # ALTERNATIVE_FIX 
'''




import numpy as np


np.random.seed(42)  # fix a seed if you want same output in each run 


# each row is a datapoint, each column are features 
X_train = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y_train = np.array([0,1,1,0]).reshape(-1, 1)    # XOR 





# each column is for a neuron, where each row is the weights of that  neuron

W1 = np.random.randn(2, 2)*(np.sqrt(2/2))
# W1 = np.random.randn(2, 3)*(np.sqrt(2/2)) # ALTERNATIVE_FIX 

# each column is for a neuron, 
b1 = np.random.rand(2)
# b1 = np.zeros(3)  # ALTERNATIVE_FIX 




W2 = np.random.randn(2,1)*(np.sqrt(2/2))
# W2 = np.random.randn(3,1)*(np.sqrt(2/3)) # ALTERNATIVE_FIX 
b2 = np.random.rand(1)
# b2 = np.zeros(1) # ALTERNATIVE_FIX 


def ReLU(Z):
    return np.maximum(0,Z)

def deriv_ReLU(Z):
    return (Z>0).astype(int)

def sigmoid(Z):

    return 1/(1+ np.exp(-Z))



epoch = 100000
LR = 0.1

for _ in range(epoch):

    Z1 = X_train@W1 + b1 
    A1 = ReLU(Z1)

    Z2 = A1@W2 + b2
    A2 = sigmoid(Z2)


    # dL/dW2 = (dL/dA2)(dA2/dZ2)(dZ2/dW2)  chain rule
    #        = (A2-Y) . A2(1-A2) . A1
    #        = ( dL/dZ ) . A1

    dL_dZ2= (A2-y_train)*A2*(1-A2)  # element wize multiplication 
    # dL_dZ2 = A2 - y_train

    dL_dW2 = A1.T @ dL_dZ2

    dL_db2 = np.sum(dL_dZ2, axis=0)  # Shape: (1,)  columnwise sum


    # dL/dW1 = dL/dA2 . dA2/dZ2 . dZ2/dA1 .dA1/dZ1 . dZ1/dW1
    #        = [dL_dZ2]         . W2      . d_relu(z1) . X


    dL_dZ1 = (dL_dZ2@W2.T)*deriv_ReLU(Z1)

    dL_dW1 = X_train.T @ dL_dZ1 

    dL_db1 = np.sum(dL_dZ1, axis=0)  # Shape: (2,)
    
    
    W2 -=  LR*dL_dW2
    b2 -=  LR*dL_db2
    W1 -=  LR*dL_dW1
    b1 -=  LR*dL_db1
    
    


## TEST 

Z1 = X_train@W1 + b1 
A1 = ReLU(Z1)

Z2 = A1@W2 + b2
A2 = sigmoid(Z2)



def predict(A):    
    return (A>=0.5).astype(int)

print(X_train)
print(predict(A2))

