import random
import math

class Neuron():
    def __init__(self, input_sz:int):
        self.W = [random.gauss()*(math.sqrt(2/input_sz)) for _ in range(input_sz)]
        self.b = 0
    
    def getZ(self, X):
        
        if len(X) != len(self.W): 
            print("invalid X size in Z calculation")
            return
        
        
        Z=self.b
        
        for x,w in zip(X, self.W):
            Z += (x*w)
        
        return Z


    def updateW(self, W):
        
        if len(W) != len(self.W): 
            print("invalid W size during weight update")
            return
        
        self.W = W 
        
    def predict(self, X):
        z = self.getZ(X)
        
        # return z

        if z>0: return 1
        return 0



X_train = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y_train = [0,1,1,1]  # LOGICAL OR 
# y_train = [0,0,0,1]  # WORKS FOR LOGICAL AND TOO
# DOESN'T WORK FOR XOR


N = Neuron(2)

LR = 0.1
epochs = 1


for epoch in range(epochs):
    for X,y in zip(X_train, y_train):
        
        y_hat = N.predict(X)
        
        # stochastic gradient descent
        N.W[0] -= ( LR*(y_hat-y)*X[0]  )
        N.W[1] -= ( LR*(y_hat-y)*X[1]  )
        N.b -= ( LR*(y_hat-y)  )
        
for X in X_train:
    print(X, N.predict(X))
    
    


        
         
        
        