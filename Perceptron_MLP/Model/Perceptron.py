import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):

        for k in range(epochs) :

            loss = 0

            for i in range(0, 800, batch_size) :

                grad = np.zeros((1, 3))
                
                X = x[i : i+batch_size] 
                Y = y[i : i+batch_size] 

                y_predict = np.dot(X , self.W) 

                y_ = np.array([1 if i >= 0 else -1 for i in y_predict]) 

                for j in range(len(Y)) :
                    if y_[j] != Y[j] :
                        loss += - np.multiply(Y[j] , np.dot(X[j], self.W) )
                        grad += - np.multiply(Y[j], X[j])

                self.W = optim.update(self.W, grad.T, lr) 
                
        return loss

    def forward(self, x):
      
        y_pred = np.dot(x, self.W) 
        y_predicted = np.array([1 if i>=0 else -1 for i in y_pred])
        y_predicted = np.expand_dims(y_predicted, 1)

        return y_predicted

