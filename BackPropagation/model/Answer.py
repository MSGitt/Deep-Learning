from collections import OrderedDict
from tkinter import E
import numpy as np

def softmax(z):
    # We provide numerically stable softmax.
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)

    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum

    return sm


class ReLU:
    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.

    ReLU(x) = x if x > 0.
              0 otherwise.
    Be careful. It's '>', not '>='.
    """

    def __init__(self):
        # 1 (True) if ReLU input <= 0
        self.zero_mask = None

    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            self.out : Values applied elementwise ReLU function on input 'z'.
        """
        self.out = None
        
        self.zero_mask = (z <= 0)
        self.out = z.copy()

        self.out[self.zero_mask] = 0

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
                - d_prev = dL/dk, where k = ReLU(z).
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        
        d_prev[self.zero_mask] = 0
        dz = d_prev


        return dz

    def update(self, learning_rate):
        # NOT USED IN ReLU
        pass

    def summary(self):
        return 'ReLU Activation'


class LeakyReLU:
    """
    Leaky ReLU function.
    Implement forward & backward path of Leaky ReLU
    """

    def __init__(self, alpha=0.1):
        
        self.mask = None
        self.alpha = alpha

    def forward(self, z):
        """
        Leaky ReLU Forward.
        LeakyReLU(x) = max(alpha*x, x)

        z --> (Leaky ReLU) --> out

        [Inputs]
            z : Leaky ReLU input in any shape.

        [Outputs]
            self.out : Values applied elementwise Leaky ReLU function on input 'z'.
        """
        
        self.out = z
        self.out[self.out <= 0] *= self.alpha 

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Leaky ReLU Backward.

        z --> (Leaky ReLU) --> out
        dz <-- (dLeakyReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. Leaky ReLU input z.
        """
        
        self.out[self.out > 0] = 1 
        self.out[self.out <= 0] = self.alpha

        dz = self.out * d_prev
        
        return dz

    def update(self, learning_rate):

        pass

    def summary(self):
        return 'Leaky ReLU Activation'


class ELU:
    """
    ELU Function.
    Implement forward & backward path of ELU.
    """
    def __init__(self, alpha=1.0):
        # 1 (True) if ELU input <= 0
        self.mask = None
        self.alpha = alpha

    def forward(self, z):
        """
        ELU Forward.
        ELU(x) = x if x >= 0 else alpha * (exp(x) - 1)

        z --> (ELU) --> out

        [Inputs]
            z : ELU input in any shape.

        [Outputs]
            self.out : Values applied elementwise ELU function on input 'z'.
        """
        self.out = None
        
        self.out = z 
        self.out[self.out <= 0] = np.exp(self.out[self.out <=0]) - 1 

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        ELU Backward.

        z --> (ELU) --> out
        dz <-- (dELU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. Leaky ReLU input z.
        """
        
        self.out[self.out > 0] = 1
        self.out[self.out <= 0] += 1

        dz = self.out * d_prev

        return dz

    def update(self, learning_rate):
        # NOT USED IN Leaky ReLU
        pass

    def summary(self):
        return 'ELU Activation'

"""
    ** Fully-Connected Layer **
    Single Fully-Connected Layer.

    Given input features,
    FC layer linearly transforms the input with weights (self.W) & bias (self.b).

    You need to implement forward and backward pass.
"""

class FCLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim / 2)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        """
        FC Layer Forward.
        Use variables : self.x, self.W, self.b

        [Input]
        x: Input features.
        - Shape : (batch size, In Channel, Height, Width)
        or
        - Shape : (batch size, input_dim)

        [Output]
        self.out : fc result
        - Shape : (batch size, output_dim)

        Tip : you do not need to implement L2 regularization here. already implemented in ClassifierModel.forward()
        """
        # Flatten input if needed.
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        self.x = x
        
        self.out = np.dot(self.x, self.W) + self.b

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        FC Layer Backward.
        Use variables : self.x, self.W

        [Input]
        d_prev: Gradients value so far in back-propagation process.
        reg_lambda: L2 regularization weight. (Not used in activation function)

        [Output]
        dx : Gradients w.r.t input x
        - Shape : (batch_size, input_dim) - same shape as input x
        """

        dx = np.dot(d_prev, self.W.T) 
        
        self.dW = np.dot(self.x.T , d_prev)
        self.db = np.sum(d_prev, axis = 0)

        # L2 regularization
        self.dW = self.dW + reg_lambda * self.W
        return dx

    def update(self, learning_rate):
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Input -> Hidden : %d -> %d ' % (self.W.shape[0], self.W.shape[1])

"""
    ** Softmax Layer **
    Softmax Layer applies softmax (WITHOUT any weights or bias)

    Given an score,
    'SoftmaxLayer' applies softmax to make probability distribution. (Not log softmax or else...)

    You need to implement forward and backward pass.
    (This is NOT an entire model.)
"""

class SoftmaxLayer:
    def __init__(self):
        # No parameters
        pass

    def forward(self, x):
        """
        Softmax Layer Forward.
        Apply softmax (not log softmax or others...) on axis-1

        Use 'softmax' function above in this file.
        We recommend you see the function.

        [Input]
        x: Score to apply softmax
        - Shape: (batch_size, # of class)

        [Output]
        y_hat: Softmax probability distribution.
        - Shape: (batch_size, # of class)
        """
        self.y_hat = softmax(x)

        return self.y_hat

    def backward(self, d_prev=1, reg_lambda=0):
        """
        Softmax Layer Backward.
        Gradients w.r.t input score.

        That is,
        Forward  : softmax prob = softmax(score)
        Backward : dL / dscore => 'dx'

        Compute dx (dL / dscore).

        [Input]
        d_prev : Gradients flow from upper layer.

        [Output]
        dx: Gradients of softmax layer input 'x'
        """
        batch_size = self.y.shape[0]

        dx = (self.y_hat - self.y) / batch_size

        return dx


    def ce_loss(self, y_hat, y):
        """
        Compute Cross-entropy Loss.
        Use epsilon (eps) for numerical stability in log.

        [Input]
        y_hat: Probability after softmax.
        - Shape : (batch_size, # of class)

        y: One-hot true label
        - Shape : (batch_size, # of class)

        [Output]
        self.loss : cross-entropy loss
        - float
        """
        eps = 1e-10
        self.y_hat = y_hat
        self.y = y
        
        self.loss = -np.sum(np.multiply(self.y , np.log(self.y_hat + eps))) / self.y.shape[0]

        return self.loss

    def update(self, learning_rate):

        pass

    def summary(self):
        return 'Softmax layer'


"""
    ** Classifier Model **
    This is an class for entire Classifier Model.
    All the functions and variables are already implemented.
    Look at the codes below and see how the codes work.

    <<< DO NOT CHANGE ANYTHING HERE >>>
"""

class ClassifierModel:
    def __init__(self,):
        self.layers = OrderedDict()
        self.softmax_layer = None
        self.loss = None
        self.pred = None

    def predict(self, x):
        # Outputs model softmax score
        for name, layer in self.layers.items():
            x = layer.forward(x)
        x = self.softmax_layer.forward(x)
        return x

    def forward(self, x, y, reg_lambda):
        # Predicts and Compute CE Loss
        reg_loss = 0
        self.pred = self.predict(x)
        ce_loss = self.softmax_layer.ce_loss(self.pred, y)

        for name, layer in self.layers.items():
            if isinstance(layer, FCLayer):
                norm = np.linalg.norm(layer.W, 2)
                reg_loss += 0.5 * reg_lambda * norm * norm

        self.loss = ce_loss + reg_loss

        return self.loss

    def backward(self, reg_lambda):
        # Back-propagation
        d_prev = 1
        d_prev = self.softmax_layer.backward(d_prev, reg_lambda)
        for name, layer in list(self.layers.items())[::-1]:
            d_prev = layer.backward(d_prev, reg_lambda)

    def update(self, learning_rate):
        # Update weights in every layer with dW, db
        for name, layer in self.layers.items():
            layer.update(learning_rate)

    def add_layer(self, name, layer):
        # Add Neural Net layer with name.
        if isinstance(layer, SoftmaxLayer):
            if self.softmax_layer is None:
                self.softmax_layer = layer
            else:
                raise ValueError('Softmax Layer already exists!')
        else:
            self.layers[name] = layer

    def summary(self):
        # Print model architecture.
        print('======= Model Summary =======')
        for name, layer in self.layers.items():
            print('[%s] ' % name + layer.summary())
        print('[Softmax Layer] ' + self.softmax_layer.summary())
        print()


class EvaluationMetric:
    def __init__(self, num_class, method='macro'):
        self.num_class = num_class
        self.method = method

    def confusion_matrix(self, num_of_class, pred, true):
        c_m_c = [{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for _ in range(num_of_class)]
        for class_idx in range(num_of_class):
            for idx in range(true.shape[0]):

                if true[idx] == class_idx  :

                    if true[idx] == pred[idx] :
                        
                        c_m_c[class_idx]["TP"] += 1 

                    else : 

                        c_m_c[class_idx]["FN"] += 1 

                else :   
                    if true[idx] == pred[idx] : 

                        c_m_c[class_idx]["TN"] += 1 
                    
                    if pred[idx] == class_idx :

                        c_m_c[class_idx]["FP"] += 1 

        return c_m_c

    def precision(self, TP, FP, FN, TN):
        
        out = TP / (TP + FP)

        return out

    def recall(self, TP, FP, FN, TN):
        
        out = TP / (TP + FN)

        return out

    def f_measure(self, precision, recall, beta=1.0):
        
        out = ((beta**2 + 1) * precision * recall) / ((beta**2) * precision + recall)

        return out

    def multiclass_f_measure(self, pred, true, num_of_class):
        f_measure_, precision_, recall_ = [], [], []
        out = {'f_measure': 0.0, 'precision': 0.0, 'recall': 0.0}

        confusion_matrix = self.confusion_matrix(num_of_class, pred, true)

        for class_idx in range(num_of_class):
            confusion_matrix_ = confusion_matrix[class_idx]
            
            precision = self.precision(confusion_matrix_['TP'], confusion_matrix_['FP'], confusion_matrix_['FN'], confusion_matrix_['TN'])
            recall = self.recall(confusion_matrix_['TP'], confusion_matrix_['FP'], confusion_matrix_['FN'], confusion_matrix_['TN'])

            f_measure_.append(self.f_measure(precision, recall))
            precision_.append(precision)
            recall_.append(recall)

        for class_idx in range(num_of_class):
            print(f'For class {class_idx}, precision: {precision_[class_idx]:.4f}\trecall: {recall_[class_idx]:.4f}\tf-measure: {f_measure_[class_idx]:.4f}')
        print()
        
        if self.method == 'macro':
            for key, val_per_class in zip(out.keys(), [f_measure_, precision_, recall_]):
                out[key] = sum(val_per_class) / len(val_per_class)
        else:
            raise ValueError('Method should be macro.')

        return out
            
