import numpy as np
from utils import _initialize, optimizer

np.random.seed(85453)

# DATA
DATA_NAME = 'AND' #'XOR_perceptron'

# HYPERPARAMETERS
batch_size = 32
num_epochs = 5000
learning_rate = 0.5

epsilon = 0.01 # not for SGD
gamma = 0.9 # not for SGD

# OPTIMIZER
OPTIMIZER = 'SGD'

assert DATA_NAME in ['AND', 'XOR_perceptron']
assert OPTIMIZER in ['SGD']

# Load dataset, model and evaluation metric
train_data, test_data, Perceptron, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)

# Make model & optimizer
model = Perceptron(num_features)
optim = optimizer(OPTIMIZER, gamma=gamma, epsilon=epsilon)

# TRAIN
loss = model.train(train_x, train_y, num_epochs, batch_size, learning_rate, optim)
print('Training Loss at the last epoch: %.2f' % loss)

# EVALUATION
test_x, test_y = test_data
pred = model.forward(test_x)
ACC = metric(pred, test_y)

print(OPTIMIZER, ' ACC on Test Data : %.2f ' % ACC)
