import numpy as np
from utils import _initialize, optimizer
import torch.optim as optim


np.random.seed(85453)
OPTIMIZER = 'SGD'

# DATA
DATA_NAME = 'XOR_mlp' # 'Moon'

# HYPERPARAMETERS
batch_size = 32
num_epochs = 5000
learning_rate = 0.5
num_hidden = 4

epsilon = 0.01 # not for SGD
gamma = 0.9 # not for SGD



assert DATA_NAME in ['Moon', 'XOR_mlp']


# Load dataset, model and evaluation    metric
train_data, test_data, MLP, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)


# Make model & optimizer
model = MLP(num_features, num_hidden)


# OPTIMIZER
optimizer = optim.SGD(model.parameters(), learning_rate)

# TRAIN
loss = model.train(train_x, train_y, num_epochs, batch_size, learning_rate, optimizer)
print('Training Loss at the last epoch: %.2f' % loss)

# EVALUATION
test_x, test_y = test_data
pred = model.forward(test_x)
ACC = metric(pred, test_y)

print(OPTIMIZER, ' ACC on Test Data : %.3f' % ACC)

