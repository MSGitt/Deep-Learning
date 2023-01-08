import os
import numpy as np
from models.Perceptron import Perceptron
from models.MLP import MLP
from optim.Optimizer import SGD


def load_reg_data(path, filename, target_at_front, normalize=False, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    # bias = np.ones((x.shape[0], 1), dtype=np.float32)
    # x = np.concatenate((bias, x), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y

def load_class_data(path, filename, target_at_front, to_perceptron_binary=False, to_binary=False, normalize=False, excludes=None):
    if excludes is None:
        excludes = []

    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        line = f.readlines()

    lines = []
    for s in line:

        if s == '\n':
            pass
        else:
            lines.append(s.strip().split(','))
    # lines = [s.strip().split(',') for s in lines ]

    header = lines[0]
    raw_data = lines[1:]
    num_feat = len(raw_data[0])
    feat_to_idx = [{} for _ in range(num_feat)]

    data = []
    for d in raw_data:
        line = []
        
        for i, f in enumerate(d):
            if i in excludes:
                continue

            try:
                line.append(float(f))
            except:
                if f in feat_to_idx[i]:
                    f_idx = feat_to_idx[i][f]
                else:
                    f_idx = len(feat_to_idx[i])
                    feat_to_idx[i][f] = f_idx
                line.append(f_idx)
        data.append(line)

    data = np.array(data, dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0].astype(np.int32)
    else:
        x, y = data[:, :-1], data[:, -1].astype(np.int32)

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)

    if to_perceptron_binary:
        y[y == 1] = 1
        y[y < -1] = 1
        y[y == 0] = -1
    elif to_binary:
        y[y > 1] = 1
        y[y < -1] = 1


    return x, y


def XOR_MLP_Data(path, filename):
    return load_class_data(path, filename, target_at_front=True, to_binary=True)
    
def XOR_perceptron_Data(path, filename):
    return load_class_data(path, filename, target_at_front=True, to_perceptron_binary=True)

def MoonData(path, filename):
    return load_class_data(path, filename, target_at_front=True, to_binary=True)

def AndData(path, filename):
    return load_class_data(path, filename, target_at_front=True, to_perceptron_binary=True)

def MSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    return mse

def Accuracy(h, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)

    total = h.shape[0]
    correct = len(np.where(h==y)[0])
    accuracy = correct / total
    return accuracy

def optimizer(optim_name, epsilon=None, gamma=None):
    if optim_name == 'SGD':
        optim = SGD(gamma=gamma, epsilon=epsilon)
    elif optim_name == 'Momentum':
        optim = Momentum(gamma=gamma, epsilon=epsilon)
    elif optim_name == 'RMSProp':
        optim = RMSProp(gamma=gamma, epsilon=epsilon)
    else:
        raise NotImplementedError
    return optim

config = {
    'XOR_mlp': ('xor', MLP, Accuracy),
    'XOR_perceptron': ('xor', Perceptron, Accuracy),
    'Moon': ('moon', MLP, Accuracy),
    'AND': ('and', Perceptron, Accuracy)
}

def _initialize(data_name):
    dir_name, model, metric = config[data_name]
    path = os.path.join('./data', dir_name)

    if data_name == 'XOR_mlp':
        train_x, train_y = XOR_MLP_Data(path, 'train.csv')
        test_x, test_y = XOR_MLP_Data(path, 'test.csv')
    elif data_name == 'XOR_perceptron':
        train_x, train_y = XOR_perceptron_Data(path, 'train.csv')
        test_x, test_y = XOR_perceptron_Data(path, 'test.csv')
    elif data_name == 'Moon':
        train_x, train_y = MoonData(path, 'train.csv')
        test_x, test_y = MoonData(path, 'test.csv')
    elif data_name == 'AND':
        train_x, train_y = AndData(path, 'train.csv')
        test_x, test_y = AndData(path, 'test.csv')
    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y), model, metric


if __name__ == '__main__':
    
    print('\nDATASET TEST FINISHED\n')


    print('OPTIMIZER TEST START\n')

    SGD = optimizer('SGD', 1, 1)
    Momentum = optimizer('Momentum', 1, 1)
    RMSProp = optimizer('RMSProp', 1, 1)

    print('OPTIMIZER TEST FINISHED\n')
