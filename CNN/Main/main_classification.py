import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models.LeNet5 import LeNet5
from models.AlexNet import AlexNet
from models.ResNet18 import ResNet18
from utils import set_random_seed
from torchsummary import summary as summary_


set_random_seed(123)
"""
    Build model architecture and do experiment.
"""
# lenet / alexnet / resnet18
model_name = 'alexnet'
# cifar10 / svhn
dataset = 'svhn'

# Hyper-parameters
num_epochs = 50
learning_rate = 0.001
reg_lambda = 0
batch_size = 100

test_every = 1
print_every = 1

# batch normalization for ResNet18
use_batch_norm = True


def main():
    # Dataset
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    np.random.seed(42)
    # We use random 10% of train data for homework
    num_train = len(trainset)
    num_train_new = int(num_train * 0.1)
    perm = np.random.permutation(num_train)
    train_idx = perm[:num_train_new]
    trainset = torch.utils.data.Subset(trainset, train_idx)

    # split train and valid
    num_train = len(trainset)
    num_valid = int(num_train * 0.1)
    perm = np.random.permutation(num_train)
    valid_idx = perm[:num_valid]
    train_idx = perm[num_valid:]

    validset = torch.utils.data.Subset(trainset, valid_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    num_class = 10
    input_channel = trainset[0][0].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'lenet':
        model = LeNet5(input_channel, num_class, learning_rate, reg_lambda, device)
    elif model_name == 'alexnet':
        model = AlexNet(input_channel, num_class, learning_rate, reg_lambda, device)
    elif model_name == 'resnet18':
        model = ResNet18(input_channel, num_class, learning_rate, reg_lambda, device, use_batch_norm)
    model = model.to(device)

    print(f'Model: {model_name}')
    print(f'Dataset: {dataset}')
    print('Training Starts...')
    model.train_(trainloader, validloader, num_epochs, test_every, print_every)

    # TEST ACCURACY
    model.restore()
    real_y, pred_y = model.predict(testloader)

    correct = len(np.where(pred_y == real_y)[0])
    total = len(pred_y)
    test_acc = correct / total

    print('Test Accuracy (Top-1) at Best Epoch : %.2f' % (test_acc))
    
    model.plot_accuracy()


if __name__ == '__main__':
    main()
