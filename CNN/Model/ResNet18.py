import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# W07 Modern ConvNets.pdf - page 23
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm, stride=1):
        super(BasicBlock, self).__init__()

        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device, use_batch_norm):
        super(ResNet18, self).__init__()

        self.output_dim = output_dim
        self.device = device
        self.loss_function = None
        self.optimizer = None

        self.use_batch_norm = use_batch_norm

        self.CONV1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)
        self.POOL1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.POOL2 = nn.AvgPool2d(kernel_size=(7, 7), stride = 1)
        self.FC1 = nn.Linear(512, output_dim)

        self.Conv_layers = nn.Sequential(self.CONV1, self.BN1, nn.ReLU(), self.POOL1, 
                                        BasicBlock(64, 64, use_batch_norm), BasicBlock(64, 64, use_batch_norm), 
                                        BasicBlock(64, 128, use_batch_norm, stride = 2), BasicBlock(128, 128, use_batch_norm), 
                                        BasicBlock(128, 256, use_batch_norm, stride = 2), BasicBlock(256, 256, use_batch_norm), 
                                        BasicBlock(256, 512, use_batch_norm, stride = 2), BasicBlock(512, 512, use_batch_norm), self.POOL2)

        self.FC_layers = nn.Sequential(self.FC1)

        self.Conv_layers = self.Conv_layers.to(device)
        self.FC_layers = self.FC_layers.to(device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_dim))

        h = self.Conv_layers(x)
        stretched_h = h.reshape(x.shape[0], -1)
        out = self.FC_layers(stretched_h)

        return out

    def predict(self, data_loader):
        self.eval()
        correct_y = []
        pred_y = []
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_x, batch_y = batch_data
                pred = self.forward(batch_x.to(self.device))
                _, predicted = torch.max(pred.data, 1)

                correct_y.append(batch_y.numpy())
                pred_y.append(predicted.cpu().numpy())
        correct_y = np.concatenate(correct_y, axis=0)
        pred_y = np.concatenate(pred_y, axis=0)
        return correct_y, pred_y

    def train_(self, trainloader, validloader, num_epochs, test_every=10, print_every=10):
        self.train_accuracy = []
        self.valid_accuracy = []
        best_epoch = -1
        best_acc = -1
        self.num_epochs = num_epochs
        self.test_every = test_every
        self.print_every = print_every

        total = 0
        correct = 0
        self.train()
        for epoch in range(1, num_epochs+1):
            start = time.time()
            epoch_loss = 0.0
            # model Train
            for b, batch_data in enumerate(tqdm(trainloader, desc="Training")):
                batch_x, batch_y = batch_data
                pred_y = self.forward(batch_x.to(self.device))
                loss = self.loss_function(pred_y, batch_y.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

                _, predicted = torch.max(pred_y.data, 1)
                total += batch_y.size(0)
                correct += (predicted.cpu() == batch_y).sum().item()

            epoch_loss /= len(trainloader)
            end = time.time()
            lapsed_time = end - start

            if epoch % print_every == 0:
                print(f'Epoch {epoch} took {lapsed_time} seconds\n')
                print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))

            if epoch % test_every == 0:
                # TRAIN ACCURACY
                train_acc = correct / total
                self.train_accuracy.append(train_acc)

                # VAL ACCURACY
                real_y, pred_y = self.predict(validloader)
                correct = (pred_y == real_y).sum().item()
                total = len(pred_y)
                valid_acc = correct / total
                self.valid_accuracy.append(valid_acc)

                if best_acc < valid_acc:
                    best_acc = valid_acc
                    best_epoch = epoch
                    torch.save(self.state_dict(), f'./best_model/ResNet18_bn_{self.use_batch_norm}.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

        return best_acc

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_function(pred, y_a) + (1 - lam) * self.loss_function(pred, y_b)

    def restore(self):
        with open(os.path.join(f'./best_model/ResNet18_bn_{self.use_batch_norm}.pt'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def plot_accuracy(self):
        """
            Draw a plot of train/valid accuracy.
            X-axis : Epoch
            Y-axis : train_accuracy & valid_accuracy
            Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
        """
        epochs = list(np.arange(1, self.num_epochs+1, self.print_every, dtype=np.int32))

        plt.plot(epochs, self.train_accuracy, label='Train Acc.')
        plt.plot(epochs, self.valid_accuracy, label='Valid Acc.')

        plt.title('Epoch - Train/Valid Acc.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig('ResNet18.png')
        plt.show()
