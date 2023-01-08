import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from tqdm import tqdm

# W06-2 Convolutional Neural Networks (CNNs).pdf - page 7
class AlexNet(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device):
        super(AlexNet, self).__init__()

        self.output_dim = output_dim
        self.device = device
        
        self.CONV1 = nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=(11, 11), stride=4)
        self.CONV2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding = 2)
        self.CONV3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding = 1)
        self.CONV4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding = 1)
        self.CONV5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding = 1)

        self.POOL1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.POOL2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.POOL3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.FC1 = nn.Linear(9216, 4096)
        self.FC2 = nn.Linear(4096, 4096)
        self.FC3 = nn.Linear(4096, output_dim)

        self.Conv_layers = nn.Sequential(self.CONV1, nn.ReLU(), self.POOL1, self.CONV2, nn.ReLU(), self.POOL2, 
                                        self.CONV3, nn.ReLU(), self.CONV4, nn.ReLU(), self.CONV5, nn.ReLU(), self.POOL3)
        self.FC_layers = nn.Sequential(self.FC1, nn.ReLU(), self.FC2, nn.ReLU(), self.FC3)

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
                batch_x = resize(batch_x, (227, 227))
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
                batch_x = resize(batch_x, (227, 227))
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
                    torch.save(self.state_dict(), './best_model/AlexNet.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

        return best_acc

    def restore(self):
        with open(os.path.join('./best_model/AlexNet.pt'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def plot_accuracy(self):
        """
            Draw a plot of train/valid accuracy.
            X-axis : Epoch
            Y-axis : train_accuracy & valid_accuracy
            Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
        """
        epochs = list(np.arange(1, self.num_epochs+1, self.test_every))

        plt.plot(epochs, self.train_accuracy, label='Train Acc.')
        plt.plot(epochs, self.valid_accuracy, label='Valid Acc.')

        plt.title('Epoch - Train/Valid Acc.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig('AlexNet.png')
        plt.show()
        