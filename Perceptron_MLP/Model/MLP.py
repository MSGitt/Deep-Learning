import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader



class MLP(nn.Module):
    def __init__(self, num_features, num_hidden):
        super().__init__()

        self.num_features = num_features
        self.num_hidden = num_hidden
        
        self.model = nn.Sequential(
            nn.Linear(self.num_features, self.num_hidden),
            nn.Tanh() ,
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Tanh() ,
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Tanh() ,
            nn.Linear(self.num_hidden, 1),
            nn.Sigmoid() )
        
        


    def train(self, x, y, epochs, batch_size, lr, optim):     

        x_train = torch.Tensor(x)
        y_train = torch.Tensor(y)

        dataset = list(zip(x_train, y_train)) 
        dataloader = DataLoader(dataset = dataset , batch_size = batch_size, shuffle = False)
        
        criterion = nn.BCELoss() 
        

        for k in range(epochs) :

            loss = 0

            for data, labels in dataloader :

                batch_x = data
                batch_y = labels

                optim.zero_grad()

                out = self.model(batch_x) 
                out = out.squeeze() 

                losses = criterion(out, batch_y) 
                losses.backward()
                optim.step()  

                loss += losses.item()
        
        return loss

    
    def forward(self, x):
        
        x_test = torch.Tensor(x)
        output = self.model(x_test)  

        y_predicted = np.array([0 if i < 0.5 else 1 for i in output])
        y_predicted = np.expand_dims(y_predicted, 1)

        return y_predicted



