import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

class RNN_classifier(nn.Module):
    def __init__(self, device, embed_dim, hidden_dim, num_classes=7):
        super(RNN_classifier, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-small-v3-discriminator')
        self.tokenizer.padding_side = 'left'

        self.build_model()

        self.to(device)

    
    def build_model(self):

        """
        self.word_embedding: (vocab_size, embed_dim)
        self.layer1: (hidden_dim, 1)
        """

        self.word_embedding = nn.Embedding(num_embeddings = len(self.tokenizer), embedding_dim = self.embed_dim)
        self.RNN = nn.RNN(input_size = self.embed_dim, hidden_size = self.hidden_dim, num_layers = 1, batch_first = True)
        self.layer1 = nn.Linear(self.hidden_dim, self.num_classes)  

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, batch_text_ids):

        """
        batch_text_ids: (batch_size, max_length)    
        output: (1, batch_size, 7)
        """

        embeddings = self.word_embedding(batch_text_ids)
        output, hidden = self.RNN(embeddings)
        output = self.layer1(output[:, -1])

        return output.squeeze()


    def train_model(self, X_train, X_valid, y_train, y_valid, num_epochs, batch_size, learning_rate):

        self.optimizer = AdamW(self.parameters(), lr=learning_rate)
        y_train = np.array(y_train)

        loss_log = []
        for e in range(num_epochs):
            epoch_loss = 0
            batch_loader = DataBatcher(np.arange(len(X_train)), batch_size=batch_size)
            for b, batch_indices in enumerate(tqdm(batch_loader, desc=f'> {e+1} epoch training ...', dynamic_ncols=True)):
                self.optimizer.zero_grad()
                batch_text = [X_train[idx] for idx in batch_indices]

                result = self.tokenizer(batch_text, padding = True,  return_tensors = 'pt') 
                out = self.forward(result['input_ids'].to(self.device))

                batch_labels = torch.Tensor(y_train[batch_indices]).long().to(self.device)
                loss = self.loss_function(out, batch_labels)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            loss_log.append(epoch_loss)
            
            valid_accuracy, valid_loss = self.predict(X_valid, y_valid, batch_size)
            print(f'>> [Epoch {e+1}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_accuracy:.2f}% / Valid loss: {valid_loss:.4f}')
  
    def predict(self, X, y, batch_size, return_preds=False):
        y = np.array(y)
        preds = torch.zeros(len(X)).to(self.device)
        total_loss = 0

        with torch.no_grad():
            batch_loader = DataBatcher(np.arange(len(X)), batch_size=batch_size)
            for batch_num, batch_indices in enumerate(tqdm(batch_loader, desc=f'> Predicting ...', dynamic_ncols=True)):
                batch_text = [X[idx] for idx in batch_indices]

                result = self.tokenizer(batch_text, padding = True,  return_tensors = 'pt') 
                out = self.forward(result['input_ids'].to(self.device))
                
                batch_labels = torch.Tensor(y[batch_indices]).long().to(self.device)
                loss = self.loss_function(out, batch_labels)
                total_loss += loss
                max_vals, max_indices = torch.max(out,1)
                preds[batch_indices] = max_indices.float()
            labels = torch.Tensor(y).to(self.device)
                

            accuracy = (preds == labels).sum().data.cpu().numpy() / y.shape[0]

        
        if return_preds:
            return accuracy, total_loss, preds.detach().cpu().numpy()
        else:
            return accuracy, total_loss