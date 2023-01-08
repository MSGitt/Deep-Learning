import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from utils import *
from tqdm import tqdm
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, device):
        super().__init__()
        self.device = device
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first=True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):

        embedded = self.dropout(self.embedding(src))
        
        outputs, hidden = self.rnn(embedded)
        outputs= outputs.permute(1,0,2)


        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, device):
        super().__init__()
        self.device = device 

        self.fc1 = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim, 1, bias = False)

        
    def forward(self, hidden, encoder_outputs):

        # _____________________ version 1 _________________________
        
        # src_len = encoder_outputs.shape[0]
        # batch_size = encoder_outputs.shape[1] 
        # dim1 = (encoder_outputs.shape[2]) // 2

        # encoder_outputs = encoder_outputs[:, :, : dim1]

        # hidden = hidden.unsqueeze(dim = 1)
        # encoder_outputs = encoder_outputs.permute(1, 2, 0) 

        # energy = torch.matmul(hidden, encoder_outputs) 
        # attention = energy.squeeze(dim = 1) 

        # attention_weight = torch.softmax(attention, dim = 1)

    
        # _____________________ version 2 _________________________

        # src_len = encoder_outputs.shape[0]
        # batch_size = encoder_outputs.shape[1] 

        # dim1 = hidden.shape[1]
        # dim2 = encoder_outputs.shape[2] 

        # w = torch.randn(dim1, dim2).to(self.device)  

        # result = torch.matmul(hidden, w) 

        # result = result.unsqueeze(dim = 1)   
        # encoder_outputs = encoder_outputs.permute(1, 2, 0)

        # energy = torch.matmul(result, encoder_outputs)
        # attention = energy.squeeze(dim = 1) 
        

        # attention_weight = torch.softmax(attention, dim = 1)

        # _____________________ version 3 _________________________

        # src_len = encoder_outputs.shape[0]
        # batch_size = encoder_outputs.shape[1] 

        # hidden = hidden.unsqueeze(dim = 1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy = torch.tanh(self.fc1(torch.cat((hidden, encoder_outputs), dim = 2)))
        # attention = self.fc2(energy).squeeze(dim = 2) 

        # attention_weight = torch.softmax(attention, dim = 1)
       

        return attention_weight

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, device):
        super().__init__()
        self.device = device

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input)) # [1, batch, hidden]
        
        a = self.attention(hidden, encoder_outputs) # [batch, length]
        a = a.unsqueeze(1) # [batch, 1, length]

        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # a = a.permute(2,1,0)
        
        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)

class Attention_seq2seq(nn.Module):
    def __init__(self, device, hidden_dim, SRC, TRG):
        super(Attention_seq2seq, self).__init__()
        self.device = device

        self.src_vocab_size = len(SRC.vocab)
        self.trg_vocab_size = len(TRG.vocab)
        

        self.SRC_field = SRC
        self.TRG_field = TRG
        self.src_pad_idx = self.SRC_field.vocab.stoi[SRC.pad_token]
        self.trg_pad_idx = self.TRG_field.vocab.stoi[TRG.pad_token]
        
        self.hidden_dim = hidden_dim
        self.attn = Attention(self.hidden_dim, self.hidden_dim, device=self.device)
        
        self.build_model()

    def build_model(self):
        self.encoder = Encoder(self.src_vocab_size, self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=0.5, device=self.device)
        self.decoder = Decoder(self.trg_vocab_size, self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=0.5, attention=self.attn, device=self.device)

        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]


        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        # src = src.to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        #first input to the decoder is the  tokens

        # input = trg[0,:]
        input = trg[:,0]
        

        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state

            output, hidden = self.decoder(input, hidden, encoder_outputs)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:,t] if teacher_force else top1


        return outputs

    def train_model(self, num_epochs, learning_rate, train_iterator, valid_iterator, clip=1):
        CE_loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for e in range(num_epochs):
            epoch_loss = 0
            self.train()

            for i, batch in enumerate(tqdm(train_iterator, desc=f'> {e+1} epoch training ...', dynamic_ncols=True)):
                optimizer.zero_grad()
                src = batch.src
                trg = batch.trg
            
                output = self.forward(src, trg)

                trg = trg.permute(1,0)
                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)

                loss = CE_loss(output, trg)
                loss.backward()
            
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss = epoch_loss / len(train_iterator)
            valid_loss = self.predict(valid_iterator)
            print(f'>> [Epoch {e+1}] Epoch loss: {epoch_loss:.3f} / Valid loss: {valid_loss:.3f}')



    def predict(self, iterator):
        CE_loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.eval()
        
        total_loss = 0
        
        with torch.no_grad():
        
            for batch in tqdm(iterator, desc=f'> Predicting ...', dynamic_ncols=True):
                src = batch.src
                trg = batch.trg

                output = self.forward(src, trg, 0) #turn off teacher forcing
                
                trg = trg.permute(1,0)
                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)

                loss = CE_loss(output, trg)

                total_loss += loss.item()
            
            loss = total_loss / len(iterator)
            
        return loss
    
    

    def translate(self, iterator):
    # eval 모드로 변경합니다. dropout 등의 함수를 비활성화합니다.
        self.eval()
        
        epoch_loss = 0

        source = []
        target = []
        prediction = []
        eos = self.TRG_field.vocab.stoi[self.TRG_field.eos_token]

        with torch.no_grad():
            for i, batch in enumerate(iterator):

                src = batch.src
                trg = batch.trg

                # 공정한 평가를 위해 teacher forcing 을 사용하지 않습니다.
                output = self.forward(src, trg, 0) 
                output = output.permute(1,0,2)
                output = output[:, 1:]
                pred = output.argmax(-1)
                trg = trg[:, 1:]
                

                src = src.detach().cpu().numpy().tolist()
                src = [[self.SRC_field.vocab.itos[w] for w in truncate_after_eos(s,eos)] for s in src]

                trg = trg.detach().cpu().numpy().tolist()
                trg = [[self.TRG_field.vocab.itos[w] for w in truncate_after_eos(s,eos)] for s in trg]

                pred = pred.detach().cpu().numpy().tolist()
                pred = [[self.TRG_field.vocab.itos[w] for w in truncate_after_eos(s,eos)] for s in pred]

                source += src
                target += trg
                prediction += pred

        return source, target, prediction

    