import torch
import torch.nn as nn

from utils import *

from tqdm import tqdm
from torch.optim import Adam


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, device):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Define Query Key Value for Multi-Head Attention
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        '''
        query = [batch_size, query_len, hidden_dim]
        key   = [batch_size, key_len, hidden_dim]
        value = [batch_size, value_len, hidden_dim]
        mask  = [batch_size, 1, 1, value_len]
        '''

        batch_size = query.shape[0]

        # Convert query, key, value to Q, K, V, respectively.
        # Q = [batch_size, query_len, hidden_dim]
        # K = [batch_size, key_len, hidden_dim]
        # V = [batch_size, value_len, hidden_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Divide hidden dimension by number of heads.
        # Q = [batch_size, query_len, num_heads,  head_dim]
        # K = [batch_size, key_len, num_heads, head_dim]
        # V = [batch_size, value_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        # Q = [batch_size, num_heads, query_len, head_dim]
        # K = [batch_size, num_heads, key_len, head_dim]
        # V = [batch_size, num_heads, value_len, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # QK^T
        # hint1 : [A, B, C, D] @ [A, B, D, E] = [A, B, C, E]
        # hint2 : Use permute() to change the dimension.
        # K = [batch_size, num_heads, head_dim, key_len]
        # energy = [batch_size, num_heads, query len, key len]
        K = K.permute(0, 1, 3, 2)
        energy = torch.matmul(Q, K)
        energy = energy / self.scale
        
        # Make energy at <pad> zero for masking.
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Take Softmax function to make the sum of attention 1.
        # attention = [batch_size, num_heads, query_len, key_len]
        attention = torch.softmax(energy, dim = -1)
        attention = self.dropout(attention) 

        # Multiply attention with V.
        # As key and value are same, their lenghths are always same.
        # x = [batch_size, num_heads, query_len, head_dim]
        x = torch.matmul(attention, V)

        # x = [batch_size, query_len, num_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch_size, query_len, hidden_dim]
        x = x.view(batch_size, -1, self.hidden_dim)

        # x = [batch_size, query_len, hidden_dim]
        x = self.fc_o(x)

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, device)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Linear(hidden_dim, hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
                                                                     
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, src, src_mask):
        
        '''
        src      = [batch_size, src_len, hidden_dim]
        src_mask = [batch_size, src_len]
        '''

        # Take self attention.
        # att_src = [batch_size, src_len, hidden_dim]
        att_src, _ = self.self_attention(src, src, src, src_mask)
        att_src = self.dropout(att_src)

        # residual connection
        # att_src = [batch_size, src_len, hidden_dim]
        att_src = src + att_src
        
        # layer normalization
        # att_src = [batch_size, src_len, hidden_dim]
        att_src = self.self_attn_layer_norm(att_src)
                
        # feedforward layer
        # ff_src = [batch size, src len, hid dim]
        ff_src = self.feedforward(att_src)
        ff_src = self.dropout(ff_src)

        # residual connection
        # ff_src = [batch size, src len, hid dim]
        ff_src = att_src + ff_src

        # layer normalization        
        # ff_src = [batch size, src len, hid dim]
        ff_src = self.ff_layer_norm(ff_src)
        
        return ff_src

class Encoder_Tr(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, device, max_length=100):
        super().__init__()

        self.device = device
        
        # Embedding matrix for tokens(words)
        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        # Embedding matrix for positional encoding
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Repeat encoder layer num_layers times
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, device) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(0.5)
        
        # It is a method introduced in the paper for stable learning
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        
    def forward(self, src, src_mask):

        '''
        src      = [batch_size, src_len] : source text
        src_mask = [batch_size, src_len] : 0 for <pad> and 1 for the others.
        '''

        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Convert token indices to embeddings.
        # tok_emb = [batch_size, src_len, hidden_dim]
        tok_emb = self.tok_embedding(src) * self.scale

        # pos = [batch size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos_emb = [batch size, src_len, hidden_dim]
        pos_emb = self.pos_embedding(pos)
        
        # Add position embedding to token embedding.
        # src = [batch size, src_len, hidden_dim]
        src = self.dropout(tok_emb + pos_emb)
                
        for layer in self.layers:
          #src = [batch size, src len, hid dim]
          src = layer(src, src_mask)

        return src

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, device)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)

        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, device)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)

        self.feedforward = nn.Linear(hidden_dim, hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        '''
        trg      = [batch size, trg len, hid dim]
        enc_src  = [batch size, src len, hid dim]
        trg_mask = [batch size, trg len]
        src_mask = [batch size, src len]
        '''

        # Take self attention for target sentence.
        att_trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        att_trg = self.dropout(att_trg)
        
        # residual connection and layer norm
        att_trg = self.self_attn_layer_norm(trg + att_trg)

        # Take self attention for target sentence with key, value from source sentence.
        enc_att_trg, attention = self.encoder_attention(att_trg, enc_src, enc_src, src_mask)
        enc_att_trg = self.dropout(enc_att_trg)

        # residual connection and layer norm
        enc_att_trg = self.enc_attn_layer_norm(att_trg + enc_att_trg)
                            
        # feed-forward
        ff_trg = self.feedforward(enc_att_trg)
        ff_trg = self.dropout(ff_trg)
        
        # residual and layer norm
        ff_trg = self.ff_layer_norm(trg + ff_trg)

        return ff_trg, attention

class Decoder_Tr(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, device, max_length=100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):

        '''
        trg      = [batch_size, trg len]          : Target text
        enc_src  = [batch_size, src_len, hidden_dim] : Output of encoder 
        trg_mask = [batch_size, trg_len]          : Mask for target text
        src_mask = [batch_size, src_len]          : Nask for source text
        '''

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        tok_emb = self.tok_embedding(trg) * self.scale

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_emb = self.pos_embedding(pos)

        trg = self.dropout(tok_emb + pos_emb)
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
                    
        return output, attention

class Transformer_seq2seq(nn.Module):
    def __init__(self, device, hidden_dim, num_enc_layers, num_dec_layers, num_enc_heads, num_dec_heads, SRC, TRG):
        super(Transformer_seq2seq, self).__init__()
        self.device = device

        self.SRC_field = SRC
        self.TRG_field = TRG

        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_enc_heads = num_enc_heads
        self.num_dec_heads = num_dec_heads

        self.src_pad_idx = self.SRC_field.vocab.stoi[SRC.pad_token]
        self.trg_pad_idx = self.TRG_field.vocab.stoi[TRG.pad_token]
        self.src_vocab_size = len(SRC.vocab)
        self.trg_vocab_size = len(TRG.vocab)
        
        self.build_model()

        self.to(device)

    def build_model(self):
        self.encoder = Encoder_Tr(self.src_vocab_size, self.hidden_dim, self.num_enc_layers, self.num_enc_heads, self.device)
        self.decoder = Decoder_Tr(self.trg_vocab_size, self.hidden_dim, self.num_dec_layers, self.num_dec_heads, self.device)

    # source mask: 0 for <pad> token and 1 for the rest in source text
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask
    
    # target mask: 0 for <pad> token, 1 for the rest in source text, and 0 for all after time t.
    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
                
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):

        '''
        src = [batch size, src len]
        trg = [batch size, trg len]
        '''

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        outputs, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return outputs, attention

    def train_model(self, num_epochs, learning_rate, train_iterator, valid_iterator, clip=1):
        CE_loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)

        optimizer = Adam(self.parameters(), lr=learning_rate)

        for e in range(num_epochs):
            epoch_loss = 0
            self.train()
            for i, batch in enumerate(tqdm(train_iterator, desc=f'> {e+1} epoch training ...', dynamic_ncols=True)):
                optimizer.zero_grad()
                src = batch.src
                trg = batch.trg
                output, _ = self.forward(src, trg[:, :-1])

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)

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
                output, _ = self.forward(src, trg[:, :-1])

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)

                loss = CE_loss(output, trg)

                total_loss += loss.item()

            loss = total_loss / len(iterator)

        return loss

    def translate_sentence(self, sentence, max_len=50):
        self.eval()
        tokens = [token.lower() for token in sentence]

        tokens = [self.SRC_field.init_token] + tokens + [self.SRC_field.eos_token]
        src_indexes = [self.SRC_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)
        src_mask = self.make_src_mask(src_tensor)
        with torch.no_grad():
            enc_src = self.encoder(src_tensor, src_mask)

        trg_indexes = [self.TRG_field.vocab.stoi[self.TRG_field.init_token]]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)

            trg_mask = self.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, _ = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.TRG_field.vocab.stoi[self.TRG_field.eos_token]:

                break
        trg_tokens = [self.TRG_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:-1]