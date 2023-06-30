from typing import Any
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import utils as ut
import language as lg


class PositionalEmebeddings:
    def __init__(self,
                 embeddings_size,
                 max_sequence_length):
        den = torch.exp(-torch.arange(0, embeddings_size, 2)*np.log(10000)/embeddings_size)
        pos = torch.arange(0, max_sequence_length).reshape(max_sequence_length, 1)
        self.pos_embedding = torch.zeros((max_sequence_length, embeddings_size))
        self.pos_embedding[:, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = self.pos_embedding.unsqueeze(0)

class EncoderLayer(nn.Module):
    def __init__(self, input_embeddings_size, input_vocab_size, num_heads, max_sequence_length, dropout_prob = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.input_embeddings_size = input_embeddings_size
        self.input_vocab_size = input_vocab_size
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.max_sequence_length = max_sequence_length
        
        self.layer_norm = nn.LayerNorm(self.input_embeddings_size)
        self.embeddings = nn.Embedding(num_embeddings=self.input_vocab_size, embedding_dim=self.input_embeddings_size)
        self.pos = PositionalEmebeddings(embeddings_size=self.input_embeddings_size,
                                         max_sequence_length = self.max_sequence_length)
        self.multihead_attention = nn.MultiheadAttention(embed_dim = self.input_embeddings_size, 
                                                         num_heads = num_heads, 
                                                         dropout = self.dropout_prob,
                                                         batch_first = True)
        self.linear = nn.Linear(self.input_embeddings_size, self.input_embeddings_size)
        
    def forward(self, batch_input, padding_mask):
        x = self.embeddings(batch_input)
        x += self.pos.pos_embedding[:, :batch_input.shape[1]].to(device=x.device)
        attn_outputs, _ = self.multihead_attention(x, x, x, key_padding_mask=padding_mask)
        x = self.layer_norm(x + attn_outputs)
        enc_output = self.layer_norm(x + self.linear(x))        
        
        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, target_embeddings_size, target_vocab_size, num_heads, max_sequence_length, dropout_prob = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.target_embeddings_size = target_embeddings_size
        self.target_vocab_size = target_vocab_size
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.max_sequence_length = max_sequence_length

        self.layer_norm = nn.LayerNorm(self.target_embeddings_size)
        self.embeddings = nn.Embedding(num_embeddings =self.target_vocab_size, embedding_dim=self.target_embeddings_size)
        self.pos = PositionalEmebeddings(embeddings_size=self.target_embeddings_size,
                                         max_sequence_length=self.max_sequence_length)
        self.multihead_attention_1 = nn.MultiheadAttention(embed_dim=self.target_embeddings_size, 
                                                         num_heads=num_heads, 
                                                         dropout=self.dropout_prob,
                                                         batch_first=True)
        
        self.multihead_attention_2 = nn.MultiheadAttention(embed_dim=self.target_embeddings_size, 
                                                         num_heads=num_heads, 
                                                         dropout=self.dropout_prob,
                                                         batch_first=True)
        
        self.linear = nn.Linear(self.target_embeddings_size, self.target_embeddings_size)
        
    def forward(self, batch_input_embds, batch_target, self_attn_padding_mask, self_attn_mask, cross_attn_padding_mask):
        x = self.embeddings(batch_target)
        x += self.pos.pos_embedding[:, :batch_target.shape[1]].to(device=x.device)
        attn_outputs_1, _ = self.multihead_attention_1(query=x, key=x, value=x, 
                                                       attn_mask=self_attn_mask,
                                                       key_padding_mask=self_attn_padding_mask)
        x = self.layer_norm(x + attn_outputs_1)
        attn_outputs_2, _ = self.multihead_attention_2(query=x, key=batch_input_embds, value=batch_input_embds, 
                                                       key_padding_mask=cross_attn_padding_mask)
        x = self.layer_norm(x + attn_outputs_2)
        dec_output = self.layer_norm(x + self.linear(x))
        
        return dec_output    
    
class Model(pl.LightningModule):
    def __init__(self, encoder, decoder, max_sequence_length, input_language=None, target_language=None, optimizer=torch.optim.Adam):
        super(Model, self).__init__()
        self.input_language = input_language
        self.target_language = target_language
        
        self.encoder = encoder
        self.decoder = decoder
        self.embeddings_size = self.encoder.input_embeddings_size
        self.target_vocab_size = self.decoder.target_vocab_size
        self.linear = nn.Linear(self.embeddings_size, self.target_vocab_size)        
        
        self.input_vocab_size = self.encoder.input_vocab_size
        self.target_vocab_size = self.decoder.target_vocab_size
        self.embeddings_size = self.encoder.input_embeddings_size
        self.max_sequence_length = max_sequence_length
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_language.word2index['PAD'])
        self.optimizer = optimizer
        
    def create_mask(self, input_batch, target_batch):
        input_sentence_length, target_sentence_length = input_batch.shape[1], target_batch.shape[1]
        
        input_padding_mask = (input_batch == self.input_language.word2index['PAD']).to(device=input_batch.device)
        target_padding_mask = (target_batch == self.target_language.word2index['PAD']).to(device=input_batch.device)
        target_self_attn_mask = torch.full((target_sentence_length, target_sentence_length), True).to(device=input_batch.device)
        target_self_attn_mask = torch.triu(target_self_attn_mask, diagonal = 1).to(device=input_batch.device)

        return input_padding_mask, target_padding_mask, target_self_attn_mask
        
        
    def forward(self, batch):
        input_batch, target_batch = batch[0], batch[1]
        input_padding_mask, target_padding_mask, target_self_attn_mask = self.create_mask(input_batch=input_batch,
                                                                                          target_batch=target_batch)
        
        encoder_output = self.encoder(input_batch, padding_mask=input_padding_mask)
        decoder_output = self.decoder(encoder_output, target_batch,
                                      self_attn_padding_mask=target_padding_mask, self_attn_mask=target_self_attn_mask,
                                      cross_attn_padding_mask=input_padding_mask)
        y_hat = self.linear(decoder_output)
        
        return y_hat
    
    def predict_step(self, batch, batch_idx=None, ground_truth_tokens=None) :
        target_tokens = torch.ones(1, 1).fill_(self.target_language.word2index['SOS']).type(torch.int64)
        encoder_output = self.encoder(batch, padding_mask=None)
        word_counter = 1
        
        while word_counter < self.max_sequence_length:
            target_token_length = target_tokens.shape[1]
            target_self_attn_mask = torch.full((target_token_length, target_token_length), True).to(device=batch.device)
            target_self_attn_mask = torch.triu(target_self_attn_mask, diagonal = 1).to(device=batch.device)
            
            decoder_output = self.decoder(encoder_output, target_tokens,
                                          self_attn_padding_mask=None, self_attn_mask=target_self_attn_mask, 
                                          cross_attn_padding_mask=None)
            y_hat = self.linear(decoder_output)
            y_hat_norm = F.softmax(y_hat, dim=-1).argmax(dim=-1)
            target_tokens = torch.cat((target_tokens, torch.ones(1, 1).fill_(y_hat_norm[0, -1]).type(torch.int64)), dim=1)

            if self.target_language.index2word[y_hat_norm[:, -1].item()] == 'EOS':
                break
            
            word_counter += 1
        
        if ground_truth_tokens:
            return target_tokens, ground_truth_tokens
         
        return target_tokens
        
    def training_step(self, batch, batch_idx=None):
        y_hat = self.forward(batch)[:, :-1]
        loss = self.criterion(torch.reshape(y_hat, (-1, self.target_language.n_words)), torch.reshape(batch[1][:, 1:], (-1, )))
        
        return {'loss': loss}
    
    def configure_optimizers(self):        
        return  self.optimizer(self.parameters(), lr=0.001)
    

def predict(model, batch, ground_truth_tokens=None) : # batch prediction not possible
    target_tokens = torch.ones(1, 1).fill_(model.target_language.word2index['SOS']).type(torch.int64)
    encoder_output = model.encoder(batch, padding_mask=None)
    word_counter = 1
    
    while word_counter < model.max_sequence_length:
        target_token_length = target_tokens.shape[1]
        target_self_attn_mask = torch.full((target_token_length, target_token_length), True).to(device=batch.device)
        target_self_attn_mask = torch.triu(target_self_attn_mask, diagonal = 1).to(device=batch.device)
        
        decoder_output = model.decoder(encoder_output, target_tokens,
                                        self_attn_padding_mask=None, self_attn_mask=target_self_attn_mask, 
                                        cross_attn_padding_mask=None)
        y_hat = model.linear(decoder_output)
        y_hat_norm = F.softmax(y_hat, dim=-1).argmax(dim=-1)
        target_tokens = torch.cat((target_tokens, torch.ones(1, 1).fill_(y_hat_norm[0, -1]).type(torch.int64)), dim=1)

        if model.target_language.index2word[y_hat_norm[:, -1].item()] == 'EOS':
            break
        
        word_counter += 1
    
    if ground_truth_tokens:                    
        return target_tokens, ground_truth_tokens
    
    return target_tokens