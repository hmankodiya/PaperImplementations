import numpy as np
import pandas as pd
import os
import unicodedata
import re

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import pytorch_lightning as pl

import utils as ut
import language as lg


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD'}
        self.word2count = {}
        self.n_words = 3  # Count SOS and EOS

    def addsentence(self, sentence):
        for word in sentence.split(' '):
            self.addword(word)

    def addword(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 0
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1
            
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def normalize(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r'', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

def index_from_sentence(lang, sentence):
    index_array = []
    for word in sentence:
        assert word in lang.word2index, f'word {word} not found in language {lang.name}'
        index_array.append(lang.word2index[word])
    return np.array(index_array)

class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, input_language, target_language, max_sequence_length):
        self.input_language = input_language
        self.target_language = target_language
        self.sentences = sentences
        self.sentences['Input'] = self.sentences['Input'].apply(lambda x: normalize(x.strip()))
        self.max_sequence_length = max_sequence_length
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        input_raw_text = self.sentences['Input'][index].strip().split(' ')
        input_indices = index_from_sentence(self.input_language, input_raw_text)
        input_indices = np.pad(input_indices,
                                      pad_width=(0, 1),
                                      mode='constant',
                                      constant_values=self.input_language.word2index['EOS'])
        
        return input_indices 

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, input_language, target_language, max_sequence_length):
        self.input_language = input_language
        self.target_language = target_language
        self.max_sequence_length = max_sequence_length
        self.pairs = pairs
        self.pairs['Input'] = self.pairs['Input'].apply(lambda x: normalize(x.strip()))
        self.pairs['Target'] = self.pairs['Target'].apply(lambda x: normalize(x.strip()))
        for _, values in enumerate(self.pairs.values):
            self.input_language.addsentence(values[0])
            self.target_language.addsentence(values[1])
            
        print(f'input language: {self.input_language.name}, unique words found {self.input_language.n_words}\ntarget language: {self.target_language.name}, unique words found {self.target_language.n_words}')
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, index):
        input_raw_text, target_raw_text = self.pairs['Input'][index].strip().split(' '), self.pairs['Target'][index].strip().split(' ')
        input_indices, target_indices = index_from_sentence(self.input_language, input_raw_text), index_from_sentence(self.target_language, target_raw_text)
        
        input_indices = np.pad(input_indices,
                               pad_width=(1, 1),
                               mode='constant',
                               constant_values=(self.input_language.word2index['SOS'], self.input_language.word2index['EOS'])) 
        target_indices = np.pad(target_indices, 
                                pad_width=(1, 1),
                                mode='constant',
                                constant_values=(self.target_language.word2index['SOS'], self.target_language.word2index['EOS']))
                
        return [torch.from_numpy(input_indices), torch.from_numpy(target_indices)]