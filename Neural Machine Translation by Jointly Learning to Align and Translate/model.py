from __future__ import unicode_literals, print_function, division
import language as lang

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import bleu_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = '.././data/'
MAX_LENGTH = 10


class SentenceData(pl.LightningDataModule):
    '''
        The batches gets loaded in a stochastic way.
        Fixed batch size: 1
    '''

    def __init__(self, input_lang, output_lang) -> None:
        super(SentenceData, self).__init__()
        self.data_dir = DATA
        self.input_lang = input_lang
        self.output_lang = output_lang

    def prepare_data(self, pairs, test_split=0.2, val=True, val_split=0.2) -> None:
        print(
            f'output language - {self.output_lang.name}, input language - {self.input_lang.name}')
        self.tensor_data = []
        for i in pairs:
            self.tensor_data.append(lang.tensorsFromPair(
                self.input_lang, self.output_lang, i))
        total_len = len(self.tensor_data)
        self.train_size = int(total_len*(1-test_split))
        self.test_size = total_len-int(total_len*(1-test_split))
        if val:
            self.train_size, self.val_size = int(
                self.train_size*(1-val_split)), self.train_size - int(self.train_size*(1-val_split))
            self.train, self.val, self.test = torch.utils.data.random_split(self.tensor_data, [self.train_size,
                                                                                               self.val_size,
                                                                                               self.test_size])
        else:
            self.train, self.test = torch.utils.data.random_split(self.tensor_data, [self.train_size,
                                                                                     self.test_size])

    def train_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.train, **kwargs)

    def val_dataloader(self, **kwargs):
        assert self.val is not None, 'validation set not constructed'
        return torch.utils.data.DataLoader(self.val, **kwargs)

    def test_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.test, **kwargs)

    def predict_dataloader(self, input_language, sentences, **kwargs):
        sentences_tensors = []
        for i in sentences:
            sentences_tensors.append((lang.tensorFromSentence(input_language, i), i))
        return torch.utils.data.DataLoader(sentences_tensors, **kwargs)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), dim=1)),
                                 dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(
            0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class Model(pl.LightningModule):
    def __init__(self, input_lang, output_lang, encoder, decoder, criterion, optimizer, learning_rate=0.01):

        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.input_lang = input_lang
        self.output_lang = output_lang
    
    def validation_step(self, batch, batch_idx):

        input_tensor, target_tensor = batch[0].view(-1, 1), batch[1].view(-1, 1)
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(
            MAX_LENGTH, self.encoder.hidden_size, device=DEVICE)
        encoder_hidden = self.encoder.initHidden()
                
        for ei in range(input_length):
            etensor = input_tensor[ei]
            encoder_output, encoder_hidden = self.encoder(
                etensor, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[lang.SOS_token]], dtype=torch.int64, device=DEVICE)
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == lang.EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.output_lang.index2word[topi.item()])
            decoder_input = topi
        
        decoded_sentence = ' '.join(decoded_words)
        target_sentence = []
        for tari in target_tensor:
            target_sentence.append(self.output_lang.index2word[tari.item()])
        target_sentence = ' '.join(target_sentence)
        
        di = {f'bleu': bleu_score([target_sentence], [decoded_sentence])}
        self.log_dict(dictionary=di, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch=batch, batch_idx=batch_idx)
        
    def training_step_end(self, step_output):
        # print('step outputs', step_output)
        train_loss_step = step_output['loss']
        self.log_dict({'train_loss_step': train_loss_step})

    def training_epoch_end(self, outputs):
        avg_bleu_score = torch.as_tensor([i['train_bleu_1'] for i in outputs]).mean()
        train_loss = torch.as_tensor([i['loss'] for i in outputs]).mean()
        self.log_dict({"train_loss_epoch":train_loss, "train_avg_bleu_1_score": avg_bleu_score}, prog_bar=True)

    def training_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch[0].view(-1, 1), batch[1].view(-1, 1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(
            MAX_LENGTH, self.encoder.hidden_size, device=DEVICE)
        encoder_hidden = self.encoder.initHidden()
        loss = 0

        for ei in range(input_length):
            etensor = input_tensor[ei]
            encoder_output, encoder_hidden = self.encoder(
                etensor, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[lang.SOS_token]], dtype=torch.int64, device=DEVICE)
        decoder_hidden = encoder_hidden
        
        target_sentence, output_sentence = [], []
        for tari in target_tensor:
            target_sentence.append(self.output_lang.index2word[tari.item()])
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            loss += self.criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
            
            _, decoder_pred_tensor = decoder_output.data.topk(1)            
            output_sentence.append(self.output_lang.index2word[decoder_pred_tensor.item()])
        
        output_sentence = ' '.join(output_sentence)
        target_sentence = ' '.join(target_sentence)
        
        train_bleu_score = bleu_score([target_sentence], [output_sentence])        
        return {'loss':loss, 'train_bleu_1': train_bleu_score}

    def predict_step(self, batch, batch_idx):
        input_tensor, input_sentence = batch[0].view(-1, 1), batch[1]
        input_length = input_tensor.size(0)
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                          encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[lang.SOS_token]], device=DEVICE)
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == lang.EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.output_lang.index2word[topi.item()])
            decoder_input = topi
           
        return input_sentence, ' '.join(decoded_words)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
    