import os

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional.text import bleu_score
from transformers import Dinov2Model, Dinov2Config

from dataset import ImageCaptionDataset, Vocab

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def to_device(tensor, device=DEVICE):
    return tensor.to(device=device)


class Dinov2Encoder(nn.Module):
    def __init__(
        self,
        config: Dinov2Config,
        dinov2_weights_path="./weights/dinov2-base-weights.pth",
        freeze=True,
    ):
        super(Dinov2Encoder, self).__init__()
        self.config = config
        self.config.image_size = 518
        self.dinov2_weights_path = dinov2_weights_path
        self.encoder = Dinov2Model(self.config)
        self.load_weights(self.dinov2_weights_path)
        self.freeze_network(freeze)

    def load_weights(self, dinov2_weights_path):
        dinov2_state_dict = torch.load(dinov2_weights_path, weights_only=True)
        self.encoder.load_state_dict(dinov2_state_dict)

    def forward(self, x):
        last_hidden_state = self.encoder(x).last_hidden_state
        cls_tokens = last_hidden_state[:, 0, :]
        return cls_tokens, last_hidden_state

    def freeze_network(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=1):
        super(TextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )

    def prepare_states(self, state):
        return state.reshape(self.num_layers, -1, self.hidden_size)

    def forward(self, x, h=None, c=None):
        output, (h, c) = self.lstm(x, (h, c))
        return output, (h, c)


class ShowAndTell(nn.Module):
    def __init__(self, vocab, image_encoder, text_encoder):
        super(ShowAndTell, self).__init__()
        self.vocab = vocab
        self.vocab_size = self.vocab.size
        self.max_length = self.vocab.max_length
        self.hidden_size = text_encoder.hidden_size

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.vocab_size
        )

    def forward(self, image, tokens=None, teacher_forcing=True):
        cls_tokens, _ = self.image_encoder(image)
        cls_tokens = self.text_encoder.prepare_states(cls_tokens)
        h0, c0 = cls_tokens, cls_tokens

        if tokens is None:
            teacher_forcing = False

        if teacher_forcing:
            text_embeddings = self.embeddings(tokens)
            outputs, (h, c) = self.text_encoder(text_embeddings, h0, c0)
            logits = self.linear(outputs)

        else:
            logits = []
            start_token = to_device(torch.tensor([self.vocab.words2index["<start>"]]))
            start_embedding = self.embeddings(start_token)

            current_input = start_embedding.unsqueeze(0)
            h, c = h0, c0
            for t in range(self.max_length):
                output, (h, c) = self.text_encoder(current_input, h, c)
                logit_t = self.linear(output.squeeze(1))

                logits.append(logit_t.unsqueeze(1))
                predicted_word = torch.argmax(logit_t, dim=-1).detach()

                if predicted_word == self.vocab.words2index["<end>"]:
                    break

                current_input = self.embeddings(predicted_word).unsqueeze(1)

            logits = torch.cat(logits, dim=1)

        torch.cuda.empty_cache()

        return logits


class Model(pl.LightningModule):
    def __init__(self, vocab, showtell_core, optimizer=torch.optim.Adam):
        super(Model, self).__init__()
        self.vocab = vocab
        self.showandtell_core = showtell_core
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.step_loss = []

    def configure_optimizers(self):
        return self.optimizer(self.showandtell_core.parameters())

    def _step(self, x, y=None, teacher_forcing=None):
        if teacher_forcing is None:
            teacher_forcing = np.random.choice([True, False], size=1)
        logits = self.showandtell_core(
            image=x, tokens=y, teacher_forcing=teacher_forcing
        )

        return logits

    def calculate_loss(self, y, y_hat):
        true_len, pred_len = y.size(1), y_hat.size(1)
        min_len = min(true_len, pred_len)
        y_truncated, y_hat_truncated = y[:, :min_len], y_hat[:, :min_len, :]
        loss = self.criterion(
            y_hat_truncated.view(-1, self.vocab.size), y_truncated.view(-1)
        )

        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.step_loss).mean()
        di = {'train_loss': avg_loss}
        self.log_dict(di, prog_bar=True)

    def training_step(self, batch, batch_idx=None):
        x, y, (_, image_id) = batch
        logits = self._step(x, y)
        loss = self.calculate_loss(y, logits)
        self.step_loss.append(loss)

        di = {"loss": loss}

        return di

    def predict_step(self, batch, batch_idx=None):
        x = batch[0]
        logits = self._step(x, y=None, teacher_forcing=False).detach().cpu()
        out_tokens = logits.argmax(-1).numpy()

        return list(map(self.vocab.decode_indexes, out_tokens))

def calculate_bleu(pred, target, ngram=2):
    return bleu_score(pred, target, n_gram=ngram)
