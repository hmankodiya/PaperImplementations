import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import to_device, HIDDEN_SIZE, calculate_bleu
from dataset import MAX_LENGTH


# TEXT-ENCODER PARAMS
NUM_LAYERS = 1
BIDIRECTIONAL = False


# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


class LSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, **kwargs):
        super(LSTMTextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = kwargs.pop("num_layers", NUM_LAYERS)
        self.hidden_size = kwargs.pop("hidden_size", HIDDEN_SIZE)
        self.bidirectional = kwargs.pop("birectional", BIDIRECTIONAL)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def prepare_states(self, state):
        return state.repeat((2 - (not self.bidirectional)) * self.num_layers, 1, 1)

    def forward(self, x, h=None, c=None):
        output, (h, c) = self.lstm(x, (h, c))
        return output, (h, c)

    @classmethod
    def from_pretrained(cls, model_path, vocab_size, **kwargs):
        model = cls(vocab_size, **kwargs)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model


class ShowAndTellLSTM(nn.Module):
    def __init__(self, tokenizer, image_encoder, text_encoder, **kwargs):
        super(ShowAndTellLSTM, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id
        )
        self.kwargs = kwargs

        assert (
            image_encoder.config.hidden_size == text_encoder.hidden_size
        ), f"Found different hidden_size for text encoder ({text_encoder.hidden_size}) and image encoder ({image_encoder.config.hidden_size})"
        self.hidden_size = text_encoder.hidden_size

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.vocab_size
        )

    def calculate_loss(self, labels, logits):
        loss = self.criterion(
            logits.view(-1, logits.size(-1)), labels.contiguous().view(-1)
        )
        return loss

    def _shift_labels(self, labels):
        # assuming theres always <|startoftext|> at 0th index of labels
        shifted_labels = labels[..., :-1].clone()
        labels = labels[..., 1:]

        return shifted_labels, labels

    @classmethod
    def from_pretrained(cls, model_path, tokenizer, image_encoder, text_encoder):
        model = cls(tokenizer, image_encoder, text_encoder)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def forward(self, pixel_values, labels=None, return_dict=False):
        cls_tokens, _ = self.image_encoder(pixel_values)
        cls_tokens = self.text_encoder.prepare_states(cls_tokens)
        h, c = cls_tokens, cls_tokens
        loss = None
        if labels is not None:
            shifted_labels, labels = self._shift_labels(labels)
            text_embeddings = self.embeddings(shifted_labels)
            outputs, (h, c) = self.text_encoder(text_embeddings, h, c)
            logits = self.linear(outputs)
            loss = self.calculate_loss(labels, logits)
        else:
            logits = []
            start_token = to_device(torch.tensor([self.tokenizer.bos_token_id]))
            start_embedding = self.embeddings(start_token)

            current_input = start_embedding.unsqueeze(0)
            for t in range(self.kwargs.get("max_length", MAX_LENGTH)):
                output, (h, c) = self.text_encoder(current_input, h, c)
                logit_t = self.linear(output.squeeze(1))

                logits.append(logit_t.unsqueeze(1))
                predicted_word = torch.argmax(logit_t, dim=-1).detach()

                if predicted_word == self.tokenizer.eos_token_id:
                    break

                current_input = self.embeddings(predicted_word).unsqueeze(1)

            logits = torch.cat(logits, dim=1)

        torch.cuda.empty_cache()

        if not return_dict:
            output = (logits,)
            return output + (loss,) if loss else output

        return dict(logits=logits, loss=loss)


class ModelLSTM(pl.LightningModule):
    def __init__(self, tokenizer, showandtell_lstm_core, optimizer=torch.optim.Adam):
        super(ModelLSTM, self).__init__()
        self.tokenizer = tokenizer
        self.showandtell_lstm_core = showandtell_lstm_core
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.train_step_loss = []
        self.val_metric = {
            "loss": [],
            "bleu": [],
            # "image_id": [],
            # "image_path": [],
            # "target": [],
            # "predictions": [],
        }

    def configure_optimizers(self):
        return self.optimizer(self.showandtell_lstm_core.parameters())

    def _step(self, pixel_values, lables=None):
        outputs = self.showandtell_lstm_core(
            pixel_values=pixel_values,
            labels=lables,
            return_dict=False,
        )

        return outputs

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_metric["loss"]).mean()
        avg_bleu = torch.stack(self.val_metric["bleu"]).mean()
        self.val_metric["loss"].clear()
        self.val_metric["bleu"].clear()

        di = {"val_loss": avg_loss, "bleu_score": avg_bleu}
        self.log_dict(di, prog_bar=True)

    def validation_step(self, batch, batch_idx=None):
        pixel_values, labels = batch["pixel_values"], batch["lables"]
        label_sequence = list(
            map(
                "".join,
                list(
                    map(
                        self.tokenizer.batch_decode,
                        labels[:, 1:].detach().cpu().numpy().tolist(),
                    )
                ),
            )
        )
        logits, loss = self._step(pixel_values, labels)
        prediction = self.tokenizer.batch_decode(logits.argmax(-1))
        bleu = calculate_bleu(prediction, label_sequence)

        self.val_metric["bleu"].append(bleu)
        self.val_metric["loss"].append(loss)

        di = {"loss": loss}

        return di

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.train_step_loss).mean()
        self.train_step_loss.clear()
        di = {"train_loss": avg_loss}
        self.log_dict(di, prog_bar=True)

    def training_step(self, batch, batch_idx=None):
        pixel_values, labels = batch["pixel_values"], batch["lables"]
        logits, loss = self._step(pixel_values, labels)

        self.train_step_loss.append(loss)
        di = {"loss": loss}

        return di

    def predict_step(self, batch, batch_idx=None):
        pixel_values = batch["pixel_values"]
        (logits,) = self._step(pixel_values, lables=None)
        logits = logits.detach().cpu()
        out_tokens = logits.argmax(-1).detach().cpu().numpy().tolist()

        prediction_sequence = list(
            map(
                "".join,
                list(map(self.tokenizer.batch_decode, out_tokens)),
            )
        )[0]

        return prediction_sequence, (logits, out_tokens)


def load_lstm_encoder(vocab_size, pretrained_model_path=None, **kwargs):
    try:
        if pretrained_model_path:
            logger.info(
                f"Loading pretrained LSTM Text Encoder from: {pretrained_model_path}"
            )
            return LSTMTextEncoder.from_pretrained(
                pretrained_model_path, vocab_size, **kwargs
            )

        logger.info(f"Initializing new LSTM Text Encoder with vocab size: {vocab_size}")
        return LSTMTextEncoder(vocab_size, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load LSTM Text Encoder: {e}")
        raise


def load_showandtell_lstm(
    tokenizer, image_encoder, text_encoder, pretrained_model_path=None
):
    try:
        if pretrained_model_path:
            logger.info(
                f"Loading pretrained ShowAndTellLSTM model from: {pretrained_model_path}"
            )
            return ShowAndTellLSTM.from_pretrained(
                pretrained_model_path, tokenizer, image_encoder, text_encoder
            )

        logger.info("Initializing new ShowAndTellLSTM model with provided components.")
        return ShowAndTellLSTM(tokenizer, image_encoder, text_encoder)
    except Exception as e:
        logger.error(f"Failed to load ShowAndTellLSTM model: {e}")
        raise


def load_lightning_showandtell_lstm(
    tokenizer, showandtell_lstm_core, optimizer=torch.optim.Adam
):
    try:
        logger.info(
            f"Initializing Lightning ModelLSTM with ShowAndTellLSTM core and optimizer: {optimizer.__name__}"
        )
        return ModelLSTM(tokenizer, showandtell_lstm_core, optimizer)
    except Exception as e:
        logger.error(f"Failed to load Lightning ModelLSTM: {e}")
        raise
