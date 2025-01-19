import os
import logging

from PIL import Image
import numpy as np
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional.text import bleu_score
from transformers import Dinov2Model, Dinov2Config, GPT2Tokenizer, LlamaTokenizer
from dataset import MAX_LENGTH

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# IMAGE-ENCODER PARAMS
HIDDEN_SIZE = 768
IMAGE_SIZE = 518
FREEZE = True

# TEXT-ENCODER PARAMS
NUM_LAYERS = 1
BIDIRECTIONAL = False


def to_device(tensor, device=DEVICE):
    return tensor.to(device=device)


def load_pretrained_gpt2_tokenizer(tokenizer_path="openai-community/gpt2", **kwargs):
    """
    Loads a pre-trained GPT-2 tokenizer and adds a special padding token.
    """
    tokenizer_config = kwargs.pop("config", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        tokenizer.add_special_tokens(
            dict(pad_token="<|pad|>", bos_token="<|startoftext|>")
        )
        logger.debug("Special tokens added to the tokenizer.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load GPT-2 tokenizer from {tokenizer_path}: {e}")
        raise


def load_pretrained_llama2_tokenizer(
    tokenizer_path="meta-llama/Llama-2-7b-hf", **kwargs
):
    """
    Loads a pre-trained Llama2 tokenizer and adds a special padding token.
    """
    tokenizer_config = kwargs.pop("config", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )
    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Special tokens added to the tokenizer.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load Llama2 tokenizer from {tokenizer_path}: {e}")
        raise


TOKENIZER_DICT = {
    "gpt2": (
        load_pretrained_gpt2_tokenizer,
        {"tokenizer_path": "openai-community/gpt2", "config": {}},
    ),
    "gpt2-xl": (
        load_pretrained_gpt2_tokenizer,
        {"tokenizer_path": "openai-community/gpt2-xl", "config": {}},
    ),
    "llama2": (
        load_pretrained_llama2_tokenizer,
        {"tokenizer_path": "meta-llama/Llama-2-7b-hf", "config": {}},
    ),
}


def load_tokenizer(tokenizer_name, tokenizer_path=None, tokenizer_config=None):
    """
    Dynamically fetch and initialize a tokenizer based on the tokenizer string.

    Args:
        tokenizer_name (str): The key corresponding to the desired tokenizer in TOKENIZER_DICT.
        tokenizer_path (str, optional): Custom tokenizer path to override the default path in TOKENIZER_DICT.
        tokenizer_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT
    Returns:
        Tokenizer object initialized with the specified parameters.

    Raises:
        ValueError: If the tokenizer string is not registered in TOKENIZER_DICT.
    """
    if tokenizer_name in TOKENIZER_DICT:
        func, kwargs = TOKENIZER_DICT[tokenizer_name]

        # Dynamically update kwargs based on provided arguments
        if tokenizer_path is not None:
            kwargs["tokenizer_path"] = tokenizer_path

        if tokenizer_config is not None:
            kwargs["config"] = tokenizer_config

        logger.info(
            f"Initializing tokenizer '{tokenizer_name}' with arguments: {kwargs}"
        )
        return func(**kwargs)
    else:
        logger.error(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )


class Dinov2Encoder(nn.Module):
    def __init__(
        self,
        config: Dinov2Config,
        freeze=True,
        **kwargs,
    ):
        super(Dinov2Encoder, self).__init__()
        self.config = config
        self.freeze = freeze
        self.encoder = Dinov2Model(self.config)
        self.freeze_network(self.freeze)

    def load_weights(self, dinov2_weights_path):
        dinov2_state_dict = torch.load(dinov2_weights_path, weights_only=True)
        self.encoder.load_state_dict(dinov2_state_dict)

    def forward(self, pixel_values):
        last_hidden_state = self.encoder(pixel_values).last_hidden_state
        cls_tokens = last_hidden_state[:, 0, :]
        return cls_tokens.contiguous(), last_hidden_state

    def freeze_network(self, freeze=FREEZE):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    @classmethod
    def from_pretrained(cls, model_path, config, freeze, **kwargs):
        model = cls(config, freeze, **kwargs)
        model.encoder.load_state_dict(torch.load(model_path, weights_only=True))
        return model


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


class ShowAndTell(nn.Module):
    def __init__(self, tokenizer, image_encoder, text_encoder, **kwargs):
        super(ShowAndTell, self).__init__()
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


class Model(pl.LightningModule):
    def __init__(self, tokenizer, showtell_core, optimizer=torch.optim.Adam):
        super(Model, self).__init__()
        self.tokenizer = tokenizer
        self.showandtell_core = showtell_core
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
        return self.optimizer(self.showandtell_core.parameters())

    def _step(self, pixel_values, input_ids=None):
        outputs = self.showandtell_core(
            pixel_values=pixel_values,
            labels=input_ids,
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
        pixel_values, labels = batch["pixel_values"], batch["input_ids"]
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
        pixel_values, labels = batch["pixel_values"], batch["input_ids"]
        logits, loss = self._step(pixel_values, labels)

        self.train_step_loss.append(loss)
        di = {"loss": loss}

        return di

    def predict_step(self, batch, batch_idx=None):
        pixel_values = batch["pixel_values"]
        (logits,) = self._step(pixel_values, input_ids=None)
        logits = logits.detach().cpu()
        out_tokens = logits.argmax(-1).detach().cpu().numpy().tolist()

        prediction_sequence = list(
            map(
                "".join,
                list(map(self.tokenizer.batch_decode, out_tokens)),
            )
        )[0]

        return prediction_sequence, (logits, out_tokens)


def calculate_bleu(pred, target, ngram=4):
    return bleu_score(pred, target, n_gram=ngram)


def load_lstm_text_encoder(vocab_size, pretrained_model_path=None, **kwargs):
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


def load_dinov2_image_encoder(
    config, freeze=FREEZE, pretrained_model_path=None, **kwargs
):
    try:
        if pretrained_model_path:
            logger.info(
                f"Loading pretrained DinoV2 Image Encoder from: {pretrained_model_path}"
            )
            return Dinov2Encoder.from_pretrained(
                pretrained_model_path, config, freeze, **kwargs
            )

        logger.info(
            f"Initializing new DinoV2 Image Encoder with config: {config} and freeze: {freeze}"
        )
        return Dinov2Encoder(config, freeze, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load DinoV2 Image Encoder: {e}")
        raise


def load_show_and_tell(
    tokenizer, image_encoder, text_encoder, pretrained_model_path=None
):
    try:
        if pretrained_model_path:
            logger.info(
                f"Loading pretrained ShowAndTell model from: {pretrained_model_path}"
            )
            return ShowAndTell.from_pretrained(
                pretrained_model_path, tokenizer, image_encoder, text_encoder
            )

        logger.info("Initializing new ShowAndTell model with provided components.")
        return ShowAndTell(tokenizer, image_encoder, text_encoder)
    except Exception as e:
        logger.error(f"Failed to load ShowAndTell model: {e}")
        raise


def load_lightning_model(tokenizer, showandtell_core, optimizer=torch.optim.Adam):
    try:
        logger.info(
            f"Initializing Lightning Model with ShowAndTell core and optimizer: {optimizer.__name__}"
        )
        return Model(tokenizer, showandtell_core, optimizer)
    except Exception as e:
        logger.error(f"Failed to load Lightning Model: {e}")
        raise



# MODEL_DICT = {
#     "dinov2": load_dinov2_image_encoder,
#     "lstm": load_lstm_text_encoder,
#     "showandtell": load_show_and_tell,
# }


# def fetch_model_func(model_string):
#     if model_string in MODEL_DICT:
#         return MODEL_DICT[model_string]

#     logger.error(f"Model '{model_string}' is not registered in MODEL_DICT.")
#     raise ValueError(f"Model '{model_string}' is not registered in MODEL_DICT.")
