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
    """
    Moves a tensor to the specified device.

    Args:
        tensor (torch.Tensor): Input tensor.
        device (str): Target device (e.g., 'cuda:0' or 'cpu').

    Returns:
        torch.Tensor: Tensor moved to the specified device.
    """
    return tensor.to(device=device)


def load_pretrained_gpt2_tokenizer(tokenizer_path="openai-community/gpt2", **kwargs):
    """
    Loads a pre-trained GPT-2 tokenizer and adds special padding tokens.

    Args:
        tokenizer_path (str): Path to the pre-trained tokenizer.
        **kwargs: Additional configuration arguments for the tokenizer.

    Returns:
        GPT2Tokenizer: Loaded tokenizer with added special tokens.

    Raises:
        Exception: If the tokenizer fails to load.
    """
    tokenizer_config = kwargs.pop("config", {})
    if not isinstance(tokenizer_config, dict):
        logger.error(f"Expected config of type dict, got: {type(tokenizer_config)}")

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
    Loads a pre-trained Llama2 tokenizer and sets the padding token to the EOS token.

    Args:
        tokenizer_path (str): Path to the pre-trained tokenizer.
        **kwargs: Additional configuration arguments for the tokenizer.

    Returns:
        LlamaTokenizer: Loaded tokenizer with adjusted padding token.

    Raises:
        Exception: If the tokenizer fails to load.
    """
    tokenizer_config = kwargs.pop("config", {})
    if not isinstance(tokenizer_config, dict):
        logger.error(f"Expected config of type dict, got: {type(tokenizer_config)}")

    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Padding token set to EOS token.")
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
    Dynamically fetches and initializes a tokenizer based on the tokenizer name.

    Args:
        tokenizer_name (str): Name of the tokenizer in TOKENIZER_DICT.
        tokenizer_path (str, optional): Override default tokenizer path.
        tokenizer_config (dict, optional): Override default configuration.

    Returns:
        Tokenizer: Initialized tokenizer object.

    Raises:
        ValueError: If the tokenizer is not registered in TOKENIZER_DICT.
    """
    if tokenizer_name in TOKENIZER_DICT:
        func, kwargs = TOKENIZER_DICT[tokenizer_name]

        # Override arguments if provided
        if tokenizer_path is not None:
            kwargs["tokenizer_path"] = tokenizer_path
        if tokenizer_config is not None:
            kwargs["config"] = tokenizer_config

        logger.info(f"Initializing tokenizer '{tokenizer_name}' with args: {kwargs}")
        return func(**kwargs)
    else:
        logger.error(f"Tokenizer '{tokenizer_name}' is not registered.")
        raise ValueError(f"Tokenizer '{tokenizer_name}' is not registered.")


class Dinov2Encoder(nn.Module):
    """
    A class representing the DinoV2 image encoder.

    Args:
        config (Dinov2Config): Configuration for the DinoV2 model.
        freeze (bool): Whether to freeze the model parameters.
    """

    def __init__(self, config: Dinov2Config, freeze=True, **kwargs):
        super(Dinov2Encoder, self).__init__()
        self.config = config
        self.freeze = freeze
        self.encoder = Dinov2Model(self.config)
        self.freeze_network(self.freeze)

    def load_weights(self, dinov2_weights_path):
        """
        Loads weights into the encoder from a given path.

        Args:
            dinov2_weights_path (str): Path to the weights file.
        """
        dinov2_state_dict = torch.load(dinov2_weights_path, weights_only=True)
        self.encoder.load_state_dict(dinov2_state_dict)

    def forward(self, pixel_values):
        """
        Forward pass for the DinoV2 encoder.

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            tuple: Class tokens and the last hidden state.
        """
        last_hidden_state = self.encoder(pixel_values).last_hidden_state
        cls_tokens = last_hidden_state[:, 0, :]
        return cls_tokens.contiguous(), last_hidden_state

    def freeze_network(self, freeze=FREEZE):
        """
        Freezes or unfreezes the model parameters.

        Args:
            freeze (bool): Whether to freeze the parameters.
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    @classmethod
    def from_pretrained(cls, model_path, config, freeze, **kwargs):
        """
        Loads a pre-trained DinoV2Encoder.

        Args:
            model_path (str): Path to the pre-trained weights.
            config (Dinov2Config): Model configuration.
            freeze (bool): Whether to freeze the parameters.

        Returns:
            Dinov2Encoder: Loaded model instance.
        """
        model = cls(config, freeze, **kwargs)
        model.encoder.load_state_dict(torch.load(model_path, weights_only=True))
        return model
