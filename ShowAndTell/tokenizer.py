import os
import logging

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, LlamaTokenizer


# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


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


# MODEL_DICT = {
#     "dinov2": load_dinov2_image_encoder,
#     "lstm": load_lstm_text_encoder,
#     "showandtell": load_show_and_tell,
# }


# def fetch_model_func(model_string):
#     if model_string in MODEL_DICT:
#         return MODEL_DICT[model_string]

#     logger.error(f"ModelLSTM '{model_string}' is not registered in MODEL_DICT.")
#     raise ValueError(f"ModelLSTM '{model_string}' is not registered in MODEL_DICT.")
