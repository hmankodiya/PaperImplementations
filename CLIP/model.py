import os
import logging
from typing import Union, List, Optional

import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPConfig

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_pretrained_clip_tokenizer(
    tokenizer_path="openai/clip-vit-base-patch32", **kwargs
):
    """
    Loads a pre-trained CLIP tokenizer.
    """
    tokenizer_config = kwargs.pop("config", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )

    try:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        logger.debug("CLIP tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load CLIP tokenizer from {tokenizer_path}: {e}")
        raise


def load_clip_model(
    model_path="openai/clip-vit-base-patch32", pretrained=True, **kwargs
):
    """
    Loads a CLIP model either from pretrained weights or random initialization.

    Args:
        model_path (str): Pretrained model name or path.
        pretrained (bool): If False, initializes model with random weights.
        kwargs: Additional model configuration arguments.

    Returns:
        CLIPModel
    """
    config_args = kwargs.get("config", "default")

    if not config_args or config_args == "default":
        model_config = CLIPConfig()
    else:
        model_config = CLIPConfig(**config_args)

    try:
        if pretrained:
            model = CLIPModel.from_pretrained(model_path, config=model_config)
            logger.debug("Pretrained CLIP model loaded successfully.")
        else:
            model = CLIPModel(config=model_config)
            logger.debug("Randomly initialized CLIP model created successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load CLIP model from {model_path}: {e}")
        raise


# --- TOKENIZER_DICT and MODEL_DICT ---

TOKENIZER_DICT = {
    "clip-vit-base-patch32": (
        load_pretrained_clip_tokenizer,
        {"tokenizer_path": "openai/clip-vit-base-patch32", "config": {}},
    ),
}

MODEL_DICT = {
    "clip-vit-base-patch32": (
        load_clip_model,
        {
            "model_path": "openai/clip-vit-base-patch32",
            "pretrained": True,
            "config": {},
        },
    ),
}


def load_tokenizer(tokenizer_name, tokenizer_path=None, tokenizer_config=None):
    """
    Dynamically fetch and initialize a tokenizer based on the tokenizer string.
    """
    if tokenizer_name in TOKENIZER_DICT:
        func, kwargs = TOKENIZER_DICT[tokenizer_name]

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


def load_model(model_name, model_path=None, model_config=None, pretrained=True):
    """
    Dynamically fetch and initialize a model based on the model string.

    Args:
        model_name (str): The key corresponding to the desired model in MODEL_DICT.
        model_path (str, optional): Custom model path.
        model_config (dict, optional): Custom model configuration.
        pretrained (bool, optional): Whether to load pretrained weights or random initialization.

    Returns:
        Model object initialized with the specified parameters.
    """
    if model_name in MODEL_DICT:
        func, kwargs = MODEL_DICT[model_name]

        if model_path is not None:
            kwargs["model_path"] = model_path

        if model_config is not None:
            kwargs["config"] = model_config

        kwargs["pretrained"] = pretrained

        logger.info(f"Initializing model '{model_name}' with arguments: {kwargs}")
        return func(**kwargs)
    else:
        logger.error(f"Model '{model_name}' is not registered in MODEL_DICT.")
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_DICT.")
