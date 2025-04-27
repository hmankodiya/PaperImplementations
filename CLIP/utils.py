import os
import shutil
import yaml
import logging
import random
import re
import string
import json

import torch

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SPLIT_KEYS = ["train", "val", "test"]

# ========== BASIC UTILS ==========


def preprocess_text(text):
    cleaned_string = re.sub(f"[{string.punctuation}0-9]", "", text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string


def load_json(filepath):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset


def write_yaml(filepath, dictionary):
    try:
        logger.debug(
            f"Attempting to write YAML file at '{filepath}' with data: {dictionary}"
        )
        with open(filepath, "w") as f:
            yaml.safe_dump(dictionary, f)
        logger.info(
            f"YAML file '{filepath}' successfully created and parameters saved."
        )
        return f"YAML file '{filepath}' successfully created and parameters saved."
    except Exception as e:
        logger.error(f"Failed to write YAML file '{filepath}'. Error: {e}")
        return str(e)


def read_yaml(filepath):
    try:
        logger.debug(f"Attempting to read YAML file from '{filepath}'")
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        logger.info(f"YAML file '{filepath}' loaded successfully.")
        logger.debug(f"Configuration loaded: {data}")
        return data
    except Exception as e:
        logger.error(f"Failed to read YAML file '{filepath}'. Error: {e}")
        raise e


def make_dir(paths):
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        try:
            if os.path.exists(path):
                logger.debug(f"Directory '{path}' already exists. Removing it.")
                shutil.rmtree(path)
                logger.info(f"Existing directory '{path}' removed.")
            os.makedirs(path)
            logger.info(f"Directory '{path}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create directory '{path}'. Error: {e}")
            raise e


def _handle_seed(seed_val=None):
    if isinstance(seed_val, int):
        return seed_val
    return random.randint(0, 2**32 - 1)


# ========== CONFIG PARSING UTILS ==========


def get_split_config(config):
    dataset_config = config.get("dataset_config", {})
    if not dataset_config:
        raise ValueError(
            "The `dataset_config` key is either empty or not present in the config."
        )

    description = dataset_config.pop("desc", "Dataset description not provided.")

    split_keys = [key for key in dataset_config.keys() if key in SPLIT_KEYS]
    if not split_keys:
        raise ValueError(f"At least one of the keys {SPLIT_KEYS} must be present.")
    if not set(split_keys).issubset(SPLIT_KEYS):
        raise KeyError(f"Dataset split keys must be a subset of {SPLIT_KEYS}.")

    train_split_config = dataset_config.get("train")
    val_split_config = dataset_config.get("val")
    test_split_config = dataset_config.get("test")

    return description, (train_split_config, val_split_config, test_split_config)


def get_tokenizer_config(config):
    try:
        tokenizer_config = config.get("tokenizer_config", {})

        if (
            "tokenizer_name" not in tokenizer_config
            or "tokenizer_path" not in tokenizer_config
        ):
            raise KeyError(
                "Missing required keys in 'tokenizer_config': 'tokenizer_name' and/or 'tokenizer_path'"
            )

        logger.info(f"Loaded tokenizer configuration: {tokenizer_config}")

        tokenizer_name = tokenizer_config["tokenizer_name"]
        tokenizer_path = tokenizer_config["tokenizer_path"]
        extra_config = {
            key: val
            for key, val in tokenizer_config.items()
            if key not in ["tokenizer_name", "tokenizer_path"]
        }

        return tokenizer_name, tokenizer_path, extra_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'tokenizer_config': {e}")
        raise


def get_model_config(config):
    try:
        model_config = config.get("model_config", {})

        if "clip_model" not in model_config:
            raise KeyError("Missing required 'clip_model' inside 'model_config'")

        clip_model_config = model_config["clip_model"]

        if (
            "model_name" not in clip_model_config
            or "model_path" not in clip_model_config
        ):
            raise KeyError(
                "Missing required keys 'model_name' and/or 'model_path' inside 'clip_model'"
            )

        logger.info(f"Loaded clip model configuration: {clip_model_config}")

        model_name = clip_model_config["model_name"]
        model_path = clip_model_config["model_path"]
        clip_model_extra_config = clip_model_config.get("config", {})
        
        pretrained = None
        if not model_path:
            pretrained = False
        
        return model_name, model_path, clip_model_extra_config, pretrained

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_dataset_config(dataset_config: dict):
    try:
        if "dataset_path" not in dataset_config:
            raise KeyError("Missing required key in 'dataset_config': 'dataset_path'")

        dataset_path = dataset_config.pop("dataset_path")
        sampling_config = dataset_config.pop("sampling_fn", dict())
        logger.info(f"Loaded dataset configuration: {dataset_config}")

        return dataset_path, sampling_config, dataset_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'dataset_config': {e}")
        raise


def get_inference_dataset_config(config: dict):
    try:
        if "inference_dataset_config" not in config:
            raise KeyError(
                "Missing required key in 'inference_dataset_config': 'config'"
            )

        config = config.pop("inference_dataset_config")
        description = config.pop("desc", "Dataset description not provided.")

        if "image_paths" not in config:
            raise KeyError(
                "Missing required key 'image_paths' inside 'inference_dataset_config'."
            )

        image_paths = config.pop("image_paths")
        logger.info(f"Loaded inference dataset configuration: {config}")

        return description, image_paths, config

    except Exception as e:
        logger.error(
            f"An error occurred while processing 'inference_dataset_config': {e}"
        )
        raise


# ========== TRAINER AND INFERENCE CONFIG ==========


def get_trainer_config(config):
    try:
        trainer_config = config.get("trainer_config", {})

        if not trainer_config:
            logger.info("`trainer_config` is missing or empty. Returning None.")
            return None

        seed = trainer_config.get("seed", "none")
        if isinstance(seed, str):
            trainer_config["seed"] = _handle_seed(seed_val=None)

        logger.info(f"Loaded trainer configuration: {trainer_config}")
        return trainer_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'trainer_config': {e}")
        raise


def get_inference_args(config):
    try:
        inference_args = config.get("inference_args", {})

        logger.info(f"Loaded inference configuration: {inference_args}")
        return inference_args

    except Exception as e:
        logger.error(f"An error occurred while processing 'inference_args': {e}")
        raise
