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
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages

SPLIT_KEYS = ["train", "val", "test"]


def preprocess_text(text):
    # Remove punctuation and digits, and convert to lowercase
    cleaned_string = re.sub(f"[{string.punctuation}0-9]", "", text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string


def load_json(filepath):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset


def _handle_seed(seed_val=None):
    if isinstance(seed_val, int):
        return seed_val

    return random.randint(0, 2**32 - 1)


def write_yaml(filepath, dictionary):
    """
    Writes a Python dictionary to a YAML file.

    Args:
        filepath (str): The path of the YAML file where the dictionary will be written.
        dictionary (dict): The Python dictionary to be written into the YAML file.

    Returns:
        str: Success message if the operation is successful.
        str: Error message if an exception occurs.
    """
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
    """
    Reads and parses a YAML file into a Python dictionary.

    Args:
        filepath (str): The path of the YAML file to read.

    Returns:
        dict: Parsed contents of the YAML file as a Python dictionary.
    """
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
    """
    Creates a directory or directories.

    If the directory/directories exist, it replaces them by deleting and recreating them.

    Args:
        paths (str or list): A string representing a single directory path or
                             a list of strings for multiple directories.

    Returns:
        None
    """
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


def get_split_config(config):
    """
    Extracts dataset split configuration from a given config dictionary.

    Args:
        config (dict): A dictionary containing dataset configuration and split keys.

    Returns:
        Tuple[str, Tuple[Any, Any, Any]]:
            - A description of the dataset (str).
            - A tuple containing configurations for train, val, and test splits.

    Raises:
        ValueError: If `dataset_config` is missing or empty, or if no split keys are found.
        KeyError: If split keys are not valid.
    """
    dataset_config = config.get("dataset_config", {})

    # Validate dataset_config
    if not dataset_config:
        raise ValueError(
            "The `dataset_config` key is either empty or not present in the config."
        )

    # Pop description with a default message if not provided
    description = dataset_config.pop("desc", "Dataset description not provided.")

    # Get the split keys and validate
    split_keys = [key for key in dataset_config.keys() if key in SPLIT_KEYS]
    if not split_keys:
        raise ValueError(f"At least one of the keys {SPLIT_KEYS} must be present.")
    if not set(split_keys).issubset(SPLIT_KEYS):
        raise KeyError(f"Dataset split keys must be a subset of {SPLIT_KEYS}.")

    # Extract configurations for splits
    train_split_config = dataset_config.get("train")
    val_split_config = dataset_config.get("val")
    test_split_config = dataset_config.get("test")

    return description, (train_split_config, val_split_config, test_split_config)


def get_tokenizer_config(config):
    """
    Extracts and validates the tokenizer configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'tokenizer_config'.

    Returns:
        tuple: A tuple containing the tokenizer name, tokenizer path, and any additional configuration.

    Raises:
        KeyError: If 'tokenizer_config' is missing or required keys are not found.
        ValueError: If the tokenizer configuration contains invalid values.
    """
    try:
        # Extract tokenizer_config
        tokenizer_config = config.get("tokenizer_config", {})

        # Validate required fields
        if (
            "tokenizer_name" not in tokenizer_config
            or "tokenizer_path" not in tokenizer_config
        ):
            raise KeyError(
                "Missing required keys in 'tokenizer_config': 'tokenizer_name' and/or 'tokenizer_path'"
            )

        # Log the configuration
        logger.info(f"Loaded tokenizer configuration: {tokenizer_config}")

        tokenizer_name = tokenizer_config.pop("tokenizer_name")
        tokenizer_path = tokenizer_config.pop("tokenizer_path")

        return (
            tokenizer_name,
            tokenizer_path,
            tokenizer_config,
        )

    except Exception as e:
        logger.error(f"An error occurred while processing 'tokenizer_config': {e}")
        raise


def get_dataset_config(dataset_config: dict):
    """
    Extracts and validates the dataset configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'dataset_config'.

    Returns:
        tuple: A tuple containing the dataset path and description.

    Raises:
        KeyError: If 'dataset_config' is missing or required keys are not found.
        ValueError: If the dataset configuration contains invalid values.
    """
    try:
        # Extract dataset_config

        # Validate required fields
        if "dataset_path" not in dataset_config:
            raise KeyError("Missing required key in 'dataset_config': 'dataset_path'")

        dataset_path = dataset_config.pop("dataset_path")
        sampling_config = dataset_config.pop("sampling_fn", dict())
        # Log the configuration
        logger.info(f"Loaded dataset configuration: {dataset_config}")

        return dataset_path, sampling_config, dataset_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'dataset_config': {e}")
        raise


def get_inference_dataset_config(config: dict):
    """
    Extracts and validates the dataset configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'dataset_config'.

    Returns:
        tuple: A tuple containing the dataset path and description.

    Raises:
        KeyError: If 'dataset_config' is missing or required keys are not found.
        ValueError: If the dataset configuration contains invalid values.
    """
    try:
        # Extract dataset_config

        # Validate required fields
        if "inference_dataset_config" not in config:
            raise KeyError(
                "Missing required key in 'inference_dataset_config': 'config'"
            )

        config = config.pop("inference_dataset_config")
        description = config.pop("desc", "Dataset description not provided.")

        if "image_paths" not in config:
            raise KeyError("Missing required key in 'image_paths': 'config'")

        image_paths = config.pop("image_paths")

        # Log the configuration
        logger.info(f"Loaded dataset configuration: {config}")

        return description, image_paths, config

    except Exception as e:
        logger.error(f"An error occurred while processing 'dataset_config': {e}")
        raise


def get_image_model_config(config):
    """
    Extracts and validates the model configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'model_config'.

    Returns:
        dict: The processed model configuration.

    Raises:
        KeyError: If 'model_config' is missing or required keys are not found.
        ValueError: If the model configuration contains invalid values.
    """
    try:
        # Validate required fields
        if "image_model" not in config:
            raise KeyError("Missing required keys in 'model_config': 'image_model'")

        config = config["image_model"]

        # Validate required fields
        if "model_name" not in config:
            raise KeyError("Missing required keys in 'model_config': 'model_name'")

        # Log the configuration
        logger.info(f"Loaded model configuration: {config}")

        model_name = config.pop("model_name")
        model_path = config.pop("model_path", None)
        freeze = config.pop("freeze", True)
        model_config = config.pop("config", {})
        

        return (model_name, model_path, freeze, model_config, config)

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_text_model_config(config):
    """
    Extracts and validates the model configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'model_config'.

    Returns:
        dict: The processed model configuration.

    Raises:
        KeyError: If 'model_config' is missing or required keys are not found.
        ValueError: If the model configuration contains invalid values.
    """
    try:

        # Validate required fields
        if "text_model" not in config:
            raise KeyError("Missing required keys in 'model_config': 'text_model'")

        config = config["text_model"]

        # Validate required fields
        if "model_name" not in config:
            raise KeyError("Missing required keys in 'model_config': 'model_name'")

        # Log the configuration
        logger.info(f"Loaded model configuration: {config}")

        model_name = config.pop("model_name")
        model_path = config.pop("model_path", None)
        config = config.pop("config", {})

        return (model_name, model_path, config)

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_showandtell_lstm_model_config(config):
    try:
        # Validate required fields
        if "showandtell_model" not in config:
            raise KeyError(
                "Missing required keys in 'model_config': 'showandtell_model'"
            )

        config = config["showandtell_model"]

        if "model_name" not in config:
            raise KeyError("Missing required keys in 'model_config': 'model_name'")

        # Log the configuration
        logger.info(f"Loaded model configuration: {config}")

        model_name = config.pop("model_name")
        model_path = config.pop("model_path", None)

        return (model_name, model_path, config)

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_showandtell_gpt2_model_config(config):
    try:
        # Validate required fields
        if "showandtell_model" not in config:
            raise KeyError(
                "Missing required keys in 'model_config': 'showandtell_model'"
            )

        config = config["showandtell_model"]

        if "model_name" not in config:
            raise KeyError("Missing required keys in 'model_config': 'model_name'")

        # Log the configuration
        logger.info(f"Loaded model configuration: {config}")

        model_name = config.pop("model_name")
        model_path = config.pop("model_path", None)

        return (model_name, model_path, config)

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_model_config(config):
    try:
        # Validate required fields
        if "model_config" not in config:
            raise KeyError("Missing required keys in 'config': 'model_config'")

        config = config["model_config"]

        return config

    except Exception as e:
        logger.error(f"An error occurred while processing 'config': {e}")
        raise


def get_showandtell_lstm_trainer_config(config):
    """
    Extracts and validates the trainer configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'trainer_config'.

    Returns:
        dict or None: The processed trainer configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract trainer_config
        trainer_config = config.get("trainer_config", {})

        # Check if trainer_config is empty
        if not trainer_config:
            logger.info("`trainer_config` is missing or empty. Returning None.")
            return None

        batch_size = trainer_config.pop("batch_size", 1)
        trainer_args = trainer_config.pop("trainer_args", {})
        logger_config = trainer_config.pop("logger_config", None)
        checkpoint_config = trainer_config.pop("model_checkpoint", {})
        save_config = trainer_config.pop("save_config", {})

        logger.info(
            f"Loaded trainer args: {trainer_args}, logger configuration: {logger_config}, checkpoint configuration: {checkpoint_config}, batch size {batch_size}, save configuration {save_config}"
        )
        return batch_size, trainer_args, logger_config, checkpoint_config, save_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'trainer_config': {e}")
        raise


def get_huggingface_trainer_config(config):
    """
    Extracts and validates the trainer configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'trainer_config'.

    Returns:
        dict or None: The processed trainer configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract trainer_config
        trainer_config = config.get("trainer_config", {})

        # Check if trainer_config is empty
        if not trainer_config:
            logger.info("'trainer_config' is missing or empty. Returning None.")
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
    """
    Extracts and validates the prediction configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'prediction_config'.

    Returns:
        dict or None: The processed prediction configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract prediction_config
        inference_args = config.get("inference_args", {})

        logger.info(f"Loaded inference configuration: {inference_args}")

        return inference_args

    except Exception as e:
        logger.error(f"An error occurred while processing 'inference_args': {e}")
        raise e
