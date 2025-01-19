import os
import string
import re
import logging
import types
import json
from typing import List, Union
import pandas as pd

import numpy as np
from PIL import Image

import torch
import torch.utils
import torch.utils.data
from transformers import DataCollatorWithPadding

from utils import preprocess_text, load_json

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages

# TOKENIZER PARAMS
MAX_LENGTH = 40

# IMAGE SIZE
SIZE = (518, 518)


def random_sample(documents, **kwargs):
    """
    Randomly samples one document from a list of documents.

    Args:
        documents (List[str]): List of documents to sample from.
        **kwargs: Additional keyword arguments for numpy's random.choice.

    Returns:
        str: A randomly sampled document.
    """
    return np.random.choice(documents, size=1, **kwargs)[0]


def choose_index(documents, index=0):
    """
    Selects a document from the list based on the given index.

    Args:
        documents (List[str]): List of documents to select from.
        index (int): Index of the document to select. Defaults to 0.

    Returns:
        str: The selected document.
    """
    return documents[index]


SAMPLING_DICT = {
    "random_sample": (random_sample, {}),
    "choose_index": (choose_index, {"index": 0}),
}


def fetch_sampling_fn(sampling_fn_name):
    """
    Fetches the sampling function and its default arguments from SAMPLING_DICT.

    Args:
        sampling_fn_name (str): Name of the sampling function.

    Returns:
        tuple: Sampling function and its default arguments.
    """
    func, kwargs = SAMPLING_DICT[sampling_fn_name]
    return func, kwargs


def tokenize_text(
    text_samples: Union[str, List[str]],
    tokenizer,
    max_length=None,
    truncation=True,
    use_encode=True,
    padding=True,
    return_tensors="np",
):
    """
    Tokenizes text using the specified tokenizer.

    Args:
        text_samples (Union[str, List[str]]): Input text or list of text samples.
        tokenizer: Tokenizer object to tokenize the text.
        max_length (int, optional): Maximum token length. Defaults to None.
        truncation (bool, optional): Whether to truncate text. Defaults to True.
        use_encode (bool, optional): Whether to use tokenizer's encode method. Defaults to True.
        padding (bool, optional): Whether to pad the text. Defaults to True.
        return_tensors (str, optional): Output format of tokenized text. Defaults to "np".

    Returns:
        List or np.ndarray: Tokenized text.
    """
    if use_encode:
        return tokenizer.encode(
            text_samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )

    return tokenizer(
        text_samples,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def normalize_image(
    pixel_values_array,
    mean=np.array([[[0.485]], [[0.456]], [[0.406]]]),
    std=np.array([[[0.229]], [[0.224]], [[0.225]]]),
):
    """
    Normalizes image pixel values based on mean and standard deviation.

    Args:
        pixel_values_array (np.ndarray): Pixel values of the image.
        mean (np.ndarray): Mean values for normalization.
        std (np.ndarray): Standard deviation values for normalization.

    Returns:
        np.ndarray: Normalized pixel values.
    """
    pixel_values_array = (pixel_values_array - mean) / std
    return pixel_values_array


def load_image(path, size, return_tensors=None, imagenet_normalize=True):
    """
    Loads and preprocesses an image from the given path.

    Args:
        path (str): Path to the image file.
        size (tuple): Desired size of the image.
        return_tensors (str, optional): Format to return the image. Defaults to None.

    Returns:
        np.ndarray or torch.Tensor: Preprocessed image.
    """
    pixel_values = Image.open(path).resize(size)
    normalized_pixel_values = np.array(pixel_values.convert("RGB")) / 255.0
    if imagenet_normalize:
        normalized_pixel_values = normalize_image(
            np.transpose(normalized_pixel_values, axes=[2, 0, 1])
        )

    if return_tensors == "pt":
        return torch.from_numpy(normalized_pixel_values)

    elif return_tensors == "list":
        return np.tolist(normalized_pixel_values)

    return normalized_pixel_values


def add_bos_eos(bos_token, eos_token, document):
    """
    Adds beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens to a document.

    Args:
        bos_token (str): Beginning-of-sequence token.
        eos_token (str): End-of-sequence token.
        document (str): Text document.

    Returns:
        str: Document with BOS and EOS tokens.
    """
    return bos_token + " " + document + " " + eos_token


class ImageCaptionDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for image captioning tasks.

    Args:
        tokenizer: Tokenizer object for text processing.
        dataset_path (str): Path to the JSON dataset file.
        return_tensors (str, optional): Format for returned tensors. Defaults to None.
        return_dict (bool, optional): Whether to return a dictionary. Defaults to False.
        sampling_fn (str or callable, optional): Sampling function for captions. Defaults to "random_sample".
        sampling_fn_args (dict, optional): Arguments for the sampling function. Defaults to None.
        **kwargs: Additional arguments for customization.
    """

    def __init__(
        self,
        tokenizer,
        dataset_path,
        return_tensors=None,
        return_dict=False,
        sampling_fn="random_sample",
        sampling_fn_args=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset = load_json(dataset_path)
        self.return_tensors = return_tensors
        self.return_dict = return_dict

        if isinstance(sampling_fn, str) and sampling_fn not in SAMPLING_DICT:
            logger.error(
                f"Sampling function '{sampling_fn}' is not registered in SAMPLING_DICT."
            )
            raise ValueError(
                f"Sampling function '{sampling_fn}' is not registered in SAMPLING_DICT."
            )

        if isinstance(sampling_fn, str):
            self.sampling_fn, default_sampling_args = fetch_sampling_fn(sampling_fn)
            self.sampling_fn_args = (
                sampling_fn_args if sampling_fn_args else default_sampling_args
            )
        elif isinstance(sampling_fn, types.FunctionType):
            self.sampling_fn_args = sampling_fn_args if sampling_fn_args else dict()
        else:
            raise ValueError(
                f"Sampling function '{sampling_fn}' should be either a callable type or str, found type {type(sampling_fn)}."
            )
        self.image_size = kwargs.pop("image_size", SIZE)
        self.kwargs = kwargs

    def get_vocab(self):
        """
        Retrieves the vocabulary size from the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        return self.tokenizer.vocab_size

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)

    def sample_document(self, documents):
        """
        Samples a document from the given list of documents.

        Args:
            documents (List[str]): List of documents to sample from.

        Returns:
            str: Sampled document.
        """
        return self.sampling_fn(documents, **self.sampling_fn_args)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict or tuple: Image pixel values and tokenized captions.
        """
        instance = self.dataset[index]
        image_id, image_path = instance["image_id"], instance["file_path"]

        pixel_values = load_image(
            image_path,
            size=self.image_size,
            return_tensors=self.return_tensors,
            imagenet_normalize=True,
        )

        sampled_document = self.sample_document(instance["captions"]).strip()
        sampled_document = add_bos_eos(
            self.tokenizer.special_tokens_map["bos_token"],
            self.tokenizer.special_tokens_map["eos_token"],
            sampled_document,
        )

        labels = tokenize_text(
            sampled_document,
            self.tokenizer,
            max_length=self.kwargs.pop("max_length", MAX_LENGTH),
            truncation=self.kwargs.pop("truncation", True),
            padding=self.kwargs.pop("padding", True),
            use_encode=True,
            return_tensors=self.return_tensors,
        )

        if self.return_dict:
            return dict(
                pixel_values=pixel_values,
                input_ids=labels,
            )

        return pixel_values, labels


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, return_tensors=None, return_dict=False, **kwargs):
        self.image_paths = image_paths
        self.return_tensors = return_tensors
        self.return_dict = return_dict
        self.image_size = kwargs.pop("image_size", SIZE)
        self.kwargs = kwargs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pixel_values = load_image(
            self.image_paths[index],
            size=self.image_size,
            return_tensors=self.return_tensors,
            imagenet_normalize=self.kwargs.get('imagenet_normalize', True)
        )

        if self.return_dict:
            return dict(pixel_values=pixel_values)

        return pixel_values


class ImageTextCollator:
    """
    A collator class for batching image and text data in PyTorch.

    Args:
        tokenizer: Tokenizer object for text processing.
        padding: Padding strategy for text data.
        return_tensors (str): Format for returned tensors.
    """

    def __init__(self, tokenizer, padding, return_tensors):
        self.text_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=padding, return_tensors=return_tensors
        )

    def __call__(self, batch):
        """
        Processes and batches a list of data samples.

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data with pixel values and text inputs.
        """
        pixel_values = []
        labels = []

        for batch_dict in batch:
            pixel_values.append(torch.from_numpy(batch_dict.pop("pixel_values")))
            labels.append(batch_dict)

        batch_dict = self.text_collator(labels)
        batch_dict.update({"pixel_values": torch.stack(pixel_values)})

        return batch_dict
