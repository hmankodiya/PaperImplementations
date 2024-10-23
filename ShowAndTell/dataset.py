import os
import string
import re

import json
import pandas as pd
from nltk.tokenize import word_tokenize

import numpy as np
from PIL import Image

import torch
import torch.utils
import torch.utils.data


MAX_LENGTH = 40
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


def preprocess_text(text):
    # Remove punctuation and digits, and convert to lowercase
    cleaned_string = re.sub(f"[{string.punctuation}0-9]", "", text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string


def load_json(filepath):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset


class Vocab:
    def __init__(self, documents, max_length=MAX_LENGTH):
        self.documents = documents
        self.max_length = max_length
        self.special_tokens = ["[SOS]", "[EOS]", "[PAD]", "[UNKN]"]
        self.tokens = self.special_tokens + sorted(
            list(
                set(
                    [
                        token
                        for document in self.documents
                        for token in word_tokenize(preprocess_text(document))
                    ]
                )
            )
        )
        self.size = len(self.tokens)

        self.words2index = {token: i for i, token in enumerate(self.tokens)}
        self.index2words = {i: token for i, token in enumerate(self.tokens)}

    def encode_document(self, document):
        return (
            [self.words2index["[SOS]"]]
            + [
                self.words2index.get(token, self.words2index["[UNKN]"])
                for token in word_tokenize(preprocess_text(document))
            ]
            + [self.words2index["[EOS]"]]
        )

    def decode_indexes(self, indexes):
        return " ".join([self.index2words.get(index) for index in indexes])


def random_sample(documents):
    return np.random.choice(documents, size=1)[0]


def choose_index(documents, index=0):
    return documents[index]


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, dataset_path, sampling_fn=random_sample, **kwargs):
        self.vocab = vocab
        self.dataset_path = dataset_path
        self.dataset = load_json(dataset_path)
        self.sampling_fn = sampling_fn
        self.kwargs = kwargs  # used for sample_document fn

    def get_vocab(self):
        return self.vocab

    def normalize_image(
        self,
        image_array,
        mean=np.array([[[0.485]], [[0.456]], [[0.406]]]),
        std=np.array([[[0.229]], [[0.224]], [[0.225]]]),
    ):
        image_array = (image_array - mean) / std
        return image_array

    def load_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        return self.normalize_image(
            np.transpose(np.array(image) / 255.0, axes=[2, 0, 1])
        )

    def load_tokens(self, document):
        return np.array(self.vocab.encode_document(document))

    def __len__(self):
        return len(self.dataset)

    def sample_document(self, documents):
        return self.sampling_fn(documents, **self.kwargs)

    def __getitem__(self, index):
        instance = self.dataset[index]
        image_id, image_path = instance["image_id"], instance["file_path"]
        image = self.load_image(image_path)
        sampled_document = self.sample_document(instance["captions"])
        tokens = self.load_tokens(sampled_document)

        return image, tokens, (image_path, image_id)
