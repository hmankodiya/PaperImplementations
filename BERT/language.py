import string
import unicodedata

import os
import re
from collections import defaultdict

import numpy as np

import torch
import torch.utils
import torch.utils.data
from transformers import BertTokenizer


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    return s


class Corpus:
    def __init__(self, corpus_filename):
        self.word2count = defaultdict(int)
        self.n_words = 0

        self.corpus_filename = corpus_filename
        assert os.path.isfile(
            self.corpus_filename
        ), f"file {self.corpus_filename} not found"
        self.sentences = []
        with open(self.corpus_filename, "r") as f:
            self.sentences = f.read().split("\n")

        self.n_sentences = len(self.sentences)

        self.populate_corpus()

    def add_word(self, word):
        self.word2count[word] += 1
        self.n_words += 1

    def add_sentence(self, sentence):
        pattern = r"([" + re.escape(string.punctuation) + r"])"
        sentence = re.sub(pattern, r" \1 ", sentence)
        for word in sentence.split(" "):
            self.add_word(word)

    def populate_corpus(self):
        for sentence in self.sentences:
            self.add_sentence(sentence)

        self.words = set(list(self.word2count.keys()))


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        corpus_filename,
        prediction_percent=0.15,
        mask_percent=0.80,
        random_percent=0.10,
    ):
        self.tokenizer = tokenizer
        self.prediction_percent = prediction_percent
        self.mask_percent = mask_percent
        self.random_percent = random_percent

        self.corpus_filename = corpus_filename
        self.corpus = Corpus(self.corpus_filename)

    def __getitem__(self, index):
        sentence = self.corpus.sentences[index]
        input_ids = self.tokenizer(sentence, return_tensors="pt")["input_ids"]
        input_ids, maked_labels, original_labels = self.mask_tokens(input_ids)

        return input_ids, maked_labels, original_labels

    def fetch_insample_tokens(self, n_samples):
        random_tokens = torch.randint(
            low=0, high=len(self.tokenizer), size=(n_samples,)
        )
        return random_tokens

    def mask_tokens(self, input_ids):
        labels = input_ids.clone()

        # Masking logic: we randomly select 15% of tokens to mask
        probability_matrix = torch.full(labels.shape, self.prediction_percent)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        masked_indices = torch.bernoulli(probability_matrix).bool()

        original_labels = labels.clone()
        labels[~masked_indices] = -1

        # Replace 80% of masked tokens with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_percent)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # Replace 10% of masked tokens with random tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_percent)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = self.fetch_insample_tokens(n_samples=int(indices_random.sum()))
        input_ids[indices_random] = random_words

        # 10% of masked tokens are left unchanged
        return input_ids, labels, original_labels

    def __len__(self):
        return self.corpus.n_sentences
