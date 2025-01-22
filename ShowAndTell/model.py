import os
import logging

import evaluate
import torch
from torchmetrics.functional.text import bleu_score


# IMAGE-ENCODER PARAMS
HIDDEN_SIZE = 768
IMAGE_SIZE = 518
FREEZE = True

METRICS_DICT = dict(bleu=evaluate.load("bleu"))

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def to_device(tensor, device=DEVICE):
    return tensor.to(device=device)


def calculate_bleu(pred, target, ngram=4):
    return bleu_score(pred, target, n_gram=ngram)


import pickle


def compute_metrics(data, tokenizer, metrics):
    generated_tokens, labels = data
    results = {}
    # with open(
    #     "/home/harsh/Desktop/Projects/PaperImplementations/ShowAndTell/showandtell_gpt2/TrainingLogs/gen_tokens.pkl",
    #     "wb",
    # ) as F:
    #     pickle.dump(generated_tokens, F)

    generated_tokens[generated_tokens == -100] = tokenizer.pad_token_id
    batch_decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    labels[labels == -100] = tokenizer.pad_token_id
    labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

    results = metrics["bleu"].compute(predictions=batch_decoded, references=labels_decoded)
    return results
