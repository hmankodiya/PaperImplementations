import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GenerationConfig, GPT2Config

from model import to_device, calculate_bleu
from dataset import MAX_LENGTH


# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


class ShowAndTellGPT2(nn.Module):
    def __init__(
        self, tokenizer, image_encoder, text_encoder, generation_config=dict(), **kwargs
    ):
        super(ShowAndTellGPT2, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        assert (
            image_encoder.config.hidden_size == text_encoder.config.n_embd
        ), f"Found different hidden_size for text encoder ({text_encoder.hidden_size}) and image encoder ({image_encoder.config.hidden_size})"

        self.image_encoder = image_encoder
        self.hidden_size = image_encoder.config.hidden_size
        self.text_encoder = text_encoder
        self.config = self.text_encoder.config
        generation_config["bos_token_id"] = self.tokenizer.bos_token_id
        generation_config["eos_token_id"] = self.tokenizer.eos_token_id
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        self.generation_config = GenerationConfig(**generation_config)
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, model_path, tokenizer, image_encoder, text_encoder):
        model = cls(tokenizer, image_encoder, text_encoder)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def forward(
        self, pixel_values, input_ids=None, labels=None, return_dict=False, **kwargs
    ):
        _, encoder_hidden_states = self.image_encoder(pixel_values)
        _ = kwargs.pop("num_items_in_batch", None)
        outs = self.text_encoder(
            input_ids=labels,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outs[:2]

    def generate(self, pixel_values, **kwargs):
        if "input_ids" in kwargs:
            del kwargs["input_ids"]

        if "labels" in kwargs:
            del kwargs["labels"]

        if "attention_mask" in kwargs:
            del kwargs["attention_mask"]

        try:
            _, encoder_hidden_states = self.image_encoder(pixel_values)
            logger.debug("Starting text generation.")
            generated_tokens = self.text_encoder.generate(
                encoder_hidden_states=encoder_hidden_states,
                tokenizer=self.tokenizer,
                generation_config=self.generation_config,  # Unpack generation_config if provided
                **kwargs,
            )

            if any(x is None for x in generated_tokens):
                logger.error(f" Nonetype token found at {generated_tokens}")
                print(f" Nonetype token found at {generated_tokens}")
                raise ValueError(f" Nonetype token found at {generated_tokens}")

            return generated_tokens

            # # Decode the generated tokens
            # decoded_text = self.tokenizer.batch_decode(
            #     generated_tokens, skip_special_tokens=True
            # )
            # logger.info("Text generation completed successfully.")

            # return decoded_text

        except Exception as e:
            logger.error(f"An error occurred during text generation: {e}")
            raise RuntimeError(f"An error occurred during text generation: {e}")


def load_pretrained_gpt2_model(
    model_path="openai-community/gpt2",
    ignore_mismatched_sizes=True,
    use_safetensors=True,
    **kwargs,
):
    """
    Loads a pre-trained GPT-2 model with an optional configuration.
    """
    model_config = GPT2Config(**kwargs)

    if not isinstance(model_config, GPT2Config):
        logger.error(
            f"Found model_config of type: {type(model_config)}; expected config of type: Dict"
        )
    try:
        if model_path:
            model = GPT2LMHeadModel.from_pretrained(
                model_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                use_safetensors=use_safetensors,
                config=model_config,
            )
            logger.debug("Model loaded successfully.")
            return model

        return GPT2LMHeadModel(config=model_config)

    except Exception as e:
        logger.error(f"Failed to load GPT-2 model from {model_path}: {e}")
        raise


def load_showandtell_gpt2(
    tokenizer, image_encoder, text_encoder, pretrained_model_path=None
):
    try:
        if pretrained_model_path:
            logger.info(
                f"Loading pretrained ShowAndTellgpt2 model from: {pretrained_model_path}"
            )
            return ShowAndTellGPT2.from_pretrained(
                pretrained_model_path, tokenizer, image_encoder, text_encoder
            )

        logger.info("Initializing new ShowAndTellGPT2 model with provided components.")
        return ShowAndTellGPT2(tokenizer, image_encoder, text_encoder)
    except Exception as e:
        logger.error(f"Failed to load ShowAndTellGPT2 model: {e}")
        raise
