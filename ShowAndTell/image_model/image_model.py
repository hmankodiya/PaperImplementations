import os
import logging

import torch
import torch.nn as nn
from transformers import (
    Dinov2Model,
    Dinov2Config,
    CLIPVisionModel,
    CLIPVisionConfig,
)

from model import FREEZE, DEVICE

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


class Dinov2Encoder(nn.Module):
    def __init__(
        self,
        config: Dinov2Config,
        freeze=True,
        **kwargs,
    ):
        super(Dinov2Encoder, self).__init__()
        self.config = config
        self.freeze = freeze
        self.encoder = Dinov2Model(self.config)
        self.freeze_network(self.freeze)

    def load_weights(self, dinov2_weights_path):
        dinov2_state_dict = torch.load(dinov2_weights_path, weights_only=True)
        self.encoder.load_state_dict(dinov2_state_dict)

    def forward(self, pixel_values):
        last_hidden_state = self.encoder(pixel_values).last_hidden_state
        cls_tokens = last_hidden_state[:, 0, :]
        return cls_tokens.contiguous(), last_hidden_state

    def freeze_network(self, freeze=FREEZE):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    @classmethod
    def from_pretrained(cls, model_path, config, freeze, **kwargs):
        if model_path is None and isinstance(model_path, str):
            raise ValueError(
                f"model_path must be of type str found type {type(model_path)}"
            )

        model = cls(config, freeze, **kwargs)

        if os.path.exists(model_path):  # Check if the path exists locally
            model.load_weights(model_path)
        else:
            model.encoder = Dinov2Model.from_pretrained(model_path, config=config)

        model.freeze_network(freeze=freeze)

        return model


def load_dinov2_image_encoder(
    config: dict = None, freeze=FREEZE, pretrained_model_path=None, **kwargs
):
    try:
        if config:
            config = Dinov2Config(**config)
        else:
            config = Dinov2Config()

        if pretrained_model_path:
            logger.info(
                f"Loading pretrained DinoV2 Image Encoder from: {pretrained_model_path}"
            )
            return Dinov2Encoder.from_pretrained(
                pretrained_model_path, config, freeze, **kwargs
            )

        logger.info(
            f"Initializing new DinoV2 Image Encoder with config: {config} and freeze: {freeze}"
        )
        return Dinov2Encoder(config, freeze, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load DinoV2 Image Encoder: {e}")
        raise


class CLIPVisionEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig, freeze=True, **kwargs):
        super(CLIPVisionEncoder, self).__init__()
        self.config = config
        self.freeze = freeze
        self.encoder = CLIPVisionModel(self.config)
        self.freeze_network(self.freeze)

    def load_weights(self, clip_weights_path):
        clip_state_dict = torch.load(clip_weights_path, weights_only=True)
        self.encoder.load_state_dict(clip_state_dict)

    def forward(self, pixel_values):
        last_hidden_state = self.encoder(pixel_values).last_hidden_state
        cls_tokens = last_hidden_state[:, 0, :]
        return cls_tokens.contiguous(), last_hidden_state

    def freeze_network(self, freeze=FREEZE):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    @classmethod
    def from_pretrained(cls, model_path, config, freeze, **kwargs):
        if model_path is None and isinstance(model_path, str):
            raise ValueError(
                f"model_path must be of type str found type {type(model_path)}"
            )

        model = cls(config, freeze=False, **kwargs)

        if os.path.exists(model_path):  # Check if the path exists locally
            model.load_weights(model_path)
        else:
            model.encoder = CLIPVisionModel.from_pretrained(model_path, config=config)

        model.freeze_network(freeze=freeze)

        return model


def load_clip_vision_encoder(
    config: dict = None,
    freeze=FREEZE,
    pretrained_model_path=None,
    **kwargs,
):
    try:
        if config:
            config = CLIPVisionConfig(**config)
        else:
            config = CLIPVisionConfig()

        if pretrained_model_path:
            logger.info(
                f"Loading pretrained CLIPVisionWithProjection Image Encoder from: {pretrained_model_path}"
            )
            return CLIPVisionEncoder.from_pretrained(
                pretrained_model_path, config, freeze, **kwargs
            )

        logger.info(
            f"Initializing new CLIPVisionWithProjection Image Encoder with config: {config}"
        )
        return CLIPVisionEncoder(config, freeze, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load CLIPVisionWithProjection Image Encoder: {e}")
        raise


IMAGE_MODEL_DICT = {
    "dinov2": (
        load_dinov2_image_encoder,
        {
            "config": None,
            "freeze": FREEZE,
            "pretrained_model_path": "/home/harsh/Desktop/Projects/PaperImplementations/ShowAndTell/weights/dinov2-base-weights.pth",
        },
    ),
    "clipvision": (
        load_clip_vision_encoder,
        {
            "config": None,
            "freeze": FREEZE,
            "pretrained_model_path": "openai/clip-vit-base-patch32",
        },
    ),
}


def load_image_model(
    model_name, model_config=None, freeze=None, pretrained_model_path=None
):
    """
    Dynamically fetch and initialize a model based on the model string.

    Args:
        model_name (str): The key corresponding to the desired model in MODEL_DICT.
        model_path (str, optional): Custom model path to override the default path in MODEL_DICT.
        model_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT.

    Returns:
        Model object initialized with the specified parameters.

    Raises:
        ValueError: If the model string is not registered in MODEL_DICT.
    """

    if model_name in IMAGE_MODEL_DICT:
        func, kwargs = IMAGE_MODEL_DICT[model_name]

        if model_config is not None:
            kwargs["config"] = model_config

        if freeze is not None:
            kwargs["freeze"] = freeze

        if pretrained_model_path is not None:
            kwargs["pretrained_model_path"] = pretrained_model_path

        logger.info(f"Initializing model '{model_name}'")
        return func(**kwargs).to(device=DEVICE)
    else:
        logger.error(f"Model '{model_name}' is not registered in IMAGE_MODEL_DICT.")
        raise ValueError(f"Model '{model_name}' is not registered in IMAGE_MODEL_DICT.")
