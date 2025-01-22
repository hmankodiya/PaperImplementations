import os
import logging

import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config

from model import FREEZE

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
        model = cls(config, freeze, **kwargs)
        model.encoder.load_state_dict(torch.load(model_path, weights_only=True))
        return model


def load_dinov2_image_encoder(
    config, freeze=FREEZE, pretrained_model_path=None, **kwargs
):
    try:
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
