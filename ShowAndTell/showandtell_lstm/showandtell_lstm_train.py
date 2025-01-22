import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser
import logging

import torch
import pytorch_lightning as pl
from transformers import Dinov2Config

from dataset import ImageCaptionDataset, ImageTextCollator
from image_model.image_model import load_dinov2_image_encoder
from tokenizer import load_tokenizer
from utils import (
    read_yaml,
    make_dir,
    get_split_config,
    get_tokenizer_config,
    get_dataset_config,
    get_model_config,
    get_image_model_config,
    get_text_model_config,
    get_showandtell_lstm_trainer_config,
    get_showandtell_lstm_model_config,
)

from showandtell_lstm.showandtell_lstm_model import (
    load_lstm_encoder,
    load_showandtell_lstm,
    load_lightning_showandtell_lstm,
)

PRECISION = "medium"
torch.set_float32_matmul_precision(PRECISION)

SAVE_PATH = "inherit"
CHECKPOINT_PATH = "inherit"
LOGGER_PATH = "/home/harsh/Desktop/Projects/PaperImplementations/ShowAndTell/showandtell_lstm/TrainingLogs"


logging.basicConfig(
    filename="/home/harsh/Desktop/Projects/PaperImplementations/ShowAndTell/logs.txt",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger(__name__)  # Logger for the main script


if __name__ == "__main__":
    """
    Main script for training and evaluating an image captioning model.

    This script initializes and configures:
        - Tokenizers
        - Datasets and DataLoaders
        - Models (image encoder, text encoder, and combined model)
        - PyTorch Lightning trainer for training and validation
    """
    # Argument parser setup
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        default="./configs/train_config.yaml",
        help="Path to the configuration file (YAML or JSON format).",
    )
    arg_parser.add_argument(
        "--validate",
        type=bool,
        required=False,
        default=False,
        help="Validate on val split.",
    )
    arg_parser.add_argument(
        "--test",
        type=bool,
        required=False,
        default=False,
        help="Test on test split.",
    )

    args = arg_parser.parse_args()
    logger.info(f"Arguments: {args.__dict__}")

    # Load configuration
    config = read_yaml(args.config)

    # Initialize Tokenizer
    tokenizer_name, tokenizer_path, tokenizer_config = get_tokenizer_config(config)
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_path=tokenizer_path,
        tokenizer_config=tokenizer_config,
    )


    # Initialize Dataset
    dataset_desc, (train_split_config, val_split_config, test_split_config) = (
        get_split_config(config)
    )

    # Prepare training dataset
    train_dataset_path, train_sampling_config, train_dataset_config = get_dataset_config(
        train_split_config
    )
    if train_sampling_config:
        sampling_fn_name, sampling_fn_args = (
            train_sampling_config.pop("sampling_fn_name", None),
            train_sampling_config,
        )

    train_dataset = ImageCaptionDataset(
        tokenizer=tokenizer,
        dataset_path=train_dataset_path,
        sampling_fn=sampling_fn_name,
        sampling_fn_args=sampling_fn_args,
        return_tensors=None,
        **train_dataset_config,
    )

    # Prepare validation dataset (if applicable)
    if args.validate:
        if not val_split_config:
            raise ValueError(
                "Validation split is missing while the `--validate` argument is set to `True`. Please provide a validation split or disable validation."
            )

        val_dataset_path, val_sampling_config, val_dataset_config = get_dataset_config(
            val_split_config
        )

        if val_sampling_config:
            val_sampling_fn_name, val_sampling_fn_args = (
                val_sampling_config.pop("sampling_fn_name", None),
                val_sampling_config,
            )

        val_dataset = ImageCaptionDataset(
            tokenizer=tokenizer,
            dataset_path=val_dataset_path,
            sampling_fn=val_sampling_fn_name,
            sampling_fn_args=val_sampling_fn_args,
            return_tensors=None,
            **val_dataset_config,
        )
        logger.info(
            f"Loaded Val Dataset: {dataset_desc}, Dataset Length: {len(val_dataset)}."
        )

    # Prepare test dataset (if applicable)
    if args.test:
        if not test_split_config:
            raise ValueError(
                "Test split is missing while the `--test` argument is set to `True`. Please provide a test split or disable testing."
            )

        test_dataset_path, test_sampling_config, test_dataset_config = (
            get_dataset_config(test_split_config)
        )

        if test_sampling_config:
            test_sampling_fn_name, test_sampling_fn_args = (
                test_sampling_config.pop("sampling_fn_name", None),
                test_sampling_config,
            )

        test_dataset = ImageCaptionDataset(
            tokenizer=tokenizer,
            dataset_path=test_dataset_path,
            sampling_fn=test_sampling_fn_name,
            sampling_fn_args=test_sampling_fn_args,
            return_tensors=None,
            **val_dataset_config,
        )
        logger.info(
            f"Loaded Test Dataset: {dataset_desc}, Dataset Length: {len(test_dataset)}."
        )

    # Initialize Models
    model_config = get_model_config(config)

    # Image encoder initialization
    image_model_name, image_model_path, freeze, image_model_config = (
        get_image_model_config(model_config)
    )
    dinov2_config = Dinov2Config(**image_model_config)
    image_encoder = load_dinov2_image_encoder(dinov2_config, freeze, image_model_path)

    # Text encoder initialization
    text_model_name, text_model_path, text_model_config = get_text_model_config(
        model_config
    )
    lstm_text_encoder = load_lstm_encoder(
        len(tokenizer), pretrained_model_path=text_model_path, **text_model_config
    )

    # Combined model initialization (Show and Tell)
    (
        showandtell_lstm_core_model_name,
        showandtell_lstm_core_model_path,
        showandtell_lstm_core_config,
    ) = get_showandtell_lstm_model_config(model_config)
    showandtell_lstm_core = load_showandtell_lstm(
        tokenizer,
        image_encoder,
        lstm_text_encoder,
        pretrained_model_path=showandtell_lstm_core_model_path,
    )

    # PyTorch Lightning model wrapper
    model = load_lightning_showandtell_lstm(tokenizer, showandtell_lstm_core)

    # Initialize Trainer and DataLoader
    batch_size, trainer_args, logger_config, checkpoint_config, save_config = (
        get_showandtell_lstm_trainer_config(config)
    )

    # Training DataLoader
    train_image_text_collator = ImageTextCollator(
        tokenizer,
        padding=train_dataset_config.get("padding", True),
        return_tensors=train_dataset_config.get("return_tensors", "pt"),
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=train_image_text_collator,
    )

    # Validation DataLoader
    val_dataloader = None
    if args.validate:
        val_image_text_collator = ImageTextCollator(
            tokenizer,
            padding=val_dataset_config.get("padding", True),
            return_tensors=val_dataset_config.get("return_tensors", "pt"),
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=1, collate_fn=val_image_text_collator
        )

    # Test DataLoader
    test_dataloader = None
    if args.test:
        test_image_text_collator = ImageTextCollator(
            tokenizer,
            padding=test_dataset_config.get("padding", True),
            return_tensors=test_dataset_config.get("return_tensors", "pt"),
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, collate_fn=test_image_text_collator
        )

    # Initialize Trainer
    train_logger = False
    if logger_config:
        train_logger = pl.loggers.TensorBoardLogger(**logger_config)

    if checkpoint_config.get("dirpath", CHECKPOINT_PATH) == "inherit":
        checkpoint_config["dirpath"] = os.path.join(
            LOGGER_PATH, logger_config.get("name", "lightning_logs"), "checkpoint"
        )

    model_checkpoint = pl.callbacks.ModelCheckpoint(**checkpoint_config)
    trainer = pl.Trainer(
        logger=train_logger,
        callbacks=[model_checkpoint],
        **trainer_args,
    )

    # Start training
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    if save_config and save_config.get("save_state_dict", True):
        save_path = save_config.get("save_path", SAVE_PATH)
        if save_path == "inherit":
            save_path = os.path.join(
                LOGGER_PATH, logger_config.get("name", ""), "model"
            )

        make_dir(save_path)
        save_path = os.path.join(save_path, "showandtell_lstm_core.pth")
        logger.info(f"Saving trained model at {save_path}")
        torch.save(showandtell_lstm_core.state_dict(), save_path)
