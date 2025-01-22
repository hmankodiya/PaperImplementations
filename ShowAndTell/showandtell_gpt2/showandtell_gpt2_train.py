import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser
import logging

import torch
from transformers import Dinov2Config, Seq2SeqTrainer, Seq2SeqTrainingArguments

from dataset import ImageCaptionDataset, ImageTextCollator
from image_model.image_model import load_dinov2_image_encoder
from tokenizer import load_tokenizer
from utils import (
    get_huggingface_trainer_config,
    get_showandtell_gpt2_model_config,
    read_yaml,
    make_dir,
    get_split_config,
    get_tokenizer_config,
    get_dataset_config,
    get_model_config,
    get_image_model_config,
    get_text_model_config,
)
from showandtell_gpt2.showandtell_gpt2_model import (
    load_showandtell_gpt2,
    load_pretrained_gpt2_model,
)
from model import METRICS_DICT, compute_metrics

PRECISION = "medium"
torch.set_float32_matmul_precision(PRECISION)

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
    train_dataset_path, train_sampling_config, train_dataset_config = (
        get_dataset_config(train_split_config)
    )

    sampling_fn_name, sampling_fn_args = (
        train_sampling_config.pop("sampling_fn_name", "random_sample"),
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
    val_dataset = None
    if args.validate:
        if not val_split_config:
            raise ValueError(
                "Validation split is missing while the `--validate` argument is set to `True`. Please provide a validation split or disable validation."
            )

        val_dataset_path, val_sampling_config, val_dataset_config = get_dataset_config(
            val_split_config
        )

        val_sampling_fn_name, val_sampling_fn_args = (
            val_sampling_config.pop("sampling_fn_name", "random_sample"),
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
    test_dataset = None
    if args.test:
        if not test_split_config:
            raise ValueError(
                "Test split is missing while the `--test` argument is set to `True`. Please provide a test split or disable testing."
            )

        test_dataset_path, test_sampling_config, test_dataset_config = (
            get_dataset_config(test_split_config)
        )

        test_sampling_fn_name, test_sampling_fn_args = (
            test_sampling_config.pop("sampling_fn_name", "random_sample"),
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
    text_model_config.update(
        {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        }
    )
    gpt2_text_encoder = load_pretrained_gpt2_model(
        text_model_path,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
        **text_model_config,
    )

    # Combined model initialization (Show and Tell)
    (
        showandtell_gpt2_core_model_name,
        showandtell_gpt2_core_model_path,
        showandtell_gpt2_core_config,
    ) = get_showandtell_gpt2_model_config(model_config)
    showandtell_gpt2 = load_showandtell_gpt2(
        tokenizer,
        image_encoder,
        gpt2_text_encoder,
        pretrained_model_path=showandtell_gpt2_core_model_path,
    )

    # Training DataLoader
    image_text_collator = ImageTextCollator(
        tokenizer,
        padding=train_dataset_config.get("padding", True),
        return_tensors=train_dataset_config.get("return_tensors", "pt"),
    )

    trainer_config = get_huggingface_trainer_config(config)
    trainer_config["logging_dir"] = os.path.join(
        trainer_config["output_dir"], "runs", trainer_config["run_name"]
    )
    save_trained_model = trainer_config.pop("save_trained_model", True)
    resume_from_checkpoint = trainer_config.pop("resume_from_checkpoint", None)
    trainer_args = Seq2SeqTrainingArguments(**trainer_config)
    trainer = Seq2SeqTrainer(
        showandtell_gpt2,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=image_text_collator,
        processing_class=None,
        compute_metrics=lambda x: (compute_metrics(x, tokenizer, METRICS_DICT)),
    )

    logger.info("Training started.")
    training_outs = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training finished.")

    if save_trained_model:
        logger.info(f'Saving model at {trainer_config["logging_dir"]}')
        torch.save(
            showandtell_gpt2.state_dict(),
            os.path.join(trainer_config["logging_dir"], "model.pth"),
        )
