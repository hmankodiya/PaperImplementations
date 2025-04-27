import os
import sys
import logging
from argparse import ArgumentParser

import torch
from transformers import Trainer, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import CLIPDataset, ImageTextCollator
from model import load_tokenizer, load_model
from utils import (
    read_yaml,
    get_trainer_config,
    get_tokenizer_config,
    get_model_config,
    get_split_config,
    get_dataset_config,  
)

PRECISION = "medium"
torch.set_float32_matmul_precision(PRECISION)

# Configure logger
logging.basicConfig(
    filename="./logs/clip_training_logs.txt",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """
    Training script for CLIP model fine-tuning.
    """

    # Argument parsing
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
    )
    arg_parser.add_argument(
        "--just_eval",
        action="store_true",
        help="If true, only evaluate the model (skip training).",
    )
    args = arg_parser.parse_args()
    logger.info(f"Arguments: {args.__dict__}")

    # Load configuration
    config = read_yaml(args.config)

    # Load Tokenizer
    tokenizer_name, tokenizer_path, tokenizer_extra_config = get_tokenizer_config(
        config
    )
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_path=tokenizer_path,
        tokenizer_config=tokenizer_extra_config,
    )

    # Load Model
    model_name, model_path, model_extra_config, pretrained = get_model_config(config)
    model = load_model(
        model_name=model_name,
        model_path=model_path,
        model_config=model_extra_config,
        pretrained=pretrained,
    )

    # Prepare Split Config
    dataset_desc, (train_split_config, val_split_config, test_split_config) = (
        get_split_config(config)
    )

    # Train Dataset
    train_dataset = None
    if not args.just_eval:
        train_dataset_path, train_sampling_config, train_dataset_config = (
            get_dataset_config(train_split_config)
        )

        train_dataset = CLIPDataset(
            tokenizer=tokenizer,
            dataset_path=train_dataset_path,
            return_tensors=None,
            **train_dataset_config,  # pass extra dataset args
        )

    # Validation Dataset
    val_dataset = None
    if val_split_config is not None:
        val_dataset_path, val_sampling_config, val_dataset_config = get_dataset_config(
            val_split_config
        )

        val_dataset = CLIPDataset(
            tokenizer=tokenizer,
            dataset_path=val_dataset_path,
            return_tensors=None,
            **val_dataset_config,
        )

    # Data Collator
    data_collator = ImageTextCollator(
        tokenizer,
        padding="longest",
        return_tensors="pt",
        return_loss=True,
    )

    # Trainer Config
    trainer_config = get_trainer_config(config)
    trainer_config["logging_dir"] = os.path.join(
        trainer_config["output_dir"], "runs", trainer_config["run_name"]
    )
    resume_from_checkpoint = trainer_config.pop("resume_from_checkpoint", None)
    save_trained_model = trainer_config.pop("save_trained_model", True)

    # TrainingArguments
    trainer_args = TrainingArguments(**trainer_config)

    # Trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset if not args.just_eval else None,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train / Evaluate
    if not args.just_eval:
        logger.info("Training started.")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training finished.")

        if save_trained_model:
            logger.info(f"Saving model at {trainer_config['output_dir']}")
            trainer.save_model(trainer_config["output_dir"])
    else:
        logger.info("Evaluation started.")
        results = trainer.evaluate()
        logger.info(f"Evaluation finished. Results: {results}")
        print(results)
