import os
import yaml

import logging

logger = logging.getLogger(__name__)


def write_yaml(filename, dictionary):
    try:
        with open(filename, "w") as f:
            yaml.safe_dump(dictionary, f)
            logger.info(f"YAML dumped at {filename}")
        return
    except Exception as e:
        return e.message


def read_yaml(filename):
    with open(filename, "r") as f:
        logger.info(f"Loading YAML at {filename}")
        return yaml.safe_load(f)
