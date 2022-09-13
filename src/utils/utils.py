import os
import logging
import yaml
from easydict import EasyDict as edict


def yaml_load(fileName):
    fc = None
    with open(fileName, "r") as f:
        fc = edict(yaml.safe_load(f))

    return fc


def create_logger(save_path, filename="log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] - %(message)s", datefmt="%y/%m/%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(save_path, filename))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
