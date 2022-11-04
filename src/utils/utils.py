import os
import io
import sys
import time
import yaml
import torch
import logging
import numpy as np
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

from src.model.loss import *


def yaml_load(fileName):
    fc = None
    with open(fileName, "r") as f:
        fc = edict(yaml.safe_load(f))

    return fc


def create_logger(save_path, phase):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="[%(asctime)s] - %(message)s", datefmt="%y/%m/%d %H:%M:%S")

    t = time.strftime("%H:%M:%S")
    filename = f"{phase}_{t}.log"
    os.makedirs(save_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(save_path, filename), mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def batch_collate(pre_batch_data):
    max_len = -1
    for data in pre_batch_data:
        if data["label"].shape[0] > max_len:
            max_len = data["label"].shape[0]
    for data in pre_batch_data:
        # add original shape
        ori_len = data["label"].shape[0]
        data["point_cnt"] = np.array([ori_len])

        # padding
        data["label"] = np.pad(data["label"], ((0, max_len - ori_len)), "constant", constant_values=0)
        data["pcd"] = np.pad(data["pcd"], ((0, max_len - ori_len), (0, 0)), "constant", constant_values=0)
        data["grid_mask"] = np.pad(data["grid_mask"], ((0, max_len - ori_len)), "constant", constant_values=0)
    batch_data = dict()
    for key in pre_batch_data[0].keys():
        batch_data[key] = torch.from_numpy(
            np.stack(([pre_batch_data[i][key] for i in range(len(pre_batch_data))]), axis=0)
        ).float()
    return batch_data


def get_loss_function(loss_name):
    if loss_name == "SoftmaxWithloss":
        return SoftmaxWithloss()
    elif loss_name == "MSE":
        return MSE()
    elif loss_name == "FocalLoss":
        return FocalLoss()
    else:
        raise ModuleNotFoundError


class SummaryWriterAvg(SummaryWriter):
    def __init__(self, *args, dump_period=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_period = dump_period
        self._avg_scalars = dict()

    def add_scalar(self, tag, value, global_step=None, disable_avg=False):
        if disable_avg or isinstance(value, (tuple, list, dict)):
            super().add_scalar(tag, np.array(value), global_step=global_step)
        else:
            if tag not in self._avg_scalars:
                self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)
            avg_scalar = self._avg_scalars[tag]
            avg_scalar.add(value)

            if avg_scalar.is_full():
                super().add_scalar(tag, avg_scalar.value, global_step=global_step)
                avg_scalar.reset()


class ScalarAccumulator(object):
    def __init__(self, period):
        self.sum = 0
        self.cnt = 0
        self.period = period

    def add(self, value):
        self.sum += value
        self.cnt += 1

    @property
    def value(self):
        if self.cnt > 0:
            return self.sum / self.cnt
        else:
            return 0

    def reset(self):
        self.cnt = 0
        self.sum = 0

    def is_full(self):
        return self.cnt >= self.period

    def __len__(self):
        return self.cnt
