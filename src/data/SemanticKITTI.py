import os
import yaml
import glob
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .dataset import DatasetTemplate


class SemanticKITTI(DatasetTemplate):
    def __init__(self, cfg, logger, split="train"):
        self.cfg = cfg
        self.split = split
        self.label_root = cfg.path.label_root
        self.pcd_root = cfg.path.pcd_root
        self.logger = logger
        self.label_paths = []
        self.label_cfg = self._load_dataset_cfg("src/data/semantic-kitti.yaml")
        self._load_path()

    def _load_path(self):
        self.logger.info(f"Loading SemanticKITTI {self.split} Dataset")
        seq_list = self.cfg.path.split[self.split]
        for sequence in seq_list:
            sequence = "{0:02d}".format(int(sequence))
            if self.split == "valid":
                self.label_paths.extend(
                    glob.glob(os.path.join(self.label_root, f"sequences/{sequence}/**/*.label"), recursive=True)
                )
                """tmp = np.array(
                    glob.glob(os.path.join(self.label_root, f"sequences/{sequence}/**/*.label"), recursive=True)
                )
                mask = np.arange(0, tmp.shape[0]) % 100 == 0
                self.label_paths.extend(tmp[mask].tolist())"""
            else:
                tmp = np.array(
                    glob.glob(os.path.join(self.label_root, f"sequences/{sequence}/**/*.label"), recursive=True)
                )
                mask = np.arange(0, tmp.shape[0]) % self.cfg.tmp.load_train_data_interval == 0
                self.label_paths.extend(tmp[mask].tolist())

        self.logger.info(f"# of label file: {len(self.label_paths)}")
        for l_path in self.label_paths:
            p_path = (
                l_path.replace(self.label_root, self.pcd_root).replace(".label", ".bin").replace("labels", "velodyne")
            )
            assert os.path.isfile(p_path), "This pcd file does not exist! >> " + p_path

    def _load_label(self, path, del_idx):
        label = np.fromfile(path, dtype=np.int32)
        label = label.reshape((-1))  # reshape to vector
        label = label & 0xFFFF  # get lower half for semantics
        label = self.label_cfg["remap_lut"][label]  # remap to xentropy format

        return np.delete(label, del_idx, axis=0)

    def _load_pcd(self, path):
        pcd = np.fromfile(path, dtype=np.float32)
        pcd = pcd.reshape((-1, 4))
        del_idx = np.where(pcd[:, :3] == [0.0, 0.0, 0.0])[0]
        return np.delete(pcd, del_idx, axis=0), del_idx

    def load_raw_data(self, label_path):
        pcd_path = (
            label_path.replace(self.label_root, self.pcd_root).replace(".label", ".bin").replace("labels", "velodyne")
        )
        pcd, del_idx = self._load_pcd(pcd_path)
        label = self._load_label(label_path, del_idx)

        return label, pcd

    def _load_dataset_cfg(self, path):
        # config setting
        DATA = yaml.safe_load(
            open(
                path,
                "r",
                encoding="utf-8",
            )
        )

        # get number of interest classes, and the label mappings
        class_strings = DATA["labels"]
        class_remap = DATA["learning_map"]
        class_inv_remap = DATA["learning_map_inv"]
        class_ignore = DATA["learning_ignore"]
        nr_classes = len(class_inv_remap)

        maxkey = max(class_remap.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(class_remap.keys())] = list(class_remap.values())
        DATA["remap_lut"] = remap_lut

        return DATA

    ## TODO
    def create_prepared_data(self):
        pbar = tqdm(range(len(self.label_paths)))
        pbar.set_description("create_data...")
        for i in pbar:
            label_path = self.label_paths[i]
            label, pcd = self.load_raw_data(label_path)
            data_dict = {"label": label, "pcd": pcd}
            data_dict = self.prepare_data(data_dict)

            l_mat_path = Path(label_path.replace(self.cfg.path.data_root, self.cfg.path.create_data))
            pre, _ = os.path.splitext(l_mat_path)
            i_mat_path = pre + ".input"
            grid_path = pre + ".grid"
            bin_path = pre + ".bin"

            os.makedirs(os.path.dirname(l_mat_path), exist_ok=True)
            np.save(l_mat_path, data_dict["label_matrix"])
            np.save(i_mat_path, data_dict["input_matrix"])
            np.save(grid_path, data_dict["grid_mask"])
            np.save(bin_path, data_dict["bin_idx"])

    def load_prepared_data(self, label_path, data_dict):
        l_mat_path = Path(label_path.replace(self.cfg.path.data_root, self.cfg.path.create_data))
        pre, _ = os.path.splitext(l_mat_path)
        i_mat_path = pre + ".input.npy"
        grid_path = pre + ".grid.npy"
        bin_path = pre + ".bin.npy"

        data_dict["label_matrix"] = np.load(l_mat_path + ".npy")
        data_dict["input_matrix"] = np.load(i_mat_path)
        data_dict["grid_mask"] = np.load(grid_path)
        data_dict["bin_idx"] = np.load(bin_path)

        return data_dict

    def augmentation_data(self, pcd, param):
        random.seed(0)
        if random.random() < param.prob:
            min, max = param.height
            pcd += random.randint(min, max)
        if random.random() < param.prob:
            pcd[:, 0] *= -1
        if random.random() < param.prob:
            pcd[:, 1] *= -1

        return pcd

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label, pcd = self.load_raw_data(label_path)
        # augmentation
        if self.split == "train":
            pcd = self.augmentation_data(pcd, param=self.cfg.augmentation)
        data_dict = {"label": label, "pcd": pcd}
        data_dict = self.prepare_data(data_dict)
        # data_dict = self.load_prepared_data(label_path, data_dict) ## TODO
        return data_dict

    def __len__(self):
        return len(self.label_paths)
