import os
import yaml
import glob
import numpy as np
from .dataset import DatasetTemplate


class SemanticKITTI(DatasetTemplate):
    def __init__(self, cfg, logger, split="train"):
        self.cfg = cfg
        self.label_root = cfg.path.label_root
        self.pcd_root = cfg.path.pcd_root
        self.logger = logger
        self.label_paths = None
        self.label_cfg = self._load_dataset_cfg("src/data/semantic-kitti.yaml")
        self._load_path()

    def _load_path(self):
        self.logger.info("Loading SemanticKITTI Dataset")

        self.label_paths = glob.glob(os.path.join(self.label_root, "**/*.label"), recursive=True)
        pcd_paths = glob.glob(os.path.join(self.pcd_root, "**/*.bin"), recursive=True)

        self.logger.info(f"# of label file: {len(self.label_paths)}\n # of pcd file: {len(pcd_paths)}")
        if len(self.label_paths) != len(pcd_paths):
            for l_path in self.label_paths:
                p_path = (
                    l_path.replace(self.label_root, self.pcd_root)
                    .replace(".label", ".bin")
                    .replace("labels", "velodyne")
                )
                assert os.path.isfile(p_path), "This pcd file does not exist! >> " + p_path

            self.logger.warning("label file and pcd file list do not match.")

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

    def load_batch(self, label_path):
        pcd_path = (
            label_path.replace(self.label_root, self.pcd_root).replace(".label", ".bin").replace("labels", "velodyne")
        )
        pcd, del_idx = self._load_pcd(pcd_path)
        label = self._load_label(label_path, del_idx)

        return label, pcd

    def _load_dataset_cfg(self, path):
        # config setting
        DATA = yaml.safe_load(open(path, "r", encoding="utf-8",))

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

    def __getitem__(self, index):
        try:
            label_path = self.label_paths[index]
            label, pcd = self.load_batch(label_path)
            data_dict = {"label": label, "pcd": pcd}
            data_dict = self.prepare_data(data_dict)
            return data_dict
        except:
            print(self.label_paths[index])
            print(pcd.shape)
            print(label.shape)

    def __len__(self):
        return len(self.label_paths)
