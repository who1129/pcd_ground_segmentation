from xmlrpc.client import boolean
from matplotlib.pyplot import axis
import numpy as np
from scipy import interpolate
import torch.utils.data as torch_data


class DatasetTemplate(torch_data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__():
        raise NotImplementedError

    def encoding_pointcloud(self, label, pcd, param):
        H = param.encoder.H
        R_unit = param.encoder.R_unit
        C = param.encoder.C
        target_range = param.pointcloud.target_range
        radius = target_range[0]
        input_matrix = None
        label_matrix = None

        # make polar bins
        polar_cone_size = 360.0 / (C - 1)
        angle = np.arctan2(pcd[:, 0], pcd[:, 1]) * 180 / np.pi
        angle += 180.0
        polar_cone = np.floor(angle / polar_cone_size).astype(int)
        R = int(radius / R_unit)
        d = np.sqrt(pcd[:, 0] ** 2 + pcd[:, 1] ** 2)
        polar_radius = np.floor(d / (radius / (R - 1))).astype(int)
        bin_idx = np.arange(0, R * C).reshape(R, C)
        label_matrix_non_ground = []
        label_matrix_ground = []
        ground_label = [9, 11, 12, 17]
        label_ = np.zeros_like(label)
        for l in ground_label:
            label_[label == l] = 1

        grid_mask = bin_idx[polar_radius, polar_cone]
        label_matrix_non_ground = np.zeros_like(bin_idx.flatten())
        label_matrix_ground = np.zeros_like(bin_idx.flatten())
        input_matrix = []
        for i in bin_idx.flatten():
            mask = grid_mask == i
            bin_pts = pcd[mask]

            depth = np.linalg.norm(bin_pts[:, :2], axis=1)
            depth = np.mean(depth)
            depth = np.log(depth)
            height = np.mean(bin_pts[:, 2]) / 3
            intensity = np.mean(bin_pts[:, 3])

            feature = [depth, height, intensity]
            input_matrix.append(feature)

            masked_label = label_[mask]
            cnt_pts = masked_label.shape[0]
            if cnt_pts == 0:
                label_matrix_non_ground[i] = -1
                label_matrix_ground[i] = -1
            else:
                label_matrix_non_ground[i] = (cnt_pts - np.sum(masked_label)) / cnt_pts

                label_matrix_ground[i] = np.sum(masked_label) / cnt_pts
        input_matrix = np.array(input_matrix).reshape(R, C, 3)
        label_matrix = np.array([label_matrix_non_ground, label_matrix_ground]).reshape(
            2, R, C
        )
        label_matrix = np.einsum("ijk->jki", label_matrix)

        # interpolate empty bins
        for i in range(3):
            c = input_matrix[:, :, i]
            x = np.arange(0, c.shape[1])
            y = np.arange(0, c.shape[0])
            # mask invalid values
            array = np.ma.masked_invalid(c)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]

            interpolated_channel = interpolate.griddata(
                (x1, y1), newarr.ravel(), (xx, yy), method="nearest"
            )
            input_matrix[:, :, i] = interpolated_channel

        return input_matrix, label_matrix, grid_mask, bin_idx

    def decoding_pointcloud(self, output_matrix, grid_mask, pcd, bin_idx):
        decoded_output = np.zeros((pcd.shape[0]), dtype=int)
        non_ground = output_matrix[:, :, 0].flatten()
        ground = output_matrix[:, :, 1].flatten()
        pred = ground > non_ground

        decoded_output = pred[grid_mask]
        return decoded_output

    def normalize_range(self, label, pcd, param):
        radius, z_min, z_max = param.target_range

        n_pcd = None
        n_label = None

        r_cond = np.sqrt(pcd[:, 0] ** 2 + pcd[:, 1] ** 2) < radius
        z_cond = np.logical_and(pcd[:, 2] > z_min, pcd[:, 2] <= z_max)
        cond = r_cond * z_cond

        n_pcd = pcd[cond]
        n_label = label[cond]

        return n_label, n_pcd

    def prepare_data(self, input_dict):
        data_dict = None
        label = input_dict["label"]
        pcd = input_dict["pcd"]

        label, pcd = self.normalize_range(label, pcd, self.cfg.pointcloud)
        # to-do:augmentation
        input_matrix, label_matrix, grid_mask, bin_idx = self.encoding_pointcloud(
            label, pcd, self.cfg
        )
        input_dict["pcd"] = pcd
        input_dict["label"] = label
        input_dict["input_matrix"] = input_matrix
        input_dict["label_matrix"] = label_matrix
        input_dict["grid_mask"] = grid_mask
        input_dict["bin_idx"] = bin_idx

        return input_dict
