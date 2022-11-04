import numpy as np
from scipy import interpolate
import torch.utils.data as torch_data


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, pcd_path, cfg):
        super().__init__()
        self.pcd_path = pcd_path
        self.cfg = cfg

    def __getitem__(self, index):
        # load pcd
        pcd = self._load_pcd(self.pcd_path[index])
        # preprocess pcd
        _, pcd = self.normalize_range(None, pcd, self.cfg.pointcloud)
        input_matrix, _, grid_mask, bin_idx = self.encoding_pointcloud(None, pcd, self.cfg)
        data_dict = {"pcd": pcd, "input_matrix": input_matrix, "grid_mask": grid_mask, "bin_idx": bin_idx}
        return data_dict

    def __len__(self):
        return len(self.pcd_path)

    def _load_pcd(self, path):
        pcd = np.fromfile(path, dtype=np.float32)
        pcd = pcd.reshape((-1, 4))
        if (np.where(pcd[:, :3] == [0.0, 0.0, 0.0])[0]).shape[0] == 0:
            raise Exception("The pcd have [0.0, 0.0, 0.0] point." + path)

        return pcd

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
        if label is not None:
            label_ = np.zeros_like(label)
            label_[np.isin(label, ground_label)] = 1

        grid_mask = bin_idx[polar_radius, polar_cone]
        label_matrix_non_ground = np.zeros_like(bin_idx.flatten())
        label_matrix_ground = np.zeros_like(bin_idx.flatten())
        input_matrix = []
        for i in bin_idx.flatten():
            mask = grid_mask == i
            if np.sum(mask) != 0:
                bin_pts = pcd[mask]
                depth = np.linalg.norm(bin_pts[:, :2], axis=1)
                depth = np.mean(depth)
                depth = np.log(depth)
                height = bin_pts[:, 2] / H
                height = np.mean(height)
                intensity = np.mean(bin_pts[:, 3])

            else:
                depth, height, intensity = [np.nan, np.nan, np.nan]

            feature = [depth, height, intensity]
            input_matrix.append(feature)
            if label is not None:
                masked_label = label_[mask]
                cnt_pts = masked_label.shape[0]
                if cnt_pts == 0:
                    label_matrix_non_ground[i] = -1
                    label_matrix_ground[i] = -1
                else:
                    label_matrix_non_ground[i] = (cnt_pts - np.sum(masked_label)) / cnt_pts
                    label_matrix_ground[i] = np.sum(masked_label) / cnt_pts
        input_matrix = np.array(input_matrix).reshape(R, C, 3)
        if label is not None:
            label_matrix = np.array([label_matrix_non_ground, label_matrix_ground]).reshape(2, R, C)
        input_matrix = np.einsum("ijk->kij", input_matrix)
        # interpolate empty bins
        for i in range(input_matrix.shape[0]):
            c = input_matrix[i, :, :]
            x = np.arange(0, c.shape[1])
            y = np.arange(0, c.shape[0])
            # mask invalid values
            array = np.ma.masked_invalid(c)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]

            interpolated_channel = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest")
            input_matrix[i, :, :] = interpolated_channel

        return input_matrix, label_matrix, grid_mask, bin_idx

    def normalize_range(self, label, pcd, param):
        radius, z_min, z_max = param.target_range

        n_pcd = None
        n_label = None

        r_cond = np.sqrt(pcd[:, 0] ** 2 + pcd[:, 1] ** 2) < radius
        z_cond = np.logical_and(pcd[:, 2] > z_min, pcd[:, 2] <= z_max)
        cond = r_cond * z_cond

        n_pcd = pcd[cond]
        if label is not None:
            n_label = label[cond]

        return n_label, n_pcd

    def prepare_data(self, input_dict):
        data_dict = None
        label = input_dict["label"]
        pcd = input_dict["pcd"]

        label, pcd = self.normalize_range(label, pcd, self.cfg.pointcloud)
        ## TODO:augmentation
        input_matrix, label_matrix, grid_mask, bin_idx = self.encoding_pointcloud(label, pcd, self.cfg)
        input_dict["pcd"] = pcd
        input_dict["label"] = label
        input_dict["input_matrix"] = input_matrix
        input_dict["label_matrix"] = label_matrix
        input_dict["grid_mask"] = grid_mask
        input_dict["bin_idx"] = bin_idx

        return input_dict


def decoding_pointcloud(output_matrix, grid_mask):
    decoded_output = np.zeros((grid_mask.shape[0]), dtype=int)
    non_ground = output_matrix[0].flatten()
    ground = output_matrix[1].flatten()
    pred = ground > non_ground
    pred = np.where(pred)
    decoded_output[np.isin(grid_mask, pred)] = 1

    return decoded_output
