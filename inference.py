import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import DatasetTemplate
from src.data.dataset import decoding_pointcloud
from src.model.model import GroundNet
from src.utils.utils import yaml_load, inf_batch_collate


class Inference:
    def __init__(self, cfg_path, device="cpu"):
        self.cfg = yaml_load(cfg_path)
        self.model = None

        self.device = device
        self._check_device()

    def predict(self, pcd_path, batch_size=8):
        """
        Args:
            pcd_path (list(str)): pcd path list
            batch_size (int): batch size at inference phase
        Returns:
            preds (list(list(int))): inferred class of each point (1=ground, 0=non-ground)
        """
        # load encoding data
        dataset = DatasetTemplate(pcd_path, self.cfg)
        valid_dataloader = DataLoader(dataset, batch_size, drop_last=False, num_workers=8, collate_fn=inf_batch_collate)
        preds = []
        with torch.no_grad():
            for it, batch_data in tqdm(enumerate(valid_dataloader), ncols=100):
                input_matrix = batch_data["input_matrix"].to(self.device)
                output = self.model(input_matrix)
                # get decoded predict data
                pred = self._get_pred(output, batch_data)
                preds.extend(pred)

        return preds

    def load_model(self, ckpt_path):
        """
        load pytorch checkpoint file
        """
        self.model = GroundNet()
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.to(self.device)
        self.model.eval()

    def _check_device(self):
        if self.device == "gpu":
            self.device = torch.device("cuda:0")
            if not torch.cuda.is_available():
                raise Exception("GPU not available")

    def _get_pred(self, output, batch_data):
        output_ = output.detach().to("cpu").numpy()
        grid_mask_ = batch_data["grid_mask"].detach().numpy()
        point_cnt_ = batch_data["point_cnt"].detach().numpy()
        pcd_ = batch_data["pcd"].detach().numpy()
        pred = []
        for i in range(pcd_.shape[0]):
            cnt = int(point_cnt_[i])
            decoded_pred = decoding_pointcloud(output_[i], grid_mask_[i])[:cnt].tolist()
            pred.append(decoded_pred)

        return pred


if __name__ == "__main__":
    infr = Inference(cfg_path="config.yaml", device="gpu")
    infr.load_model("experiments/14_srate15-interpolate/ckpts/10.pth")
    path = [
        "/home/ext/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin",
        "/home/ext/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin",
        "/home/ext/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin",
    ]
    print(infr.predict(path))
