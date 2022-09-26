import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model.model import GroundNet
from src.model.loss import SoftmaxWithloss
from src.data.SemanticKITTI import SemanticKITTI
from src.data.dataset import decoding_pointcloud
from src.utils.utils import yaml_load, create_logger, batch_collate, get_loss_function, SummaryWriterAvg


def init_model(cfg):
    model = GroundNet()
    return model


def save_model(model, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(path):
    model = GroundNet()
    model.load_state_dict(torch.load(path))
    return model


def vis_output(output, batch_data, select_idx=0, ground_label=[9, 11, 12, 17]):
    idx = select_idx
    output_ = output[idx].detach().numpy()
    grid_mask_ = batch_data["grid_mask"][idx].detach().numpy()
    pcd_ = batch_data["pcd"][idx].detach().numpy()
    label_ = batch_data["label"][idx].detach().numpy()
    decoded_output = decoding_pointcloud(output_, grid_mask_)
    fig = plt.figure(figsize=(15, 15))
    plt.axes().set_aspect("equal")

    ground_ = np.zeros_like(label_)
    ground_[np.isin(label_, ground_label)] = 1
    plt.subplot(2, 2, 1)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=label_, s=1)
    plt.subplot(2, 2, 2)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=ground_, s=1)
    plt.subplot(2, 2, 3)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=decoded_output, s=1)
    plt.subplot(2, 2, 4)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=ground_ == decoded_output, s=1, cmap="prism")

    return fig


def train(cfg, trainset):
    sw = SummaryWriterAvg(log_dir=cfg.path.exp_path, flush_secs=10, dump_period=2)
    model = init_model(cfg)
    param = cfg.learn
    loss_criterion = SoftmaxWithloss()
    # loss_criterion = get_loss_function(cfg.model.loss)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80, 90], gamma=0.1)

    train_dataloader = DataLoader(trainset, param.batch_size, drop_last=True, num_workers=4, collate_fn=batch_collate)

    model.train()
    for epoch in tqdm(range(param.total_epoch)):

        for it, batch_data in enumerate(tqdm(train_dataloader, total=int(len(train_dataloader)))):
            global_step = epoch * len(trainset) + it
            with torch.set_grad_enabled(True):
                output = model(batch_data)
                loss = loss_criterion(output, batch_data["label_matrix"])
                loss = torch.mean(loss)
                sw.add_scalar(
                    tag="train_loss", value=loss.item(), global_step=global_step,
                )

                optim.zero_grad()
                loss.backward()
                optim.step()

        sw.add_scalar(
            tag="learning_rate", value=optim.param_groups[0]["lr"], global_step=global_step,
        )
        vis = vis_output(output, batch_data)
        sw.add_figure(tag="vis", figure=vis, global_step=epoch)

        if epoch % param.save_interval == 0:
            save_model(model, os.path.join(cfg.path.exp_path, f"ckpts/{epoch}.pth"))
        lr_scheduler.step()


if __name__ == "__main__":

    cfg = yaml_load("config.yaml")
    logger = create_logger("../..", "tmp.log")

    dataset = SemanticKITTI(cfg, logger)

    train(cfg, dataset)
