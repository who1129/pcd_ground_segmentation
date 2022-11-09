import os
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from eval import evalutation
from src.model.model import GroundNet
from src.data.SemanticKITTI import SemanticKITTI
from src.data.dataset import decoding_pointcloud
from src.utils.utils import yaml_load, create_logger, batch_collate, get_loss_function, SummaryWriterAvg


def init_model(cfg):
    model = GroundNet()
    return model


def save_config(dst_path, src_path="config.yaml"):
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    shutil.copyfile(src_path, os.path.join(dst_path, "config.yaml"))


def save_model(model, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(path):
    model = GroundNet()
    model.load_state_dict(torch.load(path))
    return model


def get_pred(output, batch_data):
    output_ = output.detach().to("cpu").numpy()
    grid_mask_ = batch_data["grid_mask"].detach().numpy()
    label_ = batch_data["label"].detach().numpy()
    point_cnt_ = batch_data["point_cnt"].detach().numpy()
    label = []
    pred = []
    for i in range(label_.shape[0]):
        pred.append(decoding_pointcloud(output_[i], grid_mask_[i])[: int(point_cnt_[i])])
        label.append(label_[i][: int(point_cnt_[i])])

    return pred, label


def vis_output(output, batch_data, select_idx=0, ground_label=[9, 11, 12, 17]):
    idx = select_idx
    output_ = output[idx].detach().to("cpu").numpy()
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
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=ground_, s=1)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=decoded_output, s=1)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=ground_ == decoded_output, s=1, cmap="prism")
    plt.colorbar()

    return fig


def train(cfg, trainset, validset, cfg_path, logger):
    """logger.info("Create preprocessed dataset.")
    trainset.create_prepared_data()
    validset.create_prepared_data()"""  ## TODO

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_config(cfg.path.exp_path, cfg_path)
    logger.info("Save config path: " + cfg.path.exp_path)

    sw = SummaryWriterAvg(log_dir=cfg.path.exp_path, flush_secs=10, dump_period=1)
    model = init_model(cfg).to(device)
    param = cfg.learn
    loss_criterion = get_loss_function(cfg.model.loss)
    optim = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=param.milestones, gamma=0.1)

    train_dataloader = DataLoader(
        trainset, param.batch_size, shuffle=True, drop_last=True, num_workers=8, collate_fn=batch_collate
    )
    valid_dataloader = DataLoader(validset, param.batch_size, drop_last=False, num_workers=8, collate_fn=batch_collate)
    logger.info("Training Start")

    tbar = tqdm(list(range(param.total_epoch)), total=param.total_epoch, ncols=100, desc="epoch")
    rate = None
    for epoch in tbar:
        sw.add_scalar(
            tag="learning_rate",
            value=optim.param_groups[0]["lr"],
            global_step=epoch,
        )
        it_tbar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100, desc="it")
        it_rate = None
        for it, batch_data in enumerate(it_tbar):
            global_step = epoch * int(len(train_dataloader)) + it

            # train
            model.train()
            with torch.set_grad_enabled(True):
                optim.zero_grad()
                input_matrix = batch_data["input_matrix"].to(device)
                output = model(input_matrix)
                label_matrix = batch_data["label_matrix"].to(device)
                loss = loss_criterion(output, label_matrix)
                loss = torch.mean(loss)
                sw.add_scalar(
                    tag="train_loss",
                    value=loss.item(),
                    global_step=global_step,
                )
                loss.backward()
                optim.step()
            if it in [0, 1000]:
                vis = vis_output(output, batch_data)
                sw.add_figure(tag=f"train_vis_{it}", figure=vis, global_step=epoch)
            it_rate = it_tbar.format_dict["rate"]
        logger.info("it_FPS: " + str(round(it_rate, 3)))

        # valid
        valid_loss = []
        label_list = []
        pred_list = []
        for it, batch_data in enumerate(tqdm(valid_dataloader, total=int(len(valid_dataloader)))):
            model.eval()
            with torch.no_grad():
                input_matrix = batch_data["input_matrix"].to(device)
                output = model(input_matrix)
                label_matrix = batch_data["label_matrix"].to(device)
                loss = loss_criterion(output, label_matrix)
                valid_loss.append(torch.mean(loss).item())
                pred, label = get_pred(output, batch_data)
                label_list.extend(label)
                pred_list.extend(pred)

            if it in [0, 1000]:
                vis = vis_output(output, batch_data)
                sw.add_figure(tag=f"val_vis_{it}", figure=vis, global_step=epoch)
        m_accuracy, m_jaccard, m_recall = evalutation(pred_list, label_list, validset.label_cfg, logger)
        sw.add_scalar(
            tag="valid/m_accuracy",
            value=m_accuracy,
            global_step=epoch,
        )
        sw.add_scalar(
            tag="valid/m_iou",
            value=m_jaccard,
            global_step=epoch,
        )
        sw.add_scalar(
            tag="valid/m_recall",
            value=m_recall,
            global_step=epoch,
        )
        sw.add_scalar(
            tag="valid/loss",
            value=sum(valid_loss) / len(valid_loss),
            global_step=epoch,
        )

        if epoch % param.save_interval == 0:
            save_model(model, os.path.join(cfg.path.exp_path, f"ckpts/{epoch}.pth"))
        lr_scheduler.step()
        rate = tbar.format_dict["rate"]
    logger.info("Training End")
    logger.info("epoch_FPS: " + str(round(rate, 3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="config file path")
    args = parser.parse_args()

    cfg = yaml_load(args.cfg)
    logger = create_logger(cfg.path.log_path, "train")

    try:
        trainset = SemanticKITTI(cfg, logger)
        validset = SemanticKITTI(cfg, logger, split="valid")

        train(cfg, trainset, validset, args.cfg, logger)
    except Exception as e:
        logger.exception(e)
