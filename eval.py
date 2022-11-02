import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from src.data.SemanticKITTI import SemanticKITTI
from src.data.dataset import decoding_pointcloud
from src.model.model import GroundNet
from src.utils.torch_ioueval import iouEval
from src.utils.utils import (
    yaml_load,
    create_logger,
    batch_collate,
)


def load_model(path):
    model = GroundNet()
    model.load_state_dict(torch.load(path))
    return model


def get_pred(output, batch_data):
    output_ = output.detach().to("cpu").numpy()
    grid_mask_ = batch_data["grid_mask"].detach().numpy()
    label_ = batch_data["label"].detach().numpy()
    point_cnt_ = batch_data["point_cnt"].detach().numpy()
    pcd_ = batch_data["pcd"].detach().numpy()
    label = []
    pred = []
    pcds = []
    for i in range(label_.shape[0]):
        cnt = int(point_cnt_[i])
        pred.append(decoding_pointcloud(output_[i], grid_mask_[i])[:cnt])
        label.append(label_[i][:cnt])
        pcds.append(pcd_[i][:cnt])

    return pred, label, pcds


def evalutation(preds, labels, DATA):
    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = max(class_remap.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())

    ##ignore = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 18, 19]
    ignore = []
    print("Ignoring xentropy class ", ignore, " in IoU evaluation")

    evaluator = iouEval(nr_classes, ignore)
    evaluator.reset()

    progress = 10
    count = 0
    print("Evaluating sequences: ", end="", flush=True)
    # open each file, get the tensor, and make the iou comparison
    for label, pred in zip(labels, preds):
        count += 1
        # pred ground label to original label
        ground_label = [9, 11, 12, 17]
        tmp = np.zeros_like(pred)
        tmp[np.isin(label, ground_label)] = 1

        # accuracy with label
        ## pred = np.where(pred == 1, label, 0)
        ## pred = np.where(np.logical_and(pred == 0, tmp != 0), label, pred)

        # only ground accuracy
        pred = np.where(np.logical_and(pred == 1, tmp), label, 0)
        pred = np.where(np.logical_and(pred == 0, tmp == 0), label, pred)

        evaluator.addBatch(pred.astype(np.int64), label.astype(np.int64))

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    m_recall = evaluator.getrecall()

    print(
        "Validation set:\n"
        "Acc avg {m_accuracy:.3f}\n"
        "IoU avg {m_jaccard:.3f}\n"
        "Recall avg {m_recall:.3f}".format(m_accuracy=m_accuracy, m_jaccard=m_jaccard, m_recall=m_recall)
    )
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print(
                "IoU class {i:} [{class_str:}] = {jacc:.3f}".format(
                    i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc
                )
            )
    return m_accuracy, m_jaccard, m_recall


def vis_output(output, batch_data, select_idx=0, ground_label=[9, 11, 12, 17], it=0, path="eval_vis"):
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
    v = np.linspace(0, 18, 19, endpoint=True)
    plt.colorbar(ticks=v)
    plt.title("original label")

    plt.subplot(2, 2, 2)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=ground_, s=1)
    v = np.linspace(0, 1, 2, endpoint=True)
    plt.colorbar(ticks=v)
    plt.title("ground/non-ground label")

    plt.subplot(2, 2, 3)
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=decoded_output, s=1)
    v = np.linspace(0, 1, 2, endpoint=True)
    plt.colorbar(ticks=v)
    plt.title("predict")

    plt.subplot(2, 2, 4)
    color = np.zeros_like(ground_)
    tp = ground_ == decoded_output
    color[tp] = 0
    fp = (ground_ == 0) & (decoded_output == 1)
    color[fp] = 1
    fn = (ground_ == 1) & (decoded_output == 0)
    color[fn] = 2
    plt.scatter(pcd_[:, 0], pcd_[:, 1], c=color, s=1, cmap="jet")
    v = np.linspace(0, 2, 3, endpoint=True)
    plt.colorbar(ticks=v)
    plt.title("error\n TP=0, FP=1, FN=2")
    os.makedirs(path, exist_ok=True)
    plt.savefig(
        f"{path}/{it}.png",
        dpi="figure",
        format=None,
        metadata=None,
        bbox_inches=None,
        pad_inches=0.1,
        facecolor="auto",
        edgecolor="auto",
        backend=None,
    )

    return fig


def eval(cfg, validset, ckpt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path).to(device)
    param = cfg.learn
    valid_dataloader = DataLoader(validset, param.batch_size, drop_last=False, num_workers=8, collate_fn=batch_collate)

    label_list = []
    pred_list = []
    pcd_list = []

    for it, batch_data in enumerate(tqdm(valid_dataloader, total=int(len(valid_dataloader)))):
        model.eval()
        with torch.no_grad():
            input_matrix = batch_data["input_matrix"].to(device)
            output = model(input_matrix)
            pred, label, pcd = get_pred(output, batch_data)
            label_list.extend(label)
            pred_list.extend(pred)
            pcd_list.extend(pcd)
            if it % cfg.eval.vis_interval == 0:
                _ = vis_output(output, batch_data, it=it, path=cfg.eval.vis_path)
                np.save(f"output/{it}_ouptut", batch_data["grid_mask"].detach().numpy())
                np.save(f"output/{it}_pred", pred)
                np.save(f"output/{it}_label", label)
                np.save(f"output/{it}_pcd", pcd)

    m_accuracy, m_jaccard, m_recall = evalutation(pred_list, label_list, validset.label_cfg, pcd_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="config file path")
    parser.add_argument("--ckpt", required=True, help="model ckpt path")
    args = parser.parse_args()

    cfg = yaml_load(args.cfg)
    logger = create_logger("../..", "tmp.log")  ## TODO

    validset = SemanticKITTI(cfg, logger, split="valid")

    eval(cfg, validset, args.ckpt)
