import torch
import argparse
import numpy as np
from tqdm import tqdm
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


def evalutation(preds, labels, DATA, pcds):
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
    for label, pred, pcd in zip(labels, preds, pcds):
        count += 1
        # pred ground label to original label
        ground_label = [9, 11, 12, 17]
        tmp = np.zeros_like(pred)
        tmp[np.isin(label, ground_label)] = 1

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
