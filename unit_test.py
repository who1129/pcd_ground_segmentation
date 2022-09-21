## dataset
from src.utils.utils import yaml_load, create_logger
from src.data.SemanticKITTI import SemanticKITTI

cfg = yaml_load("config.yaml")
logger = create_logger("../..", "tmp.log")

dataset = SemanticKITTI(cfg, logger)

idx = 0
data = dataset.__getitem__(idx)
print(data["pcd"].shape)
print(data["label"].shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.axes().set_aspect("equal")
plt.subplot(121)
plt.imshow(data["label_matrix"][:, :, 0])
plt.title("label matrix - non-ground")

plt.subplot(122)
plt.imshow(data["label_matrix"][:, :, 1])
plt.colorbar()
plt.title("label matrix - ground")
plt.show()

plt.figure(figsize=(15, 5))
plt.axes().set_aspect("equal")
plt.subplot(131)
plt.imshow(data["input_matrix"][:, :, 0])
plt.title("input matrix - channel 0")

plt.subplot(132)
plt.imshow(data["input_matrix"][:, :, 1])
plt.title("input matrix - channel 1")
plt.subplot(133)
plt.imshow(data["input_matrix"][:, :, 2])
plt.colorbar()
plt.title("input matrix - channel 2")

plt.show()

pred = dataset.decoding_pointcloud(
    data["label_matrix"], data["grid_mask"], data["pcd"], data["bin_idx"]
)
plt.figure(figsize=(20, 10))
plt.axes().set_aspect("equal")

plt.subplot(131)
plt.scatter(data["pcd"][:, 0], data["pcd"][:, 1], c=data["label"], s=1)
plt.subplot(132)
import numpy as np

ground_label = [9, 11, 12, 17]
label = data["label"]
label_ = np.zeros_like(label)
for l in ground_label:
    label_[label == l] = 1
plt.scatter(
    data["pcd"][:, 0],
    data["pcd"][:, 1],
    c=data["grid_mask"],
    s=1,
    cmap="prism",
    alpha=0.5,
)

plt.subplot(133)
plt.scatter(
    data["pcd"][:, 0],
    data["pcd"][:, 1],
    c=label_ == pred,
    s=1,
    cmap="prism",
    alpha=0.5,
)

plt.show()
