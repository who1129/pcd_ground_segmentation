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
