import torch
import torch.nn as nn

import time
import matplotlib
import yaml

import yaml

with open("config.yaml") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print(config)

