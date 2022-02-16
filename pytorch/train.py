import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from unet import unet


transform = transforms.ToTensor()

train_data = datasets.VOCSegmentation(root='./Data', image_set='train', download=True, transforms=transform)
val_data = datasets.VOCSegmentation(root='./Data', image_set='val', download=True, transforms=transform)

torch.manual_seed(42)
model = unet()
print(model)
print(train_data)
print(val_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)