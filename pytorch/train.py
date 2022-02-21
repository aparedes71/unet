import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from unet import unet


means = [0.485, 0.456, 0.406]  # mean of the imagenet dataset for normalizing

stds = [0.229, 0.224, 0.225] # standard deviance of the imagenet dataset for normalizing

def deNormalize(img,means=means,stds=stds):
    red = img[...,0]*stds[0] + means[0]
    green = img[...,1]*stds[1] + means[1]
    blue = img[...,2]*stds[2] + means[2]

    denormed = np.stack([red,green,blue],axis=2)
    return denormed



transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)), transforms.Normalize(means,stds) ])
transform_target = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)) ])

train_data = datasets.VOCSegmentation(root='./Data', image_set='train', download=True, transform=transform, target_transform=transform_target)
val_data = datasets.VOCSegmentation(root='./Data', image_set='val', download=True, transform=transform, target_transform=transform_target)

torch.manual_seed(42)
model = unet()
print(model)
print(train_data)
print(val_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)


#simply used to visualize a few examples
for b,(img,label) in enumerate(train_loader):
    break
plt.figure()
plt.subplot(2,2,1)
plt.imshow(deNormalize(np.transpose(img[0,...],(1,2,0))))
plt.subplot(2,2,2)
plt.imshow(np.transpose(label[0,...],(1,2,0)))
plt.subplot(2,2,3)
plt.imshow(deNormalize(np.transpose(img[1,...],(1,2,0))))
plt.subplot(2,2,4)
plt.imshow(np.transpose(label[1,...],(1,2,0)))
plt.show()
