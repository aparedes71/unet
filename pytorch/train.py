import time
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
    red = img[...,0] * stds[0] + means[0]
    green = img[...,1] * stds[1] + means[1]
    blue = img[...,2] * stds[2] + means[2]

    denormed = np.stack([red,green,blue],axis=2)
    return denormed

# Generate dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)), transforms.Normalize(means,stds) ])
transform_target = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)) ])

train_data = datasets.VOCSegmentation(root='./Data', image_set='train', download=True, transform=transform, target_transform=transform_target)
val_data = datasets.VOCSegmentation(root='./Data', image_set='val', download=True, transform=transform, target_transform=transform_target)

print(train_data)
print(val_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=100, shuffle=False)


#Train step

def train_step(train_loader,model,criterion,opt,train_losses,train_corr):
    model.train()
    for b,(img,label) in enumerate(train_loader):
        b += 1
        opt.zero_grad()
        y = model(img)
        print(y.shape)
        print(torch.squeeze(label,dim=1).shape)
        loss = criterion(y,torch.squeeze(label,dim=1))
        loss.backward()
        opt.step()

        train_losses.append(loss)
        if b%10 == 0:
            print(f"Batch : {b} , Train Loss : {loss} Train Acc ")


if __name__ == "__main__":
    #simply used to visualize a few examples
    # for b,(img,label) in enumerate(train_loader):
    #     break
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(deNormalize(np.transpose(img[0,...],(1,2,0))))
    # plt.subplot(2,2,2)
    # plt.imshow(np.transpose(label[0,...],(1,2,0)))
    # plt.subplot(2,2,3)
    # plt.imshow(deNormalize(np.transpose(img[1,...],(1,2,0))))
    # plt.subplot(2,2,4)
    # plt.imshow(np.transpose(label[1,...],(1,2,0)))
    # plt.show()

    # print(label.shape)
    # print(label[0,0,...])

    model = unet(3,20)

    train_losses = []
    train_corr = []

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr = 0.001)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        train_step(train_loader,model,criterion,opt,train_losses,train_corr)

    print(train_losses)