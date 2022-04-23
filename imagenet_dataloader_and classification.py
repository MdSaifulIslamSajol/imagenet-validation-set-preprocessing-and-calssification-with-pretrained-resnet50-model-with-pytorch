#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:59:40 2022

@author: saiful
"""

import os
import torch
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.parallel
import torch.utils.data
from torchvision import models

root_path=os.getcwd()

path_train= "/data/imagenet_datasets/ILSVRC/Data/CLS-LOC/"
path_val= "/data/imagenet_datasets/"

transform =transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


imagenet_train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_train, "train"),                                           
                                                transform=transform)

imagenet_test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path_val, "val_subfolders3"),                                           
                                                 transform=transform)

print("imagenet_train_dataset length :", len(imagenet_train_dataset))
print("imagenet_test_dataset length :", len(imagenet_test_dataset))

data_loader_train = torch.utils.data.DataLoader(
                                            dataset=imagenet_train_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=0)

data_loader_test = torch.utils.data.DataLoader(
                                            dataset=imagenet_test_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=0)

net = models.resnet50(pretrained=True)  #models.densenet161()
net=net.cuda(2)
for param in net.parameters():
    param.requires_grad = False

batch_loss = 0
total_t=0
correct_t=0
val_acc = []
with torch.no_grad():
    net.eval()
    print("prediction has been started")
    for data_t, target_t in (data_loader_test):
        data_t, target_t = data_t.cuda(2), target_t.cuda(2)
        outputs_t = net(data_t)
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_t += torch.sum(pred_t==target_t).item()
        total_t += target_t.size(0)

    print(f' validation acc: {(100 * correct_t/total_t):.4f}\n')

net.train()

    


