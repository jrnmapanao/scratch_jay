#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:45:20 2024

@author: jrnmapanao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, n_mels=256, emb_sz=128):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.flattened_size = None
        self.fc = None
        self.emb_sz = emb_sz

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Calculate the flattened size dynamically
        if self.fc is None:
            self.flattened_size = x.view(x.size(0), -1).size(1)
            self.fc = nn.Linear(self.flattened_size, self.emb_sz).cuda()  # Define the fully connected layer
        
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

    
    
# Define the NT-Xent loss function
class NTXentLoss(nn.Module):
    def __init__(self, tau=0.05):
        super(NTXentLoss, self).__init__()
        self.tau = tau

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        similarity_matrix = torch.mm(z_i, z_j.T) / self.tau
        labels = torch.arange(batch_size).cuda()
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss