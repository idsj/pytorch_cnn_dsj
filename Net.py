import time
import torch
import os
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,       # input height
                out_channels=16,      # n_filters
                kernel_size=5,       # filter size
                stride=1,          # filter movement/step
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                padding=2,
            ),               # output shape (16, 28, 28)
            nn.ReLU(),           # activation
            # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(     # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),   # output shape (32, 14, 14)
            nn.ReLU(),           # activation
            nn.MaxPool2d(2),        # output shape (32, 7, 7)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        #img.sub_(0.5).div_(0.5)
        return img



