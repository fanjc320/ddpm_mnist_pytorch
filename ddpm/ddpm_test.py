#!/usr/bin/env python
# coding: utf-8

# refer to the slides for an intuitive math explanation of diffusion models
# https://www.figma.com/file/qGZTQDS2nBG9797921eM48/diffusion?node-id=0%3A1&t=aaps3X8j0WiVDyNk-1

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# In[2]:


torch.manual_seed(0)

# In[18]:


# global variables

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000  # number of steps the diffusion model takes
BATCH_SIZE = 32  # for the data loader

betas = torch.linspace(0.0001, 0.02, T)  # variance scheduler, values taken from official implementation
alphas = 1 - betas
alpha_bars = [alphas[0]]
for alpha in alphas[1:]: alpha_bars.append(alpha * alpha_bars[-1])


# check if alpha bars is correct
# print(betas, alphas, alpha_bars)
# checked


# In[10]:


# convenience function to show
def show_tensor(tensor):
    plt.imshow(tensor[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.show()


# In[11]:


print("device:",device)

# In[19]:


dataset = datasets.MNIST('mnist_data', download=True, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),  # convert image to PyTorch tensor
                             transforms.Lambda(lambda t: 2 * t - 1)
                         ]))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# In[7]:

# images, label = next(dataloader)
# images, label = next(enumerate(dataloader))
data = iter(dataloader)
images, label = next(data)
print(f"input image is of shape {images.shape} len:{len(images)} label:{label}")
# plt.imshow(images[0, 0], cmap='gray')
# plt.imshow(images[1, 0], cmap='gray')
# plt.show()




# In[8]:


# operates on a batch of images
# introduces noise into the input image
# generates x_t from x_0 in literature vocab
# def forward_pass(x_0, eta, t):
#     alpha_bar_t = torch.tensor([alpha_bars[t_i] for t_i in t])[:, None, None, None]
#     return torch.sqrt(alpha_bar_t) * x_0 + (1 - alpha_bar_t) * eta


# check the degradation for different t values
# checked
# x = images[33:40]
x = images[30:32]
eta = torch.randn_like(x)
lenx = len(x)
t = torch.randint(0, T, (len(x),)).long()
# t[0] = 9
print(f"x x.shape:{x.shape} lenx:{len(x)} t:{t}") # x x.shape:torch.Size([2, 1, 28, 28]) lenx:2 t:tensor([433, 129])

# degraded_images = forward_pass(x, eta, t)
# print(len(degraded_images))
# show_tensor(degraded_images)
def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 7, padding=3),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class SimpleNet(Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            block(1, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        return self.conv(x)

print("end----------------")
# In[57]:

# device = "cpu"
batch_size = 100
# model = UNet().to(device)
model = SimpleNet().to(device)
model.load_state_dict(torch.load("simple.pt"))

@torch.no_grad()  # since inference and not training
def generate_image():
    # start with a noisy image sample from N(0,I)
    x_t = torch.randn((1, 1, 28, 28)).to(device)

    model.eval()

    frames = []

    # iterative sampling to go from x_t to x_0
    for t in reversed(range(T)):
        z = torch.randn_like(x_t) if t > 0 else 0
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        factor = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        var_t = torch.sqrt(betas[t])  # another complex option available
        t = torch.tensor([t]).to(device)
        x_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (x_t - factor * model(x_t, t)) + var_t * z
        x_t = x_t_minus_1

        tmp = x_t_minus_1.cpu().detach().numpy()
        # print(f"tmp.shape{tmp.shape}") # tmp.shape(1, 1, 28, 28)
        frames.append(tmp)

    return x_t_minus_1, frames  # is actually x_0 after the last iteration


for i in range(1):
    op, frames = generate_image()

    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 1
    for i in range(1, columns * rows + 1):
        j = [0, 250, 500, 750, -1][i - 1]
        print(f"j:{j}") # 0-250-500-750--1
        img = frames[j][0, 0]# frames;[1000][1,1,28,28]
        print(f"frames.len:{len(frames)} img.shape:{img.shape}") # frames.shape:1000 img.shape:(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()



