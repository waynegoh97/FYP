# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:11:07 2022

@author: noxtu
"""

from torchvision import models
import torch.nn as nn
import torch 
import torch.optim as optim
import shutil
import random
import time
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import csv  

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t in tensor:
            t.mul_(self.std).add_(self.mean)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
class ImageDataset(Dataset):
    def __init__(self,img_path, label, transform):
        self.transform=transform
        #self.img_folder=img_folder
        self.img_path = img_path
        self.label = label

#The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.label)
 
    def __getitem__(self,index):
     
        image=Image.open(self.img_path[index],'r')
        image=self.transform(image)
        targets=self.label[index]

        return image, targets
    
def label_directory(img_path):
    '''
Purpose: Get all labels directory in a list (e.g. C://path/images/b0f0/[latitude_longitude])
    Parameters
    ----------
    img_path : TYPE
        DESCRIPTION.

    Returns
    -------
    label_dir : list
        labels directory

    '''
    label_dir = []
    for root, dirs, files in os.walk(img_path, topdown=False):
        for d in dirs:
            label_dir.append(os.path.join(root, d))
    return label_dir

def data_and_label(img_folder):
    '''
    Purpose: maps image path, image name, and labels together in list order
    Parameters
    ----------
    img_folder : string
        directory of images (e.g. ./cnn_images/b0f0_train_with_labels)

    Returns
    -------
    img_path : list
        contains image paths of all images in the folder
    label : array
        contains array of all labels

    '''
    img_path = []
    label = []
    img_name = []
    for root, dirs, files in os.walk(img_folder, topdown=False):
        for name in files:
            temp = os.path.join(root, name)
            temp = os.path.normpath(temp)
            path = temp.split(os.sep)
            path = path[-2].split("_")
            path[0] = float(path[0])
            path[1] = float(path[1])
            label.append(path)
            img_path.append(temp)
            img_name.append(name)
    label = np.array(label)
    return img_path, label, img_name

def norm_image(path, label):
    '''

    Parameters
    ----------       
    path : string
        directory of each image
    label : array
        scaled labels

    Returns
    -------
    mean : float
        mean value of all the images
    std : TYPE
        standard deviation of all the images

    '''
    transform = transforms.Compose([transforms.ToTensor()])
    data_set = ImageDataset(path, label, transform)
    loader = DataLoader(data_set, batch_size=len(data_set))
    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()
    return mean, std

def load_image(img_path, label, batch_size, mean, std, shuffle):
    '''
    Parameters
    ----------
    img_path : string
        directory of dataset used
    batch_size : int
        batch size
    mean : float
        mean of overall input data
    std : float
        standard deviation of overall input data
    shuffle : boolean
        shuffle dataloader or not

    Returns
    -------
    dataloader

    '''

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor(),transforms.Normalize((mean),(std))])
    # , transforms.Normalize((mean),(std))
    data_set=ImageDataset(img_path, label,transform)
    dataloader = DataLoader(data_set,batch_size=batch_size,shuffle=shuffle, drop_last = True)
    
    return dataloader

def load_by_label(label_dir, batch_size, shuffle=True):
    '''
Purpose: Get dataloader by individual label
    Parameters
    ----------
    label_dir : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.

    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    '''
    img_path, label, img_name = data_and_label(label_dir)
    mean, std = norm_image(img_path, label)
    dataloader = load_image(img_path, label, batch_size, mean, std, shuffle)
    return dataloader, mean, std, img_name, label

# For 23x23 images
class UJI_Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(UJI_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 23 x 23
            nn.Conv2d(
                channels_img, features_d*2, kernel_size=3, stride=2, padding=1
            ), #64x12x12
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d*2, features_d * 4, 4, 2, 1), #128x6x6
            self._block(features_d * 4, features_d * 8, 4, 2, 1), #256x3x3
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0), # 1x1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
# Generator: noise > decreasing node + batchnorm + relu > feature critic layer (64) 
class UJI_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(UJI_Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 8, 3, 1, 0),  # img: 3x3
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 6x6
            self._block(features_g * 4, features_g*2 , 4, 2, 1),  # img: 12x12
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=3, stride=2, padding=1
            ),
            # Output: N x channels_img x 23x23
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    
def initialize_weights(model):
    # Initializes weights for wgan-gp
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def wgan_gp_pretrain(save_state, gen_model_state, disc_model_state, data_dir, 
                     lr, batch, num_epoch, img_dim, img_channel, latent, critic_layer, gen_layer, critic_iter, gradient_p):
    '''
    Purpose: Training of WGAN-GP model

    Parameters
    ----------
    model_state_dir : string
        model state name to be saved as
    data_dir : string
        folder name with the training input images
    dataset : string
        define the type of dataset used (e.g. uji or ng)
    lr : float
        learning rate
    batch : int
        batch size
    num_epoch : int
        number of epochs
    img_dim : int
        image size
    img_channel : int
        number of channels (e.g. 1 for grayscale, 3 for rgb)
    latent : int
        latent noise
    critic_layer : int
        determine number of discriminator layer
    gen_layer : int
        determine number of generator layer
    critic_iter : int
        determine number of critic iterations
    gradient_p : int
        gradient penalty

    Returns
    -------
    None.

    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader, mean, std, _, _= load_by_label(data_dir, batch)
    unorm = UnNormalize(mean = mean, std = std)
        
    gen = UJI_Generator(latent, img_channel, gen_layer).to(device)
    critic = UJI_Discriminator(img_channel, critic_layer).to(device)
    gen.load_state_dict(torch.load(gen_model_state))
    critic.load_state_dict(torch.load(disc_model_state))
    
        
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0,0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0,0.9))
    gen.train()
    critic.train()
    critic_loss = []

    for epoch in range(num_epoch):
        # Target labels not needed
        gen.train()
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            cur_batch_size = data.shape[0]
    
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(critic_iter):
                noise = torch.randn(cur_batch_size, latent, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, data, fake, device=device)
                loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + gradient_p * gp)
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
    
    
            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
         
  
            
        # print(
        #     "[Epoch: %d/%d] [Batch: %d/%d] [G loss: %f] [C loss: %f]"
        #     % (epoch+1, num_epoch, batch_idx+1, len(dataloader), loss_gen.detach(), loss_critic.detach())
        # )
        critic_loss.append(-loss_critic.detach())
                        
    torch.save(gen.state_dict(), save_state)


def wgan_gp_train(gen_model_state, data_dir, lr, batch, num_epoch, img_channel, latent,
                  critic_layer, gen_layer, critic_iter, gradient_p):
    '''
    Purpose: Training of WGAN-GP model

    Parameters
    ----------
    model_state_dir : string
        model state name to be saved as
    data_dir : string
        folder name with the training input images
    dataset : string
        define the type of dataset used (e.g. uji or ng)
    lr : float
        learning rate
    batch : int
        batch size
    num_epoch : int
        number of epochs
    img_dim : int
        image size
    img_channel : int
        number of channels (e.g. 1 for grayscale, 3 for rgb)
    latent : int
        latent noise
    critic_layer : int
        determine number of discriminator layer
    gen_layer : int
        determine number of generator layer
    critic_iter : int
        determine number of critic iterations
    gradient_p : int
        gradient penalty

    Returns
    -------
    None.

    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_dir = label_directory(data_dir)
    dataloader, mean, std, _, _ = load_by_label(label_dir[0], batch)
    unorm = UnNormalize(mean=mean, std=std)
    # =============================================================================
    #     for i in range(len(label_dir)):
    #         dataloader = load_by_label(label_dir[i], batch)
    # =============================================================================
    gen = UJI_Generator(latent, img_channel, gen_layer).to(device)
    critic = UJI_Discriminator(img_channel, critic_layer).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    gen.train()
    critic.train()
    critic_loss = []

    for epoch in range(num_epoch):
        # Target labels not needed
        gen.train()
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            cur_batch_size = data.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(critic_iter):
                noise = torch.randn(cur_batch_size, latent, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, data, fake, device=device)
                loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + gradient_p * gp)
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        critic_loss.append(-loss_critic)

    torch.save(gen.state_dict(), gen_model_state)


if __name__ == "__main__":
    LEARNING_RATE =0.001 #0.001 (mnist)
    BATCH_SIZE = 4 #32 (mnist), 8 for wgan-gp uji
    IMAGE_SIZE = 23
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 1000
    FEATURES_CRITIC = 64
    FEATURES_GEN = 64
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    my_dpi = 96 # Can be found using this link https://www.infobyip.com/detectmonitordpi.php
    
    gen_saved_state = '/home/SEANGLIDET/uji/FYP_data/model_states/gen_most_sample.pt'
    disc_saved_state = '/home/SEANGLIDET/uji/FYP_data/model_states/disc_most_sample.pt'
    
    unique_loc = "b0f0" 
    
    data_dir = "/home/SEANGLIDET/uji/FYP_data/images/uji/ori_dirich/"+unique_loc
    saved_state = "/home/SEANGLIDET/uji/FYP_data/model_states/UJI_without_TL/"+unique_loc
    
    if not os.path.exists(saved_state):
        os.makedirs(saved_state)
    NUM_EPOCHS = 500
    label_dir = label_directory(data_dir)
    
    time_keeper = []
    for i in range(0,20):
        start = time.perf_counter()
        curr_label = label_dir[i].split('/') #for linux 
        save_state = saved_state + '/'+str(curr_label[-1])+'.pt'
        print("Starting ", curr_label[-1])
        wgan_gp_train(save_state, data_dir, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, CHANNELS_IMG, Z_DIM, FEATURES_CRITIC, FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)
        # wgan_gp_pretrain(save_state, gen_saved_state, disc_saved_state, label_dir[i], LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, IMAGE_SIZE,
        #                   CHANNELS_IMG, Z_DIM, FEATURES_CRITIC, FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)
        end = time.perf_counter()
        print("Appending data: ", [curr_label[-1], end-start])
         

        with open(r'{}.csv'.format(unique_loc), 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([curr_label[-1], end-start])

        
  
