# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:13:44 2022

@author: noxtu
"""
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!




def find_gan_performance(act_img, gan_img):
    sol = 0
    for i in range(len(act_img)):
        temp = abs(act_img[i] - gan_img[i])
        sol+=temp
    return sol







# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):

    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.LeakyReLU(0.2,inplace=True),
        #### END CODE HERE ####
    )


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=520, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # There is a dropdown with hints if you need them! 
            #### START CODE HERE ####
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
            #### END CODE HERE ####
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen


def get_noise(n_samples, z_dim, device='cuda'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return torch.randn(n_samples, z_dim, device = device)
    #### END CODE HERE ####





def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        #### END CODE HERE ####
    )





class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=520, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(hidden_dim, 1)

            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc






criterion = nn.BCEWithLogitsLoss()
n_epochs = 1000
z_dim = 10
display_step = 500
batch_size = 128
lr = 0.00001
device = 'cuda'
# Load MNIST dataset as tensors









gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

if torch.cuda.is_available():
    gen = gen.cuda()
    disc = disc.cuda()







def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to 
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device=device)
    noise = noise.cuda()
    fake_images = gen(noise)
    fake_images.detach_()
    if torch.cuda.is_available():
        fake_images = fake_images.cuda()
    fake_loss = criterion(disc(fake_images),torch.zeros((num_images,1),device = device))
    real_loss = criterion(disc(real),torch.ones((num_images,1),device = device))
    disc_loss = ( fake_loss + real_loss ) / 2.0

    #### END CODE HERE ####
    return disc_loss











def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images. 
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    noises = get_noise(num_images,z_dim, device = device)
    noises = noises.cuda()
    fake_images = gen(noises)
    if torch.cuda.is_available():
        fake_images = fake_images.cuda()
    out = disc(fake_images)
    if torch.cuda.is_available():
        out = out.cuda()
    gen_loss = criterion(out, torch.ones(num_images, 1).to(device))
    
    #### END CODE HERE ####
    return gen_loss











def preprocess(x):

  # Map [0, 255] to [-1, 1].
  # images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
    x = ((x+110.0)/(110.0))
    return x




def rpreprocess(x):

# Map [0, 255] to [-1, 1].
  # images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
    x = ((x*110.0)-(110.0))
    for i in range(len(x)):
        if((abs(x[i]+110))<0.001):
            x[i] = -110.0
        if(x[i]<=-104.1):
            x[i]=-110.0
    return x






from torch.utils.data import DataLoader, Dataset


import numpy as np
import pandas as pd
df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/gan+/b0f0_train.csv",header=0)
training_data_csv_in = np.array(df.iloc[:,:520])
training_data_csv_lb = np.array(df.iloc[:,520:522])

# training_data_csv_in = np.loadtxt('out_im_dirichlet.csv', delimiter=',', dtype=np.float)
# training_data_csv_lb = np.loadtxt('out_lb_dirichlet.csv', delimiter=',', dtype=np.float)

if torch.cuda.is_available():
    training_data_csv_in = torch.from_numpy(training_data_csv_in).cuda()
    training_data_csv_lb = torch.from_numpy(training_data_csv_lb).cuda()

store={}
for i in range(len(training_data_csv_in)): 
    k = training_data_csv_lb[i].tolist()
    v = training_data_csv_in[i].tolist()

    if tuple(k) in store:

        store[tuple(k)].append(v)
    else:
        store[tuple(k)] = [v]



import csv
import sys
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False

import numpy
from torch.utils.data import DataLoader, Dataset
# from utils import Logger
count = 1
for k in store.keys():
    
    val = store[k]
    store_in = []
    for v in val:
        store_in.append(numpy.asarray(v))
    tensor_store_val = torch.Tensor(store_in).cuda()

  ###
    dataloader = DataLoader(tensor_store_val, batch_size=4)
  # dataloader = DataLoader(tensor_store_val, batch_size=len(val))
    save_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/model_state/UJI/gan+_model_state/b0f0/"+str(k[0])+"_"+str(k[1])+".pt"
    print("Start training: {}/{}".format(count,len(store.keys())))
    for epoch in range(n_epochs):
    
      # Dataloader returns the batches
        for real in dataloader:

            if torch.cuda.is_available():
                real = real.cuda()

            cur_batch_size = len(real)

            if cur_batch_size != 4:
                continue
          
          # Flatten the batch of real images from the dataset
            real_b = real.view(cur_batch_size, -1).to(device)
            if torch.cuda.is_available():
                real_b = real_b.cuda()

            real = preprocess(real_b)

            if torch.cuda.is_available():
                real = real.cuda()
          ### Update discriminator ###
          # Zero out the gradients before backpropagation
            disc_opt.zero_grad()

          # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

          # Update gradients
            disc_loss.backward(retain_graph=True)

          # Update optimizer
            disc_opt.step()

          # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

          ### Update generator ###
          #     Hint: This code will look a lot like the discriminator updates!
          #     These are the steps you will need to complete:
          #       1) Zero out the gradients.
          #       2) Calculate the generator loss, assigning it to gen_loss.
          #       3) Backprop through the generator: update the gradients and optimizer.
          #### START CODE HERE ####
          
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()
          
          #### END CODE HERE ####

          # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                try:

                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")
                    
    torch.save(gen.state_dict(), save_state)
    count+=1
    
  
  
# =============================================================================
#     for i in range(100):
#         fake_noise = get_noise(4, z_dim, device=device)
#         fake = gen(fake_noise)
#         for j in range(4):
#             temp = fake[j]
#             for q in range(len(temp)):
#                 if(temp[q]<0.001):
#                     temp[q]=0
#             temp = rpreprocess(temp)
#             mindiff = find_gan_performance(real_b[0], temp) 
#             for index in range(1, len(real_b)):
#                 curdiff = find_gan_performance(real_b[index], temp)
#                 if curdiff < mindiff:
#                     mindiff = curdiff
#             if(mindiff<=265.0):
#                 with open("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/out_im_dir_plus_gan.csv", "a") as f:
#                     temp = temp.cpu().detach().numpy().tolist()
#                     writer = csv.writer(f)
#                     writer.writerow(map(lambda x: x, temp))
# 
#                 with open("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/out_lb_dir_plus_gan.csv", "a") as f:
#                     k = list(k)
#                     writer = csv.writer(f)
#                     writer.writerow(map(lambda x: x, k))
# =============================================================================






  

