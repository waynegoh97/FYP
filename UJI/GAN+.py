# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:13:44 2022

@author: noxtu
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import math
import os
import torchvision
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!


def imsave(imgs, save_dir, px, my_dpi):
    #save images according to defined pixel
    imgs = torchvision.utils.make_grid(imgs, normalize=False)
    npimgs = imgs.numpy()
    fig = plt.figure(figsize = (px/my_dpi,px/my_dpi), dpi=my_dpi, frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.imsave('C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/test.png',np.transpose(npimgs, (1,2,0)), cmap='gray')
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap = "Greys_r")
    fig.savefig(save_dir, dpi=my_dpi)
    plt.close(fig)
    
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



def dirichlet_generate(csv_dir, fid, save_dir):
    '''
    Purpose: Generate dirichlet augmentation (where if RP contains < 75 images, dirich will generate till RP contains 75 images)

    Parameters
    ----------
    csv_dir : string
        directory of stored csv files
    fid : string
        e.g. "b0f0" for uji, "floor-1" for ng
    save_dir : string
        directory of dirichlet csv to be saved at

    Returns
    -------
    None.

    '''
    #store dictionary contains coordinates (latitude, longitude) as keys and rssi data as values (according to the number of samples belonging to coordinate) e.g. {(-7345.345, 4328596): [[AP0 ... AP520],[AP0 ... AP520]]}
    store = {}

    train_df = pd.read_csv(csv_dir+fid+"_train.csv",header=0)
    train_input = np.array(train_df.iloc[:,:520])
    train_label = np.array(train_df.iloc[:,520:522])
    for j in range(len(train_input)): 
        k = train_label[j].tolist()
        v = train_input[j].tolist()
    
        if tuple(k) in store:
            store[tuple(k)].append(v)
        else:
            store[tuple(k)] = [v]

    ### Total dirichlet needed = 75 - original images, another 75 images is generated by GAN+, with a total of 150 images for localisation ###
    tempim = [] # contains [[[AP0...AP520] for N samples], [[AP0...AP520] for N samples]]]
    templb = [] # contains [[longitude,latitude],[longitude,latitude]]
    dirich_needed = []
    drop_key =[]
    count=0
    ### np.random.dirichlet takes N samples weights, where N is the total number of samples in RP. 
    ## N Dirichlet distribution adds up to 1. 
    
    for k in store.keys(): #number of unique RP
        #Just need 75 samples 
        if (len(store[k]) <= 75) & (len(store[k]) != 1):
            sample_size = len(store[k])
            dirich_needed.append(75 - sample_size)
         
            #randomly pick samples for dirichlet
            im_new  = [[0 for i in range(sample_size)] for j in range(520)]
            curr_rp_samples = np.array(store[k])
            curr_rp_samples[curr_rp_samples==100] = -110
            curr_rp_samples = curr_rp_samples.T
            #generate images according to dirichlet needed (contains excess)
            for gen in range(math.ceil(dirich_needed[count]/sample_size)):
                for s in range(sample_size):
                    weight = np.random.dirichlet(np.ones(sample_size),size=1)
                    for ap in range(520):
                        im_new[ap][s] = np.sum(np.multiply(weight,curr_rp_samples[ap][:]))
                        if(110+im_new[ap][s]<=0.000001):
                            im_new[ap][s] = -110.0
                
                tempim.append((np.array(im_new).T).tolist())
                templb.append([k[0],k[1]])
            count+=1
        else:
            drop_key.append(k)
    
    if drop_key:
        for k in range(len(drop_key)):
            del store[drop_key[k]]
    
    #Create column names
    col = ["AP"+str(i) for i in range(1,521)]
    df = pd.DataFrame(tempim[0], columns= col)
    df["LATITUDE"] = templb[0][1]
    df["LONGITUDE"] = templb[0][0]
    for i in range(1,len(tempim)):
        df2 = pd.DataFrame(tempim[i], columns= col)
        df2["LATITUDE"] = templb[i][1]
        df2["LONGITUDE"] = templb[i][0]
        df=df.append(df2,ignore_index=True)
        
    #remove excess dirichlet generated
    col.append("LATITUDE")
    col.append("LONGITUDE")
    new_df = pd.DataFrame(columns = col)
    count=0
    for key in store.keys():
        tdf = df[(df["LATITUDE"] == key[1]) & (df["LONGITUDE"] == key[0])]
        tdf = tdf.drop(tdf.index[range(len(tdf)-dirich_needed[count])])
        new_df = new_df.append(tdf,ignore_index=True)
        count+=1
        
    new_df.to_csv(save_dir + fid+".csv",index=False)


def train_most_sample():
    df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv",header=0)
    df = df.replace(100,-110)
    df = df[(df["BUILDINGID"] == 1) & (df["FLOOR"] == 0)]
    training_data_csv_in = np.array(df.iloc[:,:520])
    training_data_csv_lb = np.array(df.iloc[:,520:522])
    
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
    gen_loss = False
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 1000
    z_dim = 10
    lr = 0.00001
    device = 'cuda'
    # Load MNIST dataset as tensor

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    if torch.cuda.is_available():
        gen = gen.cuda()
        disc = disc.cuda()

        
    val = store[(-7445.557873841375,4864826.589670561)]
    store_in = []
    for v in val:
        store_in.append(np.asarray(v))
    tensor_store_val = torch.Tensor(store_in).cuda()
  
###
    dataloader = DataLoader(tensor_store_val, batch_size=4)
    
    gen_save_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/gen_gan+_most_sample.pt"
    disc_save_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/disc_gan+_most_sample.pt"
    
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
        
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()
  
    torch.save(gen.state_dict(), gen_save_state)
    torch.save(disc.state_dict(), disc_save_state)

def gan_pretrained(train_csv_dir, dirich_csv_dir, fid, bid, gen_state, disc_state, z_dim, lr, n_epochs):
    df = pd.read_csv(train_csv_dir,header=0)
    df = df.replace(100,-110)
    df = df[(df["BUILDINGID"] == bid) & (df["FLOOR"] == fid)].iloc[:,:522]
    col = ["AP"+str(i) for i in range(1,521)]
    col.append("LONGITUDE")
    col.append("LATITUDE")
    df.columns = col
    dirich = pd.read_csv(dirich_csv_dir,header=0)
    df = df.append(dirich,ignore_index=True)
    training_data_csv_in = np.array(df.iloc[:,:520])
    training_data_csv_lb = np.array(df.iloc[:,520:522])

    if torch.cuda.is_available():
        training_data_csv_in = torch.from_numpy(training_data_csv_in).cuda()
        training_data_csv_lb = torch.from_numpy(training_data_csv_lb).cuda()
    
    count = 1
    store={}
    for i in range(len(training_data_csv_in)): 
        k = training_data_csv_lb[i].tolist()
        v = training_data_csv_in[i].tolist()
    
        if tuple(k) in store:
    
            store[tuple(k)].append(v)
        else:
            store[tuple(k)] = [v]
    print("Size of unique RP: ", len(store.keys()))
    for unique in store.keys():
     
        print("Training {}/{}: ".format(count,len(store.keys())))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = Generator(z_dim).to(device)
        disc = Discriminator().to(device)
        gen.load_state_dict(torch.load(gen_state))
        disc.load_state_dict(torch.load(disc_state))
        
            
        gen_loss = False
        criterion = nn.BCEWithLogitsLoss()
        # Load MNIST dataset as tensor
    
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr) 
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
  
        val = store[unique]
        store_in = []
        for v in val:
            store_in.append(np.asarray(v))
        tensor_store_val = torch.Tensor(store_in).cuda()
      
    ###
        dataloader = DataLoader(tensor_store_val, batch_size=4)
        # directory = "/home/wayne/uji/gan+_model_state/b"+str(bid)+"f"+str(fid)+"/"
        directory = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/b"+str(bid)+"f"+str(fid)+"/"
        gen_save_state = directory+str(unique[0])+"_"+str(unique[1])+".pt"
        if not os.path.exists(directory):
            os.makedirs(directory)
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
            
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
                gen_loss.backward(retain_graph=True)
                gen_opt.step()
      
        torch.save(gen.state_dict(), gen_save_state)
        count+=1

    
def generate_img_csv():
    #generate wgan-gp images for after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_img = []
    gen = Generator(10).to(device)
    gen.load_state_dict(torch.load("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/b0f0/-7632.143599998206_4864982.217100002.pt"))
    gen.eval()
    fake_noise = get_noise(1, 10, device=device)
    fake = gen(fake_noise)
 
    fake = rpreprocess(fake[0])
    print(fake)
   
# =============================================================================
#     for i in range(len(labels)):
#         gen = Generator(10).to(device)
#         gen.load_state_dict(torch.load(model_state_dir+labels[i][0]+"_"+labels[i][1]+".pt"))
#         gen.eval()
#         print(model_state_dir+labels[i][0]+"_"+labels[i][1]+".pt")
#         for k in range(2): #75*3
#             fake_noise = get_noise(1, 10, device=device)
#             fake = gen(fake_noise)
#     
#             fake = rpreprocess(fake[0])
#             
#             temp_fake = fake.detach().cpu().numpy().tolist()
#             temp_fake.append(labels[i][0])
#             temp_fake.append(labels[i][1])
#             gen_img.append(temp_fake)
#     col = ["AP"+str(i) for i in range(1,521)]
#     col.append("LONGITUDE")
#     col.append("LATITUDE")
#     df = pd.DataFrame(gen_img, columns = col)
#     print(df)
# =============================================================================
    # print(fake)
# =============================================================================
#     curr_label = data_dir.split('\\')
#     directory = save_dir+'/'+str(curr_label[-1])
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for i in range(num_iter):
#         noise = torch.randn(1, latent, 1, 1).to(device)
#         fake_noise = get_noise(1, latent, device=device)
#         fake = gen(fake_noise).detach().cpu()
#         fake = rpreprocess(fake)
#         imsave(fake, save_dir+'/'+str(curr_label[-1])+'/'+str(i)+'.png', px, my_dpi)
# =============================================================================

    
# =============================================================================
# def find_gan_performance(act_img, gan_img):
#     sol = 0
#     for i in range(len(act_img)):
#         temp = abs(act_img[i] - gan_img[i])
#         sol+=temp
#     return sol
# =============================================================================


if __name__ == "__main__":
    
# =============================================================================
#    1. Dirichlet augmentation
# =============================================================================

    ### Input parameters ###
    fid=['b1f0']
    #fid = ['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3','b2f0','b2f1', 'b2f2', 'b2f3', 'b2f4']
    csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/train_split/"
    save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/dirichlet_add/"
    
    for i in fid:
        dirichlet_generate(csv_dir, i, save_dir)
# =============================================================================
#  2. Training GAN+ using most samples
# =============================================================================
    # train_most_sample()
    
        
# =============================================================================
#   GAN+ with pre-training (original + dirichlet)
# =============================================================================
    train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    dirich_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/dirichlet/b0f0.csv"   
    gen_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/gen_gan+_most_sample.pt"
    disc_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/UJI/gan+_model_state/disc_gan+_most_sample.pt"
# =============================================================================
#     train_csv_dir = "/home/wayne/uji/csv_files/UJI-trainingData.csv"
#     dirich_csv_dir = "/home/wayne/uji/csv_files/dirichlet/b0f0.csv"   
#     gen_state = "/home/wayne/uji/gen_gan+_most_sample.pt"
#     disc_state = "/home/wayne/uji/disc_gan+_most_sample.pt"
# =============================================================================
    fid = 0
    bid = 0
    z_dim = 10
    n_epochs = 500
    lr = 0.00001
    
    # gan_pretrained(train_csv_dir, dirich_csv_dir, fid, bid, gen_state, disc_state, z_dim, lr, n_epochs)

    # generate_img_csv()


    


    

    





  

