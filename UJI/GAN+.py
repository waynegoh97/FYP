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

def find_building_min(csv, bid):
    '''
    Purpose: Find the minimum coordinate for the building

    Parameters
    ----------
    csv : string
        csv file
    bid : int
        building id

    Returns
    -------
    bid_min : list
        list containing the minimum coordinate [latitude, longitude]

    '''
    df = pd.read_csv(csv, header=0)
    temp = df[df['BUILDINGID'] == bid]
    unique = temp.groupby(["LATITUDE","LONGITUDE"]).size().reset_index().rename(columns={0:'count'})
    unique = np.array(unique.iloc[:,:2])
    bid_min = np.amin(unique, axis=0).tolist()
    return bid_min

def build_rss_images(data,csv):
    new_data = []
    all_data = pd.read_csv(csv, header=0)
    all_data = np.array(all_data.iloc[:,:520])
    sample_records = np.transpose(np.asarray([all_data[data[0]][0:520]]))

    for j in range(1, 10):
        sample_records = np.append(sample_records, np.transpose(np.asarray([all_data[data[j]][0:520]])), axis=1)

    new_data.append(sample_records)

    new_data = np.asarray(new_data)
    new_data[new_data == 100] = -110
    return new_data

def get_cell_center_lon_lat(data_cells, min_lat, min_lon):
    cells_centers_lon_lat = []

    cell_dimension = 3

    row = float(data_cells[0])
    col = float(data_cells[1])

    # get the top left point of the cell square
    x1 = min_lon + cell_dimension * col
    y1 = min_lat + cell_dimension * row

    cell_center_x = x1 + cell_dimension / 2
    cell_center_y = y1 + cell_dimension / 2
    cells_centers_lon_lat.append(np.asarray([cell_center_x, cell_center_y]))

    cells_centers_lon_lat = np.asarray(cells_centers_lon_lat)

    return cells_centers_lon_lat
# =============================================================================
#     sample_records = np.transpose(np.asarray([all_data[data[0]][0:520]]))
# 
#     for j in range(1, 10):
#         sample_records = np.append(sample_records, np.transpose(np.asarray([all_data[data[j]][0:520]])), axis=1)
# 
#     new_data.append(sample_records)
# 
#     new_data = np.asarray(new_data)
#     new_data[new_data == 100] = -110
#     return new_data
# =============================================================================

if __name__ == "__main__":
# =============================================================================
#     Dirichlet augmentation
# =============================================================================
    ### Input parameters ###
    csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    bid = [0,1,2]
    
    #Find minimum coordinates for each building
    # min_coord = []
    # for i in bid:
    #     min_coord.append(find_building_min(csv, i))
    
    ### Input parameters ###
    fid = ['b2f4']#, 'b2f1', 'b2f2', 'b2f3', 'b2f4'] #'b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3',
    csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/train_split/"
    
    #store dictionary contains coordinates (latitude, longitude) as keys and rssi data as values (according to the number of samples belonging to coordinate) e.g. {(-7345.345, 4328596): [[AP0 ... AP520],[AP0 ... AP520]]}
    store = {}
    for i in fid:
        train_df = pd.read_csv(csv_dir+i+"_train.csv",header=0)
        train_input = np.array(train_df.iloc[:,:520])
        train_label = np.array(train_df.iloc[:,520:522])
        for j in range(len(train_input)): 
            k = train_label[j].tolist()
            v = train_input[j].tolist()
        
            if tuple(k) in store:
                store[tuple(k)].append(v)
            else:
                store[tuple(k)] = [v]
# =============================================================================
#   Add a dict that stores number of dirich samples needed for each RP, then subtract from the df
# =============================================================================
    ### Total dirichlet needed = 75 - original images, another 75 images is generated by GAN+, with a total of 150 images for localisation ###
    tempim = []
    templb = []

    for k in store.keys(): #number of unique RP
        #Just need 75 samples 
        if (len(store[k]) <= 75) & (len(store[k]) != 1):
            sample_size = len(store[k])
            dirich_needed = 75 - sample_size
            #randomly pick samples for dirichlet
            im_new  = [[0 for i in range(sample_size)] for j in range(520)]
            curr_rp_samples = np.array(store[k])
            curr_rp_samples[curr_rp_samples==100] = -110
            curr_rp_samples = curr_rp_samples.T
            for gen in range(math.ceil(dirich_needed/sample_size)):
                for s in range(sample_size):
                    weight = np.random.dirichlet(np.ones(sample_size),size=1)
                    for ap in range(520):
                        im_new[ap][s] = np.sum(np.multiply(weight,curr_rp_samples[ap][:]))
                        if(110+im_new[ap][s]<=0.000001):
                            im_new[ap][s] = -110.0
                
                tempim.append((np.array(im_new).T).tolist())
                templb.append([k[0],k[1]])
    
    #Create df
    #Create column names
    col = ["AP"+str(i) for i in range(1,521)]
    df = pd.DataFrame(tempim[0], columns= col)
    df["LATITUDE"] = templb[0][0]
    df["LONGITUDE"] = templb[0][1]
    for i in range(1,len(tempim)):
        df2 = pd.DataFrame(tempim[i], columns= col)
        df2["LATITUDE"] = templb[i][0]
        df2["LONGITUDE"] = templb[i][1]
        df=df.append(df2,ignore_index=True)
        
    df.to_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/dirichlet/b2f4.csv",index=False)
    

    
    
# =============================================================================
#     df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/gan+/b1f3_train.csv",header=0)
#     training_data_csv_in = np.array(df.iloc[:,:520])
#     training_data_csv_lb = np.array(df.iloc[:,520:522])
#     
#     # training_data_csv_in = np.loadtxt('out_im_dirichlet.csv', delimiter=',', dtype=np.float)
#     # training_data_csv_lb = np.loadtxt('out_lb_dirichlet.csv', delimiter=',', dtype=np.float)
#     
#     if torch.cuda.is_available():
#         training_data_csv_in = torch.from_numpy(training_data_csv_in).cuda()
#         training_data_csv_lb = torch.from_numpy(training_data_csv_lb).cuda()
#     
#     store={}
#     for i in range(len(training_data_csv_in)): 
#         k = training_data_csv_lb[i].tolist()
#         v = training_data_csv_in[i].tolist()
#     
#         if tuple(k) in store:
#     
#             store[tuple(k)].append(v)
#         else:
#             store[tuple(k)] = [v]
#     
#     
#     
#     
#     
#     test_generator = True # Whether the generator should be tested
#     gen_loss = False
#     error = False
#     criterion = nn.BCEWithLogitsLoss()
#     n_epochs = 1000
#     z_dim = 10
#     display_step = 500
#     batch_size = 128
#     lr = 0.00001
#     device = 'cuda'
#     # Load MNIST dataset as tensor
# 
#     gen = Generator(z_dim).to(device)
#     gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
#     disc = Discriminator().to(device) 
#     disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
# 
#     if torch.cuda.is_available():
#         gen = gen.cuda()
#         disc = disc.cuda()
#     
#     # from utils import Logger
#     count = 1
#     for k in store.keys():
#         
#         val = store[k]
#         store_in = []
#         for v in val:
#             store_in.append(np.asarray(v))
#         tensor_store_val = torch.Tensor(store_in).cuda()
#     
#       ###
#         dataloader = DataLoader(tensor_store_val, batch_size=4)
#       # dataloader = DataLoader(tensor_store_val, batch_size=len(val))
#         save_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/model_state/UJI/gan+_model_state/b1f3/"+str(k[0])+"_"+str(k[1])+".pt"
#         print("Start training: {}/{}".format(count,len(store.keys())))
#         for epoch in range(n_epochs):
#         
#           # Dataloader returns the batches
#             for real in dataloader:
#     
#                 if torch.cuda.is_available():
#                     real = real.cuda()
#     
#                 cur_batch_size = len(real)
#     
#                 if cur_batch_size != 4:
#                     continue
#               
#               # Flatten the batch of real images from the dataset
#                 real_b = real.view(cur_batch_size, -1).to(device)
#                 if torch.cuda.is_available():
#                     real_b = real_b.cuda()
#     
#                 real = preprocess(real_b)
#     
#                 if torch.cuda.is_available():
#                     real = real.cuda()
#               ### Update discriminator ###
#               # Zero out the gradients before backpropagation
#                 disc_opt.zero_grad()
#     
#               # Calculate discriminator loss
#                 disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
#     
#               # Update gradients
#                 disc_loss.backward(retain_graph=True)
#     
#               # Update optimizer
#                 disc_opt.step()
#     
#               # For testing purposes, to keep track of the generator weights
#                 if test_generator:
#                     old_generator_weights = gen.gen[0][0].weight.detach().clone()
#     
#               ### Update generator ###
#               #     Hint: This code will look a lot like the discriminator updates!
#               #     These are the steps you will need to complete:
#               #       1) Zero out the gradients.
#               #       2) Calculate the generator loss, assigning it to gen_loss.
#               #       3) Backprop through the generator: update the gradients and optimizer.
#               #### START CODE HERE ####
#               
#                 gen_opt.zero_grad()
#                 gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
#                 gen_loss.backward(retain_graph=True)
#                 gen_opt.step()
#               
#               #### END CODE HERE ####
#     
#               # For testing purposes, to check that your code changes the generator weights
#                 if test_generator:
#                     try:
#     
#                         assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
#                         assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
#                     except:
#                         error = True
#                         print("Runtime tests have failed")
#                         
#         torch.save(gen.state_dict(), save_state)
#         count+=1
# =============================================================================
        
      
      
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
    # =============================================================================
    # num_gen = 150 #number of image to generate
    # for i in range(num_gen):
    #     fake_noise = get_noise(1,z_dim,device=device)
    #     fake = gen(fake_noise)
    # =============================================================================
    





  

