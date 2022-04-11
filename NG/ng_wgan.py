# -*- coding: utf-8 -*-
from torchvision import models
# from torchsummary import summary
import torch.nn as nn
import torch 
import torch.optim as optim
import shutil
import random

# import sys
# sys.path.append("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/")
from util import *

# For 19x19 images
class NG_Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(NG_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 19 x 19 
            nn.Conv2d(
                channels_img, features_d*2, kernel_size=3, stride=2, padding=1
            ), #64x10x10
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d*2, features_d * 4, 4, 2, 1), #128x5x5
            self._block(features_d * 4, features_d * 8, 3, 2, 1), #256x3x3
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

class NG_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(NG_Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 8, 3, 1, 0),  # img: 3x3
            self._block(features_g * 8, features_g * 4, 3, 2, 1),  # img: 5x5
            self._block(features_g * 4, features_g*2 , 4, 2, 1),  # img: 10x10
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=3, stride=2, padding=1
            ),
            # Output: N x channels_img x 19x19
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
    
def wgan_gp_train(gen_model_state, disc_model_state, data_dir, lr, batch, num_epoch, img_dim, img_channel, latent, critic_layer, gen_layer, critic_iter, gradient_p):
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
    dataloader, mean, std, _, _= load_by_label(label_dir[0], batch)
    unorm = UnNormalize(mean = mean, std = std)
# =============================================================================
#     for i in range(len(label_dir)):
#         dataloader = load_by_label(label_dir[i], batch)
# =============================================================================
    gen = NG_Generator(latent, img_channel, gen_layer).to(device) 
    critic = NG_Discriminator(img_channel, critic_layer).to(device)
    initialize_weights(gen)
    initialize_weights(critic)
    
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
         
  
            
        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [G loss: %f] [C loss: %f]"
            % (epoch+1, num_epoch, batch_idx+1, len(dataloader), loss_gen, loss_critic)
        )
        critic_loss.append(-loss_critic)
            
        if epoch%20 == 0:
            with torch.no_grad():
                gen.eval()
                noise = torch.randn(1, latent, 1, 1).to(device)
                fake = gen(noise).detach().cpu()
                imshow(unorm(fake))
                

    torch.save(gen.state_dict(), gen_model_state)
    torch.save(critic.state_dict(), disc_model_state)
    plt.figure(figsize=(10, 7))
    plt.plot(critic_loss, label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Critic Loss')
    plt.title('Critic plot')
    plt.legend(frameon=False)
    
def generate_wgan_img(num_iter, model_state_dir, data_dir, latent, gen_layer, img_channel, px, my_dpi, save_dir):
    #generate wgan-gp images for after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = NG_Generator(latent, img_channel, gen_layer).to(device) 
    gen.load_state_dict(torch.load(model_state_dir))
    gen.eval()
    
    # label_dir = label_directory(data_dir)
    _, mean, std, _, _ = load_by_label(data_dir, 1)
    unorm = UnNormalize(mean = mean, std = std)
    curr_label = data_dir.split('\\')
    directory = save_dir+'/'+str(curr_label[-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(num_iter):
        noise = torch.randn(1, latent, 1, 1).to(device)
        fake = gen(noise).detach().cpu()
        # imshow(unorm(fake))
        imsave(unorm(fake), save_dir+'/'+str(curr_label[-1])+'/'+str(i)+'.png', px, my_dpi)
        
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
        
    gen = NG_Generator(latent, img_channel, gen_layer).to(device)
    critic = NG_Discriminator(img_channel, critic_layer).to(device)
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
         
  
            
        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [G loss: %f] [C loss: %f]"
            % (epoch+1, num_epoch, batch_idx+1, len(dataloader), loss_gen.item(), loss_critic.item())
        )
        critic_loss.append(-loss_critic.item())
                        
    torch.save(gen.state_dict(), save_state)
    
def generate_wgan_img(num_iter, model_state_dir, data_dir, latent, gen_layer, img_channel, px, my_dpi, save_dir):
    #generate wgan-gp images for after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = NG_Generator(latent, img_channel, gen_layer).to(device)
    gen.load_state_dict(torch.load(model_state_dir))
    gen.eval()
    
    # label_dir = label_directory(data_dir)
    _, mean, std, _, _ = load_by_label(data_dir, 1)
    unorm = UnNormalize(mean = mean, std = std)
    curr_label = data_dir.split('\\')
    directory = save_dir+'/'+str(curr_label[-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(num_iter):
        noise = torch.randn(1, latent, 1, 1).to(device)
        fake = gen(noise).detach().cpu()
        # imshow(unorm(fake))
        imsave(unorm(fake), save_dir+'/'+str(curr_label[-1])+'/'+str(i)+'.png', px, my_dpi)
        
def split_img():
    ### arrange training datas by shfting images into 3 cases
    num_img = 150
    case_name = ['_wgan', '_original_wgan', '_mix'] #floor id followed by case name
    ### Find floor id ###
    floor_id = ['floor-1','floor1','floor2']
    ######
    for k in range(len(floor_id)):
        for i in range(len(case_name)):
            ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/train_img/" + floor_id[k]+"_train/" #original image folder
            aug_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/wgan/"+floor_id[k]+"/" #augmented image folder
            new_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/train_img/"+floor_id[k]+case_name[i]+"/" #new data image path
            
            
            label_name = os.listdir(ori_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for l in range(len(label_name)):
                if not os.path.exists(new_dir+label_name[l]):
                    os.makedirs(new_dir+label_name[l])
                    
            if i == 0:
                for i in range(len(label_name)):
                    img_name = os.listdir(aug_dir + label_name[i])
                    for img in img_name:
                        shutil.copy(aug_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/wgan_'+img)
            if i == 1: 
                for i in range(len(label_name)):
                    img_name = os.listdir(ori_dir+label_name[i])
                    for img in img_name:
                        shutil.copy(ori_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/'+img)
                    img_name = os.listdir(aug_dir + label_name[i])
                    for img in img_name:
                        shutil.copy(aug_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/wgan_'+img)
            if i == 2:
                file_size = []
                for i in range(len(label_name)):
                    img_name = os.listdir(ori_dir+label_name[i])
                        
                    file_size.append(len(img_name))
                    for img in img_name:
                        shutil.copy(ori_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/'+img)
                aug_img = [150-x for x in file_size]
                
                for i in range(len(label_name)):
                    img_names = os.listdir(aug_dir+label_name[i])
                    selected = random.sample(img_names, aug_img[i])
                    for file in selected:
                        shutil.copy(aug_dir+label_name[i]+'/'+file, new_dir+label_name[i]+'/wgan_'+file)
        
if __name__ == "__main__":
# =============================================================================
#     1. Checking GAN model
# =============================================================================
# =============================================================================
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     critic = NG_Discriminator(1, 64).to(device)
#     gen = NG_Generator(100, 1, 64).to(device) 
#     summary(critic, (1,19,19))
#     summary(gen, (100,1,1))
# =============================================================================
# =============================================================================
# 
# =============================================================================
    LEARNING_RATE =0.001 #0.001 (mnist)
    BATCH_SIZE = 4 #32 (mnist), 8 for wgan-gp uji
    IMAGE_SIZE = 19
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 1000
    FEATURES_CRITIC = 64
    FEATURES_GEN = 64
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    num_gen = 400
    my_dpi = 96 # Can be found using this link https://www.infobyip.com/detectmonitordpi.php
# =============================================================================
#     data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/most_sample/' #for most labels
#     gen_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/model_state/NG/gen_wgangpUJI-1000-0_001_4.pt'
#     disc_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/model_state/NG/disc_wgangpUJI-1000-0_001_4.pt'
# =============================================================================
    gen_saved_state = '/home/wayne/ng/gen_wgangpUJI-1000-0_001_4.pt'
    disc_saved_state = '/home/wayne/ng/disc_wgangpUJI-1000-0_001_4.pt'
    # wgan_gp_train(gen_saved_state, disc_saved_state, data_dir, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, IMAGE_SIZE, CHANNELS_IMG, Z_DIM, FEATURES_CRITIC,
    #         FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)
    unique_loc = ['floor-1','floor1','floor2']
    for loc in range(len(unique_loc)):
# =============================================================================
        data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/WGAN-GP+/train_img/'+unique_loc[loc]
        gen_img_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/WGAN-GP+/generated_img/'+unique_loc[loc]
        saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/NG/WGAN-GP+_model_states/'+unique_loc[loc]
# =============================================================================
#         data_dir = '/home/wayne/ng/images/WGAN-GP+/train_img/'+unique_loc[loc]
#         # gen_img_dir = '/home/wayne/ng/images/WGAN-GP+/generated_img/'+unique_loc[loc]
#         save_state = '/home/wayne/ng/extendedGAN+_model_state/'+unique_loc[loc]
        
        if not os.path.exists(gen_img_dir):
            os.makedirs(gen_img_dir)
        # if not os.path.exists(save_state):
        #     os.makedirs(save_state)
        NUM_EPOCHS = 500
        label_dir = label_directory(data_dir)

        for i in range(len(label_dir)):    
            curr_label = label_dir[i].split('\\')
            # curr_label = label_dir[i].split('/')
            # save_state = '/home/wayne/ng/wgan_model_states/'+unique_loc[loc]+'/'+str(curr_label[-1])+'.pt'
            save_state = saved_state + '/'+str(curr_label[-1])+'.pt'
    
            # wgan_gp_pretrain(save_state, gen_saved_state, disc_saved_state, label_dir[i], LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, IMAGE_SIZE,
            #                   CHANNELS_IMG, Z_DIM, FEATURES_CRITIC, FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)
            generate_wgan_img(num_gen, save_state, label_dir[i], Z_DIM, FEATURES_GEN, CHANNELS_IMG, IMAGE_SIZE, my_dpi, gen_img_dir)
# ==========================================================================
#     save_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/model_state/NG/wgan_model_states/floor2/30040.490428222794_30306.64760825761.pt'
#     label_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/train_only/floor2/30040.490428222794_30306.64760825761'
#     generate_wgan_img(150, save_state, label_dir, 100, 64, 1, 19, my_dpi, gen_img_dir)    
# =============================================================================
    # split_img()













