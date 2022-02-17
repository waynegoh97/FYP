# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 10:53:57 2022

@author: noxtu
"""
import torch
import torch.nn as nn

from torchvision import models
from torchsummary import summary
from util import *


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
    
def wgan_gp_train(gen_model_state, disc_model_state, data_dir, dataset, lr, batch, num_epoch, img_dim, img_channel, latent, critic_layer, gen_layer, critic_iter, gradient_p):
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
    gen = UJI_Generator(latent, img_channel, gen_layer).to(device) 
    critic = UJI_Discriminator(img_channel, critic_layer).to(device)
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
            % (epoch+1, num_epoch, batch_idx+1, len(dataloader), loss_gen.item(), loss_critic.item())
        )
        critic_loss.append(-loss_critic.item())
            
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
    
def resnet18():
    '''

    Returns
    -------
    resnet18 : model
        structure of the CNN model

    '''
    resnet18 = models.resnet18()
    n_inputs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(n_inputs, 2) #change last layer to 2 outputs
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return resnet18
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18().to(device)
    # summary(model, (3,23,23))
    print(model)
# =============================================================================
#     critic = UJI_Discriminator(1, 64).to(device)
#     gen = UJI_Generator(100, 1, 64).to(device) 
#     # model = models.resnet18().to(device)
#     summary(critic, (1,23,23))
#     summary(gen, (100,1,1))
# =============================================================================
    
