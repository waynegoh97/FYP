# -*- coding: utf-8 -*-
import csv

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch
import torch.optim as optim
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

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
    def __init__(self, img_path, label, transform):
        self.transform = transform
        # self.img_folder=img_folder
        self.img_path = img_path
        self.label = label

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image = Image.open(self.img_path[index], 'r')
        image = self.transform(image)
        targets = self.label[index]

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
                                    transforms.ToTensor(), transforms.Normalize((mean), (std))])
    # , transforms.Normalize((mean),(std))
    data_set = ImageDataset(img_path, label, transform)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)

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
                channels_img, features_d * 2, kernel_size=3, stride=2, padding=1
            ),  # 64x12x12
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 128x6x6
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 256x3x3
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0),  # 1x1x1
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
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 12x12
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=3, stride=2, padding=1
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

# For 18x18 images
class N4_Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(N4_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 18 x 18
            nn.Conv2d(
                channels_img, features_d * 2, kernel_size=2, stride=2, padding=0
            ),  # 64x9x9
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d * 2, features_d * 4, 3, 2, 1),  # 128x5x5
            self._block(features_d * 4, features_d * 8, 3, 2, 0),  # 256x2x2
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=2, stride=1, padding=0),  # 1x1x1
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
class N4_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(N4_Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 8, 2, 1, 0),  # img: 2x2
            self._block(features_g * 8, features_g * 4, 3, 2, 0),  # img: 5x5
            self._block(features_g * 4, features_g * 2, 3, 2, 1),  # img: 9x9
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=2, stride=2, padding=0
            ),
            # Output: N x channels_img x 18x18
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


def wgan_gp_train(gen_model_state, disc_model_state, data_dir, lr, batch, num_epoch, img_dim, img_channel, latent,
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

        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [G loss: %f] [C loss: %f]"
            % (epoch + 1, num_epoch, batch_idx + 1, len(dataloader), loss_gen, loss_critic)
        )
        critic_loss.append(-loss_critic)

        # if epoch % 20 == 0:
        #     with torch.no_grad():
        #         gen.eval()
        #         noise = torch.randn(1, latent, 1, 1).to(device)
        #         fake = gen(noise).detach().cpu()
        #         imshow(unorm(fake))


    torch.save(gen.state_dict(), gen_model_state)
    torch.save(critic.state_dict(), disc_model_state)



def wgan_gp_pretrain(save_state, gen_model_state, disc_model_state, data_dir,
                     lr, batch, num_epoch, img_channel, latent, critic_layer, gen_layer, critic_iter,
                     gradient_p):
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
    dataloader, mean, std, _, _ = load_by_label(data_dir, batch)
    unorm = UnNormalize(mean=mean, std=std)
    gen = N4_Generator(latent, img_channel, gen_layer).to(device)
    critic = N4_Discriminator(img_channel, critic_layer).to(device)
    gen.load_state_dict(torch.load(gen_model_state))
    critic.load_state_dict(torch.load(disc_model_state))

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

        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [G loss: %f] [C loss: %f]"
            % (epoch+1, num_epoch, batch_idx+1, len(dataloader), loss_gen.detach(), loss_critic.detach())
        )
        critic_loss.append(-loss_critic.detach())

    torch.save(gen.state_dict(), save_state)



def generate_wgan_img(num_iter, model_state_dir, data_dir, latent, gen_layer, img_channel, px, my_dpi, save_dir):
    # generate wgan-gp images for after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = N4_Generator(latent, img_channel, gen_layer).to(device)
    gen.load_state_dict(torch.load(model_state_dir))
    gen.eval()

    _, mean, std, _, _ = load_by_label(data_dir, 1)
    unorm = UnNormalize(mean=mean, std=std)
    curr_label = data_dir.split('\\')
    directory = save_dir + '/' + str(curr_label[-1])
    # directory = save_dir+'/'+curr_label
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(num_iter):
        noise = torch.randn(1, latent, 1, 1).to(device)
        fake = gen(noise).detach().cpu()
        imsave(unorm(fake), save_dir + '/' + str(curr_label[-1]) + '/' + str(i) + '.png', px, my_dpi)
        # imsave(unorm(fake), save_dir+'/'+curr_label+'/'+str(i)+'.png', px, my_dpi)

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
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)


def abs_diff_gen(ori_dir, gen_dir, img_dim, ori_diff):
    # Loading data

    ori_dataloader, _, _, _, _ = load_by_label(ori_dir, 1, False)
    gen_dataloader, _, _, gen_img_name, gen_label = load_by_label(gen_dir, 1, False)
    # Extract tensors of images and turn it into array
    ori_list = []
    gen_list = []
    gen_abs_diff_dict = {}
    for batch_idx, (data, _) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
    for batch_idx, (data, _) in enumerate(gen_dataloader):
        gen_list.append(data.numpy())
    ori_list = np.array(ori_list).reshape((-1, img_dim, img_dim))
    gen_list = np.array(gen_list).reshape((-1, img_dim, img_dim))
    # Find the abs diff of each generated images against all original images
    for k in range(len(gen_list)):
        result = []
        diff = []
        for ori in range(len(ori_list)):
            result.append(np.absolute(ori_list[ori] - gen_list[k]))
        for i in range(len(result)):
            diff.append(np.array(result[i]).sum())
        gen_abs_diff_dict[gen_img_name[k]] = diff

    # Find the min diff for each generated images [list contains min diff in image order (e.g. 0.png, 1.png...)]
    min_list = []
    img_name = []
    for i in range(len(gen_list)):
        # min_list.append(sum(gen_abs_diff_dict[str(i)+'.png'])/len(gen_abs_diff_dict[str(i)+'.png']))
        if str(i) + '.png' in gen_abs_diff_dict:
            min_list.append(min(gen_abs_diff_dict[str(i) + '.png']))
            img_name.append(str(i) + '.png')
    dict_col = {'img_name': img_name, 'abs_score': min_list}
    df = pd.DataFrame(dict_col)
    df = df[df['abs_score'] <= (ori_diff)]
    #df = df.sort_values(by=['abs_score'], ignore_index=True)
    dfa = np.array(df.iloc[:len(df), 0])
    # =============================================================================
    #     for rename in range(20):
    #         new_name = 'new_'+str(rename)+'.png'
    #         os.rename(gen_dir + '/' +df.iloc[rename,0], gen_dir +'/'+new_name)
    # =============================================================================
    return dfa


def abs_diff_ori(ori_dir, img_dim):
    ori_dataloader, _, _, _, _ = load_by_label(ori_dir, 1, False)
    ori_list = []
    for batch_idx, (data, _) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
    ori_list = np.array(ori_list).reshape((-1, img_dim, img_dim))

    result = []
    diff = []
    for ori in range(len(ori_list)):
        for i in range(len(ori_list) - (ori + 1)):
            result.append(np.absolute(ori_list[ori] - ori_list[ori + i + 1]))

        for i in range(len(result)):
            diff.append(np.array(result[i]).sum())

    max_diff = max(diff)
    return max_diff

def overall_threshold(loclist, csvdir):
    size = 0
    threshold = 0
    for bfid in loclist:
        df = pd.read_csv(csvdir+bfid+".csv",header=0)
        size+= len(df)
        temp_th =  df["max_threshold"].sum()
        threshold+= temp_th
    threshold = threshold/size
    return threshold


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
    #   2. WGAN-GP training for most sample
    # =============================================================================
    LEARNING_RATE = 0.001  # 0.001 (mnist)
    BATCH_SIZE = 4  # 32 (mnist), 8 for wgan-gp uji
    IMAGE_SIZE = 18
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 1000
    FEATURES_CRITIC = 64
    FEATURES_GEN = 64
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    num_gen = 400
    my_dpi = 96  # Can be found using this link https://www.infobyip.com/detectmonitordpi.php
    # data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/most_sample/'  # for most labels
    # gen_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/model_state/WGAN-GP/gen_most_sample.pt'
    # disc_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/model_state/WGAN-GP/disc_most_sample.pt'
    #
    # wgan_gp_train(gen_saved_state, disc_saved_state, data_dir, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, IMAGE_SIZE, CHANNELS_IMG, Z_DIM, FEATURES_CRITIC,
    #         FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)


    # =============================================================================
    #     3. WGAN-GP pre-train for each RP
    # =============================================================================
    # unique_loc = "F1Sb"
    # data_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/train/"+unique_loc #"/home/SEANGLIDET/n4/images/WGAN-GP/train/"+unique_loc
    # saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/"+unique_loc #"/home/SEANGLIDET/n4/model_state/WGAN-GP/"+unique_loc
    # gen_saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/gen_most_sample.pt"#'/home/SEANGLIDET/n4/model_state/gen_most_sample.pt'
    # disc_saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/disc_most_sample.pt"#'/home/SEANGLIDET/n4/model_state/disc_most_sample.pt'
    #
    # if not os.path.exists(saved_state):
    #     os.makedirs(saved_state)
    # NUM_EPOCHS = 500
    # label_dir = label_directory(data_dir)
    # # label_dir = ['58.8_12.86', '58.8_15.31', '58.8_17.76'] #29,32
    # # ['3.68_15.31'] #22,23
    #
    # for i in range(22,23):#len(label_dir)):
    #     # curr_label = label_dir[i].split("/")
    #     curr_label = label_dir[i].split('\\') #for windows pc
    #     save_state = saved_state + '/'+str(curr_label[-1])+'.pt'
    #     print(curr_label, save_state)
    #     wgan_gp_pretrain(save_state, gen_saved_state, disc_saved_state, label_dir[i], LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
    #                       CHANNELS_IMG, Z_DIM, FEATURES_CRITIC, FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP)



    # =============================================================================
    #     4. Generate images
    # =============================================================================
    # =============================================================================
    unique_loc = ["F1Sb"]


    for u in unique_loc:
        data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/train/'+u
        gen_img_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/unfiltered/'+u
        saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/'+u

         #for individual gen
        # if not os.path.exists(gen_img_dir):
        #     os.makedirs(gen_img_dir)
        # l = data_dir + "\\-7691.338399998844_4864928.212899998"
        # save_state = saved_state + '/-7691.338399998844_4864928.212899998.pt'
        # generate_wgan_img(num_gen, save_state, l, Z_DIM, FEATURES_GEN, CHANNELS_IMG, IMAGE_SIZE, my_dpi, gen_img_dir)

        label_dir = label_directory(data_dir)
        for i in range(22,23):#len(label_dir)):
            curr_label = label_dir[i].split('\\')  # for windows pc
            print(curr_label)
            save_state = saved_state + '/' + str(curr_label[-1]) + '.pt'


            generate_wgan_img(num_gen, save_state, label_dir[i], Z_DIM, FEATURES_GEN, CHANNELS_IMG, IMAGE_SIZE,
                              my_dpi, gen_img_dir)


    # =============================================================================
    # =============================================================================
    #   4. Filtering images
    # =============================================================================

    # Finding threshold by finding the max diff between original images

    # unique_loc = "F2Sb"
    # img_dim = 18
    # max_threshold = []
    #
    # csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/max_threshold/"
    # ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/train_only/"+unique_loc
    # curr_labels = os.listdir(ori_dir)
    # for k in curr_labels:
    #     max_threshold.append([k, abs_diff_ori(ori_dir+"/"+k, img_dim)])
    #     df = pd.DataFrame(max_threshold, columns = ["labels", "max_threshold"])
    #     df.to_csv(csv_dir+unique_loc+".csv", index=False)

    #Filtering generated images with local threshold

    bidd = ['F1Sa']
    for bid in bidd:
        ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/train_only/" + bid + "/"
        curr_labels = os.listdir(ori_dir)

        img_dim = 18
        gen_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/unfiltered/" + bid + "/"
        threshold_df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/max_threshold/" + bid + ".csv")
        new_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/filtered/" + bid + "/"
        size = []
        # for individual filter
        # lb = "58.8_17.76"
        # ori_diff = threshold_df[threshold_df["labels"] == lb]
        # ori_diff = ori_diff.iloc[0, 1]
        # df = abs_diff_gen(ori_dir + lb, gen_dir + lb, img_dim, ori_diff)
        # size.append(len(df))
        # for k in range(len(df)):
        #     shutil.copy(gen_dir + lb + "/" + df[k], new_dir + lb + '/' + df[k])

        for i in range(len(curr_labels)):
            ori_diff = threshold_df[threshold_df["labels"] == curr_labels[i]]
            ori_diff = ori_diff.iloc[0, 1]
            df = abs_diff_gen(ori_dir + curr_labels[i], gen_dir + curr_labels[i], img_dim, ori_diff)
            size.append(len(df))
            for k in range(len(df)):
                if not os.path.exists(new_dir + curr_labels[i]):
                    os.makedirs(new_dir + curr_labels[i])
                shutil.copy(gen_dir + curr_labels[i] + "/" + df[k], new_dir + curr_labels[i] + '/' + df[k])
        print(size)

    #Filtering with overall threshold

    # bid = ['floor-1','floor1','floor2']#["b0f0","b0f1", "b0f2", "b0f3", "b1f0","b1f1", "b1f2", "b1f3","b2f0","b2f1", "b2f2", "b2f3", "b2f4"]
    # csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/max_threshold/"
    # othreshold = overall_threshold(bid,csv_dir)
    # print(othreshold)
    # bid = ["floor2"]
    # for b in bid:
    #     ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/train_img/" + b + "_train/"
    #     curr_labels = os.listdir(ori_dir)
    #
    #     img_dim = 19
    #     gen_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/GAN+/unfiltered/" + b + "/"
    #     new_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/GAN+/filtered/" + b + "/"
    #     size = []
    #
    #
    #     for i in range(len(curr_labels)):
    #         df = abs_diff_gen(ori_dir + curr_labels[i], gen_dir + curr_labels[i], img_dim, othreshold)
    #
    #         size.append(len(df))
    #         for k in range(len(df)):
    #             if not os.path.exists(new_dir + curr_labels[i]):
    #                 os.makedirs(new_dir + curr_labels[i])
    #             shutil.copy(gen_dir + curr_labels[i] + "/" + df[k], new_dir + curr_labels[i] + '/' + df[k])
    #     print(b, size)




