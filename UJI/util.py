# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:54:01 2022

@author: noxtu
"""
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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

def imshow(imgs):
    #print images onto plot (ensure that it is denormalized)
    imgs = torchvision.utils.make_grid(imgs, normalize=False)
    npimgs = imgs.numpy()
    plt.figure(figsize = (8,8), frameon=False)
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap = "Greys_r")
    plt.xticks([])
    plt.yticks([])
  
    
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
        

def ori_img(num_img, img_dir):
    #plot image from saved folder
    for i in range(num_img):
        img = Image.open(img_dir+str(i)+".png")
        plt.figure(figsize = (8,8), frameon=False)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

def abs_diff_gen(ori_dir, gen_dir, img_dim, ori_diff):
    #Loading data
    
    ori_dataloader, _,_,_,_= load_by_label(ori_dir, 1, False)
    gen_dataloader, _, _, gen_img_name, gen_label= load_by_label(gen_dir, 1, False)
    #Extract tensors of images and turn it into array
    ori_list = []
    gen_list = []
    gen_abs_diff_dict = {}
    for batch_idx, (data,_) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
    for batch_idx, (data,_) in enumerate(gen_dataloader):
        gen_list.append(data.numpy())
    ori_list = np.array(ori_list).reshape((-1,img_dim,img_dim))
    gen_list = np.array(gen_list).reshape((-1,img_dim,img_dim))
    #Find the abs diff of each generated images against all original images
    for k in range(len(gen_list)):
        result = [] 
        diff = []
        for ori in range(len(ori_list)):
            result.append(np.absolute(ori_list[ori] - gen_list[k]))
        for i in range(len(result)):
            diff.append(np.array(result[i]).sum())
        gen_abs_diff_dict[gen_img_name[k]] = diff


    #Find the min diff for each generated images [list contains min diff in image order (e.g. 0.png, 1.png...)]
    min_list = []
    img_name = []
    for i in range(len(gen_list)):
        # min_list.append(sum(gen_abs_diff_dict[str(i)+'.png'])/len(gen_abs_diff_dict[str(i)+'.png']))
        if 'wgan_'+str(i)+'.png' in gen_abs_diff_dict:
            min_list.append(min(gen_abs_diff_dict['wgan_'+str(i)+'.png']))
            img_name.append('wgan_'+str(i)+'.png')
    dict_col = {'img_name': img_name, 'abs_score': min_list}
    df = pd.DataFrame(dict_col)
    df = df[df['abs_score'] <= ori_diff]
    df = df.sort_values(by=['abs_score'], ignore_index=True)
    dfa = np.array(df.iloc[:30,0])
# =============================================================================
#     for rename in range(20):
#         new_name = 'new_'+str(rename)+'.png'
#         os.rename(gen_dir + '/' +df.iloc[rename,0], gen_dir +'/'+new_name)
# =============================================================================
    return dfa

def abs_diff_ori(ori_dir, img_dim):
    # ori_label_dir = label_directory(ori_dir)
    # ori_dataloader, _,_,_,_= load_by_label(ori_label_dir[0], 1, False)
    ori_dataloader, _,_,_,_= load_by_label(ori_dir, 1, False)
    ori_list = []
    for batch_idx, (data,_) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
    ori_list = np.array(ori_list).reshape((-1,img_dim,img_dim))
    
    result = []
    diff = []
    for ori in range(len(ori_list)):
        for i in range(len(ori_list)-(ori+1)):
            result.append(np.absolute(ori_list[ori] - ori_list[ori+i+1]))
            
        for i in range(len(result)):
            diff.append(np.array(result[i]).sum())
        
    max_diff = max(diff)
    # avg_diff = sum(diff)/len(diff)
    return max_diff

def ED_gen(ori_dir, gen_dir):
    ori_label_dir = label_directory(ori_dir)
    ori_dataloader, _,_,_,_= load_by_label(ori_label_dir[0], 1, False)
    gen_label_dir = label_directory(gen_dir)
    gen_dataloader, _, _, gen_img_name, gen_label= load_by_label(gen_label_dir[0], 1, False)
    
    #Extract tensors of images and turn it into array
    ori_list = []
    gen_list = []
    ED = {}
    for batch_idx, (data,_) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
    for batch_idx, (data,_) in enumerate(gen_dataloader):
        gen_list.append(data.numpy())
    #Find the abs diff of each generated images against all original images
    for k in range(len(gen_list)):
        result = [] 
        for ori in range(len(ori_list)):
            result.append(np.linalg.norm(ori_list[ori] - gen_list[k]))
            
        ED[gen_img_name[k]] = min(result)
    
    df = pd.DataFrame(ED.items(), columns = ['img_name', 'ED_score'])
    df = df.sort_values(by=['ED_score'],ignore_index=True)
    return df
        
def ED_ori(ori_dir):
    ori_label_dir = label_directory(ori_dir)
    ori_dataloader, _,_,ori_img_name,_= load_by_label(ori_label_dir[0], 1, False)
    ori_list = []
    
    for batch_idx, (data,_) in enumerate(ori_dataloader):
        ori_list.append(data.numpy())
        
    result = []
    for ori in range(len(ori_list)):
        for i in range(len(ori_list)-(ori+1)):
            result.append(np.linalg.norm(ori_list[ori] - ori_list[ori+i+1]))
        ED = max(result)
    return ED
