import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import csv
from matplotlib import pyplot as plt
from torchvision import transforms, models
import shutil
from PIL import Image
from pytorchtools import EarlyStopping
import torchvision
import math

'''
Loading data and utils to filter images
'''
class UnNormalize(object):
    '''
    To denormalize data images
    '''
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
    '''
    For loading of images to dataloader
    '''
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
    img_path : string
        image directory

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
    img_name: list
        contains image names

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
    Purpose: normalize images

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
    Purpose: loading images to dataloader
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
    Purpose: loading data by individual labels and retrieve the relevant information for usage (for absolute difference score)
    Parameters
    ----------
    label_dir: string
        label directory
    batch_size: int
        batch size
    shuffle: boolean
        True or false

    Returns
    -------
    dataloader: dataloader
    mean: float
    std: float
    img_name: list
    label: list

    '''
    img_path, label, img_name = data_and_label(label_dir)
    mean, std = norm_image(img_path, label)
    dataloader = load_image(img_path, label, batch_size, mean, std, shuffle)
    return dataloader, mean, std, img_name, label

def abs_diff_gen(ori_dir, gen_dir, img_dim, ori_diff):
    '''
    Purpose: Find the min difference between generated image and local threshold
    Parameters
    ----------
    ori_dir: string
        original images directory
    gen_dir: string
        generated images directory
    img_dim: int
        image dimension
    ori_diff: df
        contains df of local threshold

    Returns
    -------

    '''
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
        if str(i) + '.png' in gen_abs_diff_dict:
            min_list.append(min(gen_abs_diff_dict[str(i) + '.png']))
            img_name.append(str(i) + '.png')
    dict_col = {'img_name': img_name, 'abs_score': min_list}
    df = pd.DataFrame(dict_col)
    df = df[df['abs_score'] <= (ori_diff)]
    dfa = np.array(df.iloc[:len(df), 0])

    return dfa

def abs_diff_ori(ori_dir, img_dim):
    '''
    Purpose: Find local threshold for absolute difference score
    Parameters
    ----------
    ori_dir: string
        original images directory
    img_dim: int
        image dimension

    Returns
    -------
    max_diff: list contains max difference between each images in the label
    '''
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

def scale_label(train_label, valid_label, test_label):
    """
    Purpose: Scale dataset labels before training any model.
    Output coordinates is scaled by subtracting the origin of the room (still in meters, hence does not need to convert back after prediction model)

    Parameters
    ----------

    train_label : array
        contains training labels(longitude and latitude) dataset
    valid_label : array
        contains validation labels(longitude and latitude) dataset
    test_label : array
        contains testing labels(longitude and latitude) dataset

    Returns training, validation, and testing coordinate labels in an array
    -------
    """
    origin = np.amin(train_label, axis=0)

    scaled_train_labels = train_label - origin
    scaled_test_labels = test_label - origin
    scaled_valid_labels = valid_label - origin

    return scaled_train_labels, scaled_valid_labels, scaled_test_labels

def path_and_scaled_labels(train_dir, valid_dir, test_dir):
    '''
    Purpose: data normalization
    Parameters
    ----------
    train_dir : string
        training dataset images directory
    valid_dir : string
        validation dataset images directory
    test_dir : string
        testing dataset images directory

    Returns
    -------
    train : array
        scaled labelled training dataset coordinates
    valid : array
        scaled labelled validation dataset coordinates
    test : array
        scaled labelled testing dataset coordinates
    train_path : list
        path of all training dataset image name
    valid_path : list
        path of all validation dataset image name
    test_path : list
        path of all testing dataset image name

    '''
    train_path, train_label, _ = data_and_label(train_dir)
    valid_path, valid_label, _ = data_and_label(valid_dir)
    test_path, test_label, _ = data_and_label(test_dir)

    train, valid, test = scale_label(train_label, valid_label, test_label)

    return train, valid, test, train_path, valid_path, test_path

def normalize_input_and_load(train_dir, valid_dir, test_dir, batch_size):
    '''

    Parameters
    ----------
    train_dir : string
        training dataset images directory
    valid_dir : string
        validation dataset images directory
    test_dir : string
        testing dataset images directory
    batch_size : int
        batch size

    Returns training, validation, and testing loader
    -------

    '''
    train_label, valid_label, test_label, train_path, valid_path, test_path = path_and_scaled_labels(train_dir,
                                                                                                     valid_dir,
                                                                                                     test_dir)

    train_mean, train_std = norm_image(train_path, train_label)
    valid_mean, valid_std = norm_image(valid_path, valid_label)
    test_mean, test_std = norm_image(test_path, test_label)

    trainloader = load_image(train_path, train_label, batch_size, train_mean, train_std, True)
    validloader = load_image(valid_path, valid_label, batch_size, valid_mean, valid_std, True)
    testloader = load_image(test_path, test_label, batch_size, test_mean, test_std, False)

    return trainloader, validloader, testloader


'''
GAN+ code without pre-training (taken from previous GAN+ code)
'''
class GAN_Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim, im_dim, hidden_dim=128):
        super(GAN_Generator, self).__init__()
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
        nn.LeakyReLU(0.2, inplace=True),
        #### END CODE HERE ####
    )

class GAN_Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_dim, hidden_dim=128):
        super(GAN_Discriminator, self).__init__()
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
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #### END CODE HERE ####
    )

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
    fake_loss = criterion(disc(fake_images), torch.zeros((num_images, 1), device=device))
    real_loss = criterion(disc(real), torch.ones((num_images, 1), device=device))
    disc_loss = (fake_loss + real_loss) / 2.0

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
    noises = get_noise(num_images, z_dim, device=device)
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
    return torch.randn(n_samples, z_dim, device=device)
    #### END CODE HERE ####


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

def GAN_train(train_csv_dir,dirich_csv_dir, num_ap, lb_x_name, lb_y_name, save_model_dir, z_dim, lr, n_epochs):
    '''
    Purpose: GAN training
    Parameters
    ----------
    train_csv_dir: string
        training dataset csv directory
    dirich_csv_dir: string
        dirichlet dataset csv directory
    num_ap: int
        total APs
    lb_x_name: string
        x label name
    lb_y_name: string
        y label name
    save_model_dir: string
        model state directory
    z_dim: int
        latent noise
    lr: float
        learning rate
    n_epochs: int
        epochs

    Returns
    -------

    '''
    count = 1
    df = pd.read_csv(train_csv_dir, header=0)
    df = df.replace(100, -110)
    col = ["AP" + str(i) for i in range(1, num_ap + 1)]
    col.append(lb_x_name)
    col.append(lb_y_name)
    df.columns = col
    dirich = pd.read_csv(dirich_csv_dir, header=0)
    df = df.append(dirich, ignore_index=True)
    training_data_csv_in = np.array(df.iloc[:, :num_ap])
    training_data_csv_lb = np.array(df.iloc[:, -2:])

    if torch.cuda.is_available():
        training_data_csv_in = torch.from_numpy(training_data_csv_in).cuda()
        training_data_csv_lb = torch.from_numpy(training_data_csv_lb).cuda()

    store = {}
    for i in range(len(training_data_csv_in)):
        k = training_data_csv_lb[i].tolist()
        v = training_data_csv_in[i].tolist()

        if tuple(k) in store:

            store[tuple(k)].append(v)
        else:
            store[tuple(k)] = [v]
    print("Size of unique RP: ", len(store.keys()))
    # kk = list(store)
    # checklist=[]
    # for u in range(60,113):
    #     checklist.append(kk[u])
    for unique in store.keys():
        # if unique in checklist:
            print("Training {}/{}: ".format(count,len(store.keys())))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gen = GAN_Generator(z_dim, num_ap).to(device)
            disc = GAN_Discriminator(num_ap).to(device)
            # gen.load_state_dict(torch.load(gen_state))
            # disc.load_state_dict(torch.load(disc_state))

            if torch.cuda.is_available():
                gen = gen.cuda()
                disc = disc.cuda()
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
            gen_save_state = save_model_dir+str(unique[0])+"_"+str(unique[1])+"_gan+.pt"
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
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

def generate_img_csv(z_dim,num_ap,count,path,generate_csv_dir, lb_x_name, lb_y_name):
    '''
    Purpose: Generate GAN RSSI and save to csv file
    Parameters
    ----------
    z_dim: int latent noise
    num_ap: int number of AP
    count: int number of samples to generate per RP
    path: string path of saved model state
    generate_csv_dir: string path of newly generated csv file
    lb_x_name: string x label name
    lb_y_name: string y label name

    Returns
    -------

    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = []
    for ap in range(num_ap):
        col.append("AP" + str(ap + 1))
    col.extend([lb_x_name, lb_y_name])
    with open(r'{}'.format(generate_csv_dir), 'a',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(col)

    state = os.listdir(path)
    for s in state:
        st = s.split("_")
        gen = GAN_Generator(z_dim, num_ap).to(device)
        gen.load_state_dict(torch.load(path+s))
        gen.eval()
        info = []
        for i in range(count):
            fake_noise = get_noise(1, z_dim, device=device)
            fake = gen(fake_noise)
            fake = rpreprocess(fake[0]).cpu().detach().tolist()
            fake.extend([st[0], st[1]])
            info.append(fake)
        with open(r'{}'.format(generate_csv_dir),'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(info)

def preprocess_dirichlet(csv_file, num_ap, lb_x_name, lb_y_name):
    '''
    Obtain dictionary of csv file, where dict = {(longitude,latitude): [[sample1][sample2]]}
    Parameters
    ----------
    csv_file: string of csv path
    fid: int of floor id
    bid: int of building id
    num_ap: number of ap in dataset

    Returns: dictionary
    -------

    '''
    #Dataset
    df = pd.read_csv(csv_file, header=0)
    unique_loc = df.groupby([lb_x_name, lb_y_name]).size().reset_index().rename(columns={0: 'count'})
    labels = np.array(unique_loc.iloc[:,0:2])
    data_map = {} #keys = (longitude, latitude), access by sample_map[labels[0][0],labels[0][1]]
    for label in labels:
        rp = df[(df[lb_x_name] == label[0]) & (df[lb_y_name] == label[1])]
        rp = np.array(rp.iloc[:,:num_ap])
        data_map[label[0],label[1]] = rp
    return data_map

def generate_image(data_dict, img_dim, num_ap, max_rssi, image_path):
    '''
    Generate images
    Parameters
    ----------
    data_dict: dictionary of bfid
    img_dim: image size to generate
    num_ap: total ap of dataset
    max_rssi: max rssi from max_rssi()
    image_path: string of path to save generated images

    Returns
    -------

    '''
    keys = list(data_dict.keys())
    for rp in range(len(data_dict)):
        rssi = data_dict[keys[rp]]
        empty = np.full((len(rssi),(img_dim*img_dim)-num_ap), -110)
        rssi = np.concatenate((rssi, empty), axis=1)
        if not os.path.exists(image_path+"/"+str(keys[rp][0])+"_"+str(keys[rp][1])):
            os.makedirs(image_path+"/"+str(keys[rp][0])+"_"+str(keys[rp][1]))
        for size in range(len(rssi)):
            img = rssi[size].reshape(img_dim, img_dim)
            plt.imsave(image_path+"/"+str(keys[rp][0])+"_"+str(keys[rp][1])+"/{}.png".format(size), img, vmin=-110, vmax=max_rssi, cmap="gray")



'''
Localisation 
'''
def resnet18():
    '''

    Returns
    -------
    resnet18 : model
        structure of the  model

    '''
    resnet18 = models.resnet18()
    n_inputs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(n_inputs, 2)  # change last layer to 2 outputs
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)  # change input depth from 3 to 1

    return resnet18


def train_model(trainloader, validloader, model, lr, epochs, patience, state_name, save_dir):
    '''

    Parameters
    ----------
    trainloader : tensor
        contains training data
    validloader : tensor
        contains validation data
    model : model
        contains the training model
    lr : float
        learning rate
    epochs : int
        number of epochs
    patience : int
        patience for early stopping
    state_name : string
        saving model state
    save_dir: string
        model state directory

    Returns
    -------
    None.

    '''
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, valid_losses = [], []
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    torch.backends.cudnn.benchmark = True

    for e in range(epochs):
        total_train_loss = 0
        total_valid_loss = 0
        model.train()
        for data, target in trainloader:
            data = data.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.detach()

        model.eval()
        with torch.no_grad():
            for data, target in validloader:
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = criterion(output, target)
                total_valid_loss += loss.detach()

        train_loss = total_train_loss / len(trainloader.dataset)
        valid_loss = total_valid_loss / len(validloader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e + 1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_dir + state_name)
            valid_loss_min = valid_loss

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def predict_result(model_state, model, testloader, save_dir):
    '''

    Parameters
    ----------
    model_state : string
        saved model state name
    model : model
        structure of training model
    testloader : tensor
        contains testing data
    save_dir: string
        model state directory

    Returns
    -------
    predict_output : list
        outputs of predicted coordinates
    label : list
        actual labels

    '''
    model.load_state_dict(torch.load(save_dir + model_state))
    predict_output = np.empty((0, 2))
    label = np.empty((0, 2))
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        model.eval()

        for test_input, test_output in testloader:
            test_input = test_input.type(torch.FloatTensor)
            test_output = test_output.type(torch.FloatTensor)

            if torch.cuda.is_available():
                test_input, test_output = test_input.cuda(), test_output.cuda()
            predict = model(test_input)
            predict_output = np.append(predict_output, predict.cpu().numpy(), axis=0)
            label = np.append(label, test_output.cpu().numpy(), axis=0)
    return predict_output, label

def prediction_error(predict_output, label):
    '''

    Parameters
    ----------
    predict_output : array
        contains all testing prediction output
    label : array
        contains actual labels of testing dataset

    Returns
    -------
    Print out the ED errors

    '''
    num_test_samples = len(predict_output)
    error_NN = [None] * num_test_samples
    for i in range(num_test_samples):
        error_NN[i] = np.linalg.norm(predict_output[i] - label[i])
    print('Average error: ', np.mean(error_NN),
          '\nMinimum error:', np.amin(error_NN), '\nMaximum error:', np.amax(error_NN), '\nVariance:',
          np.var(error_NN))
    result = [np.mean(error_NN), np.amin(error_NN), np.amax(error_NN), np.var(error_NN)]
    return result


def prediction_and_result(test_img_dir, train_img_dir, valid_img_dir, model, save_dir, predicted_csv_dir,
                          result_csv_dir, bfid, data_name):
    pred = []
    result = []
    state_name = "0.0003_{}_{}_{}.pt".format(i, bfid,data_name)
    bsize = len(os.listdir(test_img_dir))
    _, _, testloader = normalize_input_and_load(train_img_dir, valid_img_dir, test_img_dir, bsize)

    predict_output, label = predict_result(state_name, model, testloader, save_dir)
    _, train_label, _ = data_and_label(train_img_dir)
    origin = np.amin(train_label, axis=0)
    info = [bfid, i, data_name]
    info.extend(prediction_error(predict_output, label))
    result.append(info)
    for size in range(len(predict_output)):
        pred.append([bfid, i, (predict_output[size][0] + origin[0]), (predict_output[size][1] + origin[1]),
                     (label[size][0] + origin[0]), (label[size][1] + origin[1])])
    pred_df = pd.DataFrame(pred, columns=['FID', 'TRAIN_NUM', 'PREDICTED_LONGITUDE', 'PREDICTED_LATITUDE',
                                          'ACTUAL_LONGITUDE', 'ACTUAL_LATITUDE'])
    pred_df.to_csv(predicted_csv_dir, index=False)
    rdf = pd.DataFrame(result, columns=['BFID', 'TRAIN_NUM', 'CASE', 'MEAN', 'MIN', 'MAX', 'VAR'])
    rdf.to_csv(result_csv_dir, index=False)

'''
WGAN-GP
'''
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


def wgan_gp_train(gen_model_state, disc_model_state, data_dir, lr, batch, num_epoch, img_channel, latent, critic_layer, gen_layer,
               critic_iter, gradient_p, model_choice, key):
    '''
    Purpose: Training of WGAN-GP model

    Parameters
    ----------
    gen_model_state : string
        model state of generator directory
    disc_model_state: string
        model state of critic directory
    data_dir : string
        folder name with the training input images
    lr : float
        learning rate
    batch : int
        batch size
    num_epoch : int
        number of epochs
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
    model_choice: dictionary
        selection of generator and critic models
    key: string
        N4 or NG or UJI

    Returns
    -------
    None.

    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_dir = label_directory(data_dir)
    dataloader, mean, std, _, _ = load_by_label(label_dir[0], batch)

    # =============================================================================
    #     for i in range(len(label_dir)):
    #         dataloader = load_by_label(label_dir[i], batch)
    # =============================================================================
    gen = eval(model_choice[key+'_gen'])
    critic = eval(model_choice[key+'_critic'])
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


    torch.save(gen.state_dict(), gen_model_state)
    torch.save(critic.state_dict(), disc_model_state)



def wgan_gp_pretrain(save_state, gen_model_state, disc_model_state, data_dir,
                     lr, batch, num_epoch, img_channel, latent, critic_layer, gen_layer, critic_iter,
                     gradient_p, model_choice, key):
    '''
    Purpose: Training of WGAN-GP model

    Parameters
    ----------
    save_state: string
        model state save directory
    gen_model_state: string
        generator directory
    disc_model_state: string
        critic directory
    data_dir : string
        folder name with the training input images
    lr : float
        learning rate
    batch : int
        batch size
    num_epoch : int
        number of epochs
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
    model_choice: dictionary
        selection of generator and critic models
    key: string
    NG or N4 or UJI


    Returns
    -------
    None.

    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader, mean, std, _, _ = load_by_label(data_dir, batch)
    gen = eval(model_choice[key+'_gen'])
    critic = eval(model_choice[key+'_critic'])
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
    '''
    Purpose: Generate images from WGAN-GP model states
    Parameters
    ----------
    num_iter: int
        number of images to generate
    model_state_dir: string
        model state directory
    data_dir: string
        training data directory
    latent: int
        latent noise
    gen_layer: int
        number of generator layer
    img_channel: int
        channel image
    px: int
        image size
    my_dpi: int
        dpi of screen
    save_dir: string
        directory of image to be generated and saved at

    Returns
    -------

    '''
    # generate wgan-gp images for after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = N4_Generator(latent, img_channel, gen_layer).to(device)
    gen.load_state_dict(torch.load(model_state_dir))
    gen.eval()

    _, mean, std, _, _ = load_by_label(data_dir, 1)
    unorm = UnNormalize(mean=mean, std=std)
    curr_label = data_dir.split('/')
    directory = save_dir + '/' + str(curr_label[-1])
    # directory = save_dir+'/'+curr_label
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(num_iter):
        noise = torch.randn(1, latent, 1, 1).to(device)
        fake = gen(noise).detach().cpu()
        imsave(unorm(fake), save_dir + '/' + str(curr_label[-1]) + '/' + str(i) + '.png', px, my_dpi)

def imsave(imgs, save_dir, px, my_dpi):
    #save images according to defined pixel
    imgs = torchvision.utils.make_grid(imgs, normalize=False)
    npimgs = imgs.numpy()
    fig = plt.figure(figsize = (px/my_dpi,px/my_dpi), dpi=my_dpi, frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap = "Greys_r")
    fig.savefig(save_dir, dpi=my_dpi)
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)

'''
Dirichlet
'''
def dirichlet_generate(train_df, save_dir, num_gen, num_ap):
    '''
    Purpose: Generate dirichlet csv
    Parameters
    ----------
    train_df: df
        train dataset df
    save_dir: string
        directory of dirichlet csv to be saved at
    num_gen: int
        number of samples to generate per RP
    num_ap: int
        number of APs

    Returns
    -------

    '''
    # store dictionary contains coordinates (latitude, longitude) as keys and rssi data as values (according to the number of samples belonging to coordinate) e.g. {(-7345.345, 4328596): [[AP0 ... AP520],[AP0 ... AP520]]}
    store = {}
    train_input = np.array(train_df.iloc[: ,:num_ap])
    train_label = np.array(train_df.iloc[: ,-2:])
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
    count = 0
    ### np.random.dirichlet takes N samples weights, where N is the total number of samples in RP.
    ## N Dirichlet distribution adds up to 1.

    for k in store.keys():  # number of unique RP

        sample_size = len(store[k])
        dirich_needed.append(num_gen - sample_size)

        # randomly pick samples for dirichlet
        im_new = [[0 for i in range(sample_size)] for j in range(345)]
        curr_rp_samples = np.array(store[k])
        curr_rp_samples[curr_rp_samples == 100] = -110
        curr_rp_samples = curr_rp_samples.T
        # generate images according to dirichlet needed (contains excess)
        for gen in range(math.ceil(num_gen / sample_size)):
            for s in range(sample_size):
                weight = np.random.dirichlet(np.ones(sample_size), size=1)
                for ap in range(num_ap):
                    im_new[ap][s] = np.sum(np.multiply(weight, curr_rp_samples[ap][:]))
                    if (110 + im_new[ap][s] <= 0.000001):
                        im_new[ap][s] = -110.0

            tempim.append((np.array(im_new).T).tolist())
            templb.append([k[0], k[1]])
        count += 1

    # Create column names
    col = ["AP" + str(i) for i in range(1, num_ap+1)]
    df = pd.DataFrame(tempim[0], columns=col)
    df["LATITUDE"] = templb[0][1]
    df["LONGITUDE"] = templb[0][0]
    for i in range(1, len(tempim)):
        df2 = pd.DataFrame(tempim[i], columns=col)
        df2["LATITUDE"] = templb[i][1]
        df2["LONGITUDE"] = templb[i][0]
        df = df.append(df2, ignore_index=True)

    # remove excess dirichlet generated
    col.append("LATITUDE")
    col.append("LONGITUDE")
    new_df = pd.DataFrame(columns=col)

    for key in store.keys():
        tdf = df[(df["LATITUDE"] == key[1]) & (df["LONGITUDE"] == key[0])]
        tdf = tdf.drop(tdf.index[range(len(tdf) - num_gen)])
        new_df = new_df.append(tdf, ignore_index=True)

    new_df.to_csv(save_dir, index=False)

if __name__ == "__main__":
    #########################################################################################################################
    '''
    GAN training
    '''
    #Input parameters
    fid = "F1Sb"
    train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/preprocessed/{}_train.csv".format(fid)
    dirich_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/preprocessed/{}_dirichlet.csv".format(fid)
    model_state_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/GAN+/{}/".format(fid)
    generate_gan_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/test.csv"
    lb_x_name, lb_y_name = "x", "y"
    num_ap = 301
    z_dim = 10
    epochs = 1000
    lr = 0.00001
    num_generate = 100

    ###Start of program###
    # GAN_train(train_csv_dir,dirich_csv_dir,num_ap, lb_x_name, lb_y_name, model_state_dir,z_dim,lr,epochs)
    ###End of program###
    '''
    Generate RSSI into CSV file for GAN model states
    '''
    ###Start of program###
    # generate_img_csv(z_dim, num_ap,num_generate,model_state_dir,generate_gan_dir,lb_x_name,lb_y_name)
    ###End of program###

    #########################################################################################################################
    '''
    WGAN-GP training
    '''
    model_choice = {'NG_gen': 'NG_Generator(latent, img_channel, gen_layer).to(device)', 'NG_critic': 'NG_Discriminator(img_channel, critic_layer).to(device)',
                    'N4_gen': 'N4_Generator(latent, img_channel, gen_layer).to(device)', 'N4_critic': 'N4_Discriminator(img_channel, critic_layer).to(device)',
                    'UJI_gen': 'NG_Generator(latent, img_channel, gen_layer).to(device)', 'UJI_critic': 'UJI_Discriminator(img_channel, critic_layer).to(device)'}

    key = 'N4'
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

    data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/most_sample/'  # for most labels
    gen_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/model_state/WGAN-GP/gen_most_sample.pt'
    disc_saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/model_state/WGAN-GP/disc_most_sample.pt'

    ###Start of program###
    # wgan_gp_train(gen_saved_state, disc_saved_state, data_dir, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, CHANNELS_IMG, Z_DIM, FEATURES_CRITIC, FEATURES_GEN,
    #               CRITIC_ITERATIONS, LAMBDA_GP, model_choice, key)
    ###End of program###

    '''
    WGAN-GP pretrain
    '''
    unique_loc = "F1Sb"
    data_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/train/"+unique_loc +"/"
    saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/"+unique_loc +"/"
    gen_saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/gen_most_sample.pt"
    disc_saved_state = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/disc_most_sample.pt"

    if not os.path.exists(saved_state):
        os.makedirs(saved_state)
    NUM_EPOCHS = 500
    label_dir = label_directory(data_dir)

    for i in range(len(label_dir)):
        curr_label = label_dir[i].split("/")
        save_state = saved_state + '/'+str(curr_label[-1])+'.pt'
        print(curr_label, save_state)
        wgan_gp_pretrain(save_state, gen_saved_state, disc_saved_state, label_dir[i], LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, CHANNELS_IMG,
                          Z_DIM, FEATURES_CRITIC, FEATURES_GEN, CRITIC_ITERATIONS, LAMBDA_GP, model_choice, key)

    '''
    Generate WGAN-GP images
    '''
    unique_loc = ["F1Sb"]

    for u in unique_loc:
        data_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/train/' + u + '/'
        gen_img_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/extendedGAN+/WGAN-GP/unfiltered/' + u + '/'
        saved_state = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/WGAN-GP/' + u + '/'

        ###Start of program###
        label_dir = label_directory(data_dir)
        for i in range(len(label_dir)):
            curr_label = label_dir[i].split('/')  # for windows pc
            print(curr_label)
            save_state = saved_state + '/' + str(curr_label[-1]) + '.pt'

            generate_wgan_img(num_gen, save_state, label_dir[i], Z_DIM, FEATURES_GEN, CHANNELS_IMG, IMAGE_SIZE,
                              my_dpi, gen_img_dir)
        ###End of program###
    #########################################################################################################################
    '''
    Generate images from CSV files
    '''
    #Input parameters
    img_dim = 18
    gen_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/GAN+/unfiltered/{}/".format(fid)
    dir_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/GAN+/{}_GAN+.csv".format(fid)
    max_rssi = 6

    ###Start of program###
    # dir_dict = preprocess_dirichlet(dir_csv, num_ap, lb_x_name, lb_y_name)
    # generate_image(dir_dict, img_dim, num_ap, max_rssi, gen_path)
    ###End of program###

    '''
    Finding threshold between original images
    '''
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

    '''
    Filter images using Absolute Difference Score
    '''

    # ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/train_only/{}/".format(fid)
    # curr_labels = os.listdir(ori_dir)
    #
    # threshold_df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/max_threshold/{}.csv".format(fid))
    # new_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/GAN+/filtered/{}/".format(fid)
    # size = []
    #
    # for i in range(len(curr_labels)):
    #     ori_diff = threshold_df[threshold_df["labels"] == curr_labels[i]]
    #     ori_diff = ori_diff.iloc[0, 1]
    #     df = abs_diff_gen(ori_dir + curr_labels[i], gen_path + curr_labels[i], img_dim, ori_diff)
    #     size.append(len(df))
    #     for k in range(len(df)):
    #         if not os.path.exists(new_dir + curr_labels[i]):
    #             os.makedirs(new_dir + curr_labels[i])
    #         shutil.copy(gen_path + curr_labels[i] + "/" + df[k], new_dir + curr_labels[i] + '/' + df[k])
    # print(size)

    '''
    Combining images from different folders
    '''
    # bid = ['F1Sa', 'F1Sb', 'F2Sa','F2Sb']  # ["b0f0","b0f1", "b0f2", "b0f3", "b1f0","b1f1", "b1f2", "b1f3","b2f0","b2f1", "b2f2", "b2f3", "b2f4"]
    #
    # for b in bid:
    #     folder1 = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/GAN+/localisation300/" + b + "/"
    #     folder2 = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/GAN+/filtered/" + b + "/"
    #
    #     labels = os.listdir(folder1)
    #     for i in range(len(labels)):
    #         init_img_name = os.listdir(folder1 + labels[i])
    #
    #         new_len = 300 - len(init_img_name)
    #         img_name = os.listdir(folder2 + labels[i])
    #
    #         for img in range(new_len):
    #
    #             shutil.copy(folder2 + labels[i] + "/" + img_name[img], folder1 + labels[i] + '/gan_' + img_name[img])

    #########################################################################################################################

    '''
    Localisation training
    '''
    #Input parameters
    batch_size = 32
    lr = 0.0003
    epochs = 300
    patience = 100
    bfid = 'F1Sa' #Current building-floor ID (e.g. B1F1 (UJI), F1Sa (N4), floor-1 (NG))
    state_name = "0.0003_{}_{}_train.pt".format(0,bfid) #Model state name
    save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/N4/GAN+/{}/".format(bfid) #Directory to save model state at
    train_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/GAN+/localisation150/{}/".format(bfid) #Training input images directory
    valid_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/valid/{}/".format(bfid) #Validation images directory
    test_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/test/{}/".format(bfid)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ###Start of program###
    # model = resnet18()
    # trainloader, validloader, testloader = normalize_input_and_load(train_img_dir, valid_img_dir, test_img_dir,
    #                                                                 batch_size)
    # train_model(trainloader, validloader, model, lr, epochs, patience, state_name, save_dir)
    ###End of program###

    '''
    Localisation prediction
    '''
    test_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/N4/images/original/test/{}/".format(bfid)  # Testing images directory
    data_name = "GAN+150"
    predicted_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/results/pred_{}.csv".format(data_name)
    result_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/N4/csv_files/results/result_{}.csv".format(data_name)

    ###Start of program###
    # prediction_and_result(test_img_dir, train_img_dir, valid_img_dir, model, save_dir, predicted_csv_dir, result_csv_dir, bfid, data_name)
    ###End of program###

    #########################################################################################################################

    '''
    Dirichlet
    '''
    bid = 2
    num_gen = 100
    num_ap = 345
    csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/combined_train_modified_2m.csv"

    df = pd.read_csv(csv_dir, header=0)
    df = df.replace(100, -110)
    df = df[df["FLOOR"] == bid]
    # df = df[(df.BUILDINGID == bid) & (df.FLOOR == fid)]
    save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/dirich_floor" + str(bid) + ".csv"

    ###Start of program###
    dirichlet_generate(df, save_dir, num_gen, num_ap)
    ###End of program###