# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 19:35:52 2022

@author: noxtu
"""
import numpy as np
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch
import sys
sys.path.append("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/UJI/")
from pytorchtools import EarlyStopping
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


### Code flow ###
# =============================================================================
# 1. data_and_label(img folder directory) will retrieve image path, labels and image name 
# 2. The labels will be passed into scale_label to subtract from origin
# 3. path_and_scaled_labels will then return the paths and scaled labels of training, validation, and testing dataset
# 4. norm_image will return the mean and std required for normalising images (separately by training, validation, testing)
# 5. load_image will load the images and normalize them using step 4 into dataloader
# 6. normalize_input_and_load uses step 4 & 5 to return trainloader, validloader, and testloader
# 7. define resnet18 model
# 8. Train the model and save the model state
# 9. predict_result takes in testing dataset, and using the saved model state, return predicted and true labels
# 10. prediction_error will return the ED based on predicted output vs true labels
# =============================================================================

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
    
def uji_id(train_csv_dir):
    df = pd.read_csv(train_csv_dir, header=0)
    unique = df.groupby(["BUILDINGID","FLOOR"]).size().reset_index().rename(columns={0:'count'})
    unique_loc = np.array(unique.iloc[:,0:2])
    loc = []
    for i in range(len(unique_loc)):
        loc.append("b"+str(unique_loc[i][0]) + "f" + str(unique_loc[i][1]))
    return loc 
    
def data_and_label(img_folder):
    '''
    
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

def scale_label(train_label, valid_label, test_label):
    """
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

    Purpose: Scale dataset labels before training any model. 
    Output coordinates is scaled by subtracting the origin of the room (still in meters, hence does not need to convert back after prediction model)
    """    
    origin = np.amin(train_label,axis=0)
    
    scaled_train_labels = train_label - origin
    scaled_test_labels = test_label - origin
    scaled_valid_labels = valid_label - origin
    
    return scaled_train_labels, scaled_valid_labels, scaled_test_labels

def path_and_scaled_labels(train_dir, valid_dir, test_dir):
    '''

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
    train_path , train_label, _ = data_and_label(train_dir)
    valid_path , valid_label, _ = data_and_label(valid_dir)
    test_path , test_label, _ = data_and_label(test_dir)
    
    train, valid, test = scale_label(train_label, valid_label, test_label) 
    
    return train, valid, test, train_path, valid_path, test_path

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

    dataloader = DataLoader(data_set,batch_size=batch_size,shuffle=shuffle, drop_last = True, num_workers = 0, pin_memory = True)
    return dataloader

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
    train_label, valid_label, test_label, train_path, valid_path, test_path = path_and_scaled_labels(train_dir, valid_dir, test_dir)
    
    train_mean, train_std = norm_image(train_path, train_label)
    valid_mean, valid_std = norm_image(valid_path, valid_label)
    test_mean, test_std = norm_image(test_path, test_label)
   
    trainloader = load_image(train_path, train_label, batch_size, train_mean, train_std, True)
    validloader = load_image(valid_path, valid_label, batch_size, valid_mean, valid_std, True)
    testloader = load_image(test_path, test_label, batch_size, test_mean, test_std, False)
    
    return trainloader, validloader, testloader

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
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #change input depth from 3 to 1
    
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
    valid_loss_min = np.Inf # set initial "min" to infinity
    early_stopping = EarlyStopping(patience= patience, verbose=True)
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
                
        train_loss = total_train_loss/len(trainloader.dataset)
        valid_loss = total_valid_loss/len(validloader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
            
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e+1, train_loss, valid_loss))
    
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_dir+state_name)
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

    Returns
    -------
    predict_output : list
        outputs of predicted coordinates
    label : list
        actual labels

    '''
    model.load_state_dict(torch.load(save_dir+model_state))
    predict_output = np.empty((0,2))
    label = np.empty((0,2))
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
    print('Average error: ',np.mean(error_NN),
          '\nMinimum error:', np.amin(error_NN), '\nMaximum error:', np.amax(error_NN), '\nVariance:', np.var(error_NN))
    result = [np.mean(error_NN),np.amin(error_NN),np.amax(error_NN),np.var(error_NN)]
    return result

def label_directory(img_path):
    '''
Purpose: Get all labels directory in a list
    Parameters
    ----------
    img_path : TYPE
        DESCRIPTION.

    Returns
    -------
    label_dir : TYPE
        DESCRIPTION.

    '''
    label_dir = []
    for root, dirs, files in os.walk(img_path, topdown=False):
        for d in dirs:
            label_dir.append(os.path.join(root, d))
    return label_dir

def result_compare_plot(result, keys, scenarios, title_name):
    naming = ['Average Error', 'Minimum Error', 'Maximum Error', 'Variance']
    plot_color = ['-or', '-ob', '-oc', '-og', '-oy', '-om', '-ok']
    key_count = 0
    num_compare = int(len(keys)/len(scenarios))

    for i in range(len(naming)):
        plt.figure(figsize=(10, 7))
        for k in range(num_compare):
            plt.plot(result[keys[key_count]],plot_color[k],label= keys[key_count])

            key_count+=1
        plt.xlabel('Model state')
        plt.ylabel(naming[i])
        plt.title(title_name)
        plt.legend(frameon=False)
        


if __name__ == '__main__':
    ##### Parameters #####
    ##### Code for single runs #####
# =============================================================================
    batch_size = 32
    lr = 0.0003 #test for 0.001, 0.0001, 0.0003
    epochs = 300
    patience = 100 #old used 30
    state_name = "0.0003_test.pt"
    save_dir = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/personal/'
    train_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/personal/b0f1_train/"
    valid_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/personal/valid_img/b0f1_valid/"
    test_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/personal/test_img/b0f1_test/"
    # save_dir = '/home/wayne/wgan/results/'
    # train_img_dir = "/home/wayne/wgan/uji_images/train_img/b0f0_train"
    # valid_img_dir = "/home/wayne/wgan/uji_images/valid_img/b0f0_valid"
    # test_img_dir = "/home/wayne/wgan/uji_images/test_img/b0f0_test"
    model = resnet18()
    ##### Normalize data input and output. Load data. #####
    trainloader, validloader, testloader = normalize_input_and_load(train_img_dir, valid_img_dir, test_img_dir, batch_size)
    train_model(trainloader, validloader, model, lr, epochs, patience, state_name, save_dir)
    predict_output, label = predict_result(state_name,model,testloader, save_dir)
    result = prediction_error(predict_output, label)
    print(result)
# =============================================================================
    
    
    ##### Code for multiple runs #####
    ##### Parameters #####
# =============================================================================
#     num_train = 5
#     batch_size = 32
#     lr = 0.0003
#     epochs = 300 
#     patience = 100 #old used 30
#     save_dir = '/home/wayne/uji/results/'
#     # save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/vpn/"
#     model = resnet18()
#     ### 4 types of training data for each unique building and floor [e.g. b0f0_original, b0f0_wgan, b0f0_original_wgan, b0f0_mix]
# # =============================================================================
# #   Image directory structure
# # uji_images
# #     train_img > b0f0_train, b0f0_wgan, b0f0_original_wgan, b0f0_mix
# #     valid_img > b0f0_valid, b0f1_valid
# #     test_img > b0f0_test, b0f1_test
# # =============================================================================
# # =============================================================================
# #     train_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/uji_images_vpn/train_img/"
# #     valid_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/uji_images_vpn/valid_img/"
# #     test_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/uji_images_vpn/test_img/"
# # =============================================================================
#     train_dir = "/home/wayne/uji/images/train_img/"
#     valid_dir = "/home/wayne/uji/images/valid_img/"
#     test_dir = "/home/wayne/uji/images/test_img/"
#     
# # =============================================================================
# #     train_img_dir = os.listdir(train_dir)
# #     valid_img_dir = os.listdir(valid_dir)
# #     test_img_dir = os.listdir(test_dir)
# # =============================================================================
#     train_img_dir = ['b0f0_extendedWGAN+','b0f1_extendedWGAN+','b0f2_extendedWGAN+','b0f3_extendedWGAN+','b1f0_extendedWGAN+','b1f1_extendedWGAN+','b1f2_extendedWGAN+','b1f3_extendedWGAN+',
#                      'b2f0_extendedWGAN+','b2f1_extendedWGAN+','b2f2_extendedWGAN+','b2f3_extendedWGAN+','b2f4_extendedWGAN+']
#     valid_img_dir = ['b0f0_valid','b0f1_valid','b0f2_valid','b0f3_valid','b1f0_valid','b1f1_valid','b1f2_valid','b1f3_valid','b2f0_valid','b2f1_valid','b2f2_valid','b2f3_valid','b2f4_valid']
#     test_img_dir = ['b0f0_test','b0f1_test','b0f2_test','b0f3_test','b1f0_test','b1f1_test','b1f2_test','b1f3_test','b2f0_test','b2f1_test','b2f2_test','b2f3_test','b2f4_test']
#     
# 
#     for img_folder in range(len(train_img_dir)):
#      
#         for i in range(num_train):
#             state_name = str(lr)+'_'+ str(i)+'_'+train_img_dir[img_folder]+'.pt' #e.g. 0.001_0_b0f0_wgan.pt
#          
#             trainloader, validloader, testloader = normalize_input_and_load(train_dir + train_img_dir[img_folder], 
#                                                                             valid_dir + valid_img_dir[img_folder], 
#                                                                             test_dir + test_img_dir[img_folder], batch_size)
#             
#             train_model(trainloader, validloader, model, lr, epochs, patience, state_name, save_dir)
# =============================================================================
           

    
    ##### Prediction Error #####
# =============================================================================
#     num_train = 5
#     batch_size = 24 #must check from test folder with smallest number of labels
#     result = [] ### order: mean, min, max, var
#     state_details = []### order: lr, train_num, fid, case
#     pred = []
#     
#     lr = 0.0003
#     case_name = ['_extendedGAN+']#['_train', '_wgan', '_original_wgan', '_mix']
#     # train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/csv_files/UJI-trainingData.csv"
#     save_dir = "/home/wayne/uji/results/"
#     floor_id = ['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3', 'b2f0', 'b2f1', 'b2f2', 'b2f3', 'b2f4']
#      
# 
#     train_dir = "/home/wayne/uji/images/extendedGAN+/"
#     valid_dir = "/home/wayne/uji/images/valid_img/"
#     test_dir = "/home/wayne/uji/images/test_img/"
#     #solve the testloader
#     next_list = 0
#     curr_index = 0
# 
#     model = resnet18()
#     ### csv file: lr, train_num, fid, case, mean, min, max, var
#  
#     for t in range(num_train):
#         for fid in floor_id:
#             print(fid)
#             curr_test = test_dir + fid + "_test"
#             curr_valid = valid_dir + fid + "_valid"
#             for case in case_name:
#                 curr_train = train_dir + fid + case
#                 ### For b1f0 and b2f0 unknown error ###
#                 
#                 #######
#                 _, train_label, _ = data_and_label(curr_train)
#                 origin = np.amin(train_label,axis=0)
#                 _, _, testloader = normalize_input_and_load(curr_train, curr_valid, curr_test, batch_size)
#                 state_name = str(lr)+'_'+str(t)+'_'+fid+case+'.pt'
#                 predict_output, label = predict_result(state_name, model, testloader, save_dir)
#                 for size in range(len(predict_output)):
#                     pred.append([fid,(predict_output[size][0]+origin[0]), (predict_output[size][1]+origin[1]), (label[size][0]+origin[0]), (label[size][1]+origin[1])])
#                 result.append(prediction_error(predict_output, label))
#                 state_details.append([lr, t, fid, case])
#     df = pd.DataFrame(state_details, columns = ['LR','TRAIN_COUNT','FID','CASE'])
#     df[['MEAN', 'MIN', 'MAX', 'VAR']] = pd.DataFrame(result)
#     pred_df = pd.DataFrame(pred, columns = ['FID', 'PREDICTED_LONGITUDE','PREDICTED_LATITUDE', 'ACTUAL_LONGITUDE','ACTUAL_LATITUDE'])
# 
#     df.to_csv('/home/wayne/uji/uji_results.csv', index=False)
#     pred_df.to_csv('/home/wayne/uji/pred.csv')
# =============================================================================
    
  ### Run this first to combine ED of all floors together ###
# =============================================================================
#   1. Label all the learning rates 
#   2. State the number of training states per learning rate and case
#   3. Change the floor id (e.g. for building only list down all [b1f0,b1f1,b1f2,b1f3])
#   4. Change the result_dir for the new csv file
#   5. Change the main result.csv file
# =============================================================================
    ##### Combine results #####
    # lr = [0.0003]#[0.001, 0.0001, 0.0003]
    # num_times = 5
    # case_name = ['_extendedGAN+']#['_train', '_wgan', '_original_wgan', '_mix']
    # floor_id = ['b0f0', 'b0f1', 'b0f2', 'b0f3']
    # result_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/results/extendedGAN+/b0_results.csv"
    # df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/results/extendedGAN+/uji_results.csv", header=0)
    # #need to combine same lr, num and case name, then divide by total combined
    # building_df = pd.DataFrame()
    # for i in floor_id:
    #     temp = df[df['FID'] == i]
    #     building_df = building_df.append(temp, ignore_index=True)
    #
    # temp_id = []
    # temp_avg = []
    # for case in case_name:
    #     for l in lr:
    #         for num in range(num_times):
    #             temp_df= building_df[(building_df['LR'] == l) & (building_df['TRAIN_COUNT'] == num)
    #                                  & (building_df['CASE'] == case)]
    #             temp_id.append([case, l, num])
    #             temp_avg.append(temp_df[['MEAN','MIN','MAX','VAR']].mean(axis=0).tolist())
    #
    # df = pd.DataFrame(temp_id, columns = ['CASE', 'LR', 'TRAIN_COUNT'])
    # df[['MEAN', 'MIN', 'MAX', 'VAR']] = pd.DataFrame(temp_avg)
    # df.to_csv(result_dir, index =False)
    
  ### Run this to obtain a graph comparison between the same LR ###
    ##### Plotting results #####
    #each graph shows the same lr, plot name is the case name
    #for individual floors
# =============================================================================
#     df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/UJI/csv_files/uji_results.csv", header=0)
#     df = df[df['FID'] == 'b2f4']
#     bid = "Building 2 Floor 4"
# =============================================================================
    
    #for grouping buildings
# =============================================================================
#     df = pd.read_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/results/extendedGAN+/uji_results.csv", header = 0)
#     bid = "Building 0"
#     lr = [0.0003]
#     case_name = ['_extendedGAN+']#['_train', '_wgan', '_original_wgan', '_mix']
#     scenarios = ['_avg', '_min', '_max', '_var']
#     plot_dict = {}
#     
#     for l in lr:
#         for case in case_name:
#             selected_df = df[(df['LR'] == l) & (df['CASE'] == case)]
# 
#             plot_dict[str(l)+case+scenarios[0]] = selected_df['MEAN'].tolist()
#             plot_dict[str(l)+case+scenarios[1]] = selected_df['MIN'].tolist()
#             plot_dict[str(l)+case+scenarios[2]] = selected_df['MAX'].tolist()
#             plot_dict[str(l)+case+scenarios[3]] = selected_df['VAR'].tolist()
# 
#     for l in lr:
#         dict_keys = []
#         for sc in scenarios:
#             for case in case_name:
#                 dict_keys.append(str(l)+case+sc)
#         result_compare_plot(plot_dict, dict_keys, scenarios, bid)
# 
# =============================================================================
    
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   