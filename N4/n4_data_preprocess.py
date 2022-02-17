# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:01:25 2022

@author: noxtu
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


def n4_id(train_csv_dir):
    '''
    Purpose: Getting unique locations x,y 

    Parameters
    ----------
    train_csv_dir : string
        directory of training dataset

    Returns
    -------
    unique_loc : 2d list
        contains [[buildingID, floorID]]

    '''
    df = pd.read_csv(train_csv_dir, header=0)
    unique = df.groupby(["x", "y"]).size().reset_index().rename(columns={0:'count'})
    unique_loc = np.array(unique.iloc[:,0:2])
    return unique_loc 

def find_max(train_csv_dir, test_csv_dir):
    train_df = pd.read_csv(train_csv_dir, header=0)
    test_df = pd.read_csv(test_csv_dir, header=0)
    train_rssi = train_df.iloc[:,:301].to_numpy()
    test_rssi = test_df.iloc[:,:301].to_numpy()
    train_max = train_rssi.max()
    test_max = test_rssi.max()
    if train_max > test_max: 
        print("train: ", train_max)
        return train_max
  
    else:
        print("test: ", test_max)
        return train_max

def generate_image(train_csv_dir, test_csv_dir, img_dim, save_dir, csv_dir, split, max_rssi, train_csv_name):
    '''
    Purpose: Generate images for training localisation model and for augmentation

    Parameters
    ----------
    train_csv_dir : string
        training csv file directory
    test_csv_dir : string
        testing csv file directory
    img_dim : int
        image size
    save_dir : string
        directory of images to be saved
    csv_dir : string
        directory of csv files to be saved
    split : boolean, optional
        if true, generate images for localisation. if false, generate images for augmentation The default is False.

    Returns
    -------
    None.

    '''  
    
    ### Create directory path to save images
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
         
    train_df = pd.read_csv(train_csv_dir, header = 0)
    test_df = pd.read_csv(test_csv_dir, header = 0)
    train_df = train_df.iloc[:,:-25]
    test_df = test_df.iloc[:,:-25]
    train_df = train_df.drop(['time'],axis=1)
    test_df = test_df.drop(['time'],axis=1)
    ### Create extra columns to fit the image dimension
    img_size_diff = (img_dim*img_dim)-301
    for diff in range(img_size_diff):
        train_df.insert(301,"P"+str(diff), -110)
        test_df.insert(301,"P"+str(diff), -110)
        
    if (split == True):
            
        train, valid = train_test_split(train_df, test_size = 0.2)
    
        train.to_csv(csv_dir+"/"+train_csv_name+"_train.csv", index=False)
        valid.to_csv(csv_dir+"/"+train_csv_name+"_valid.csv", index=False)
        
        
        unique_loc_train = train.groupby(["x","y"]).size().reset_index().rename(columns={0:'count'})
        unique_loc_train = np.array(unique_loc_train.iloc[:,:2])
        unique_loc_valid = valid.groupby(["x","y"]).size().reset_index().rename(columns={0:'count'})
        unique_loc_valid = np.array(unique_loc_valid.iloc[:,:2])
        
        for unique in range(len(unique_loc_train)):
            rssi = train[(train["x"] == unique_loc_train[unique][0]) & 
                         (train["y"] == unique_loc_train[unique][1])]
            rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
            if not os.path.isdir(save_dir+"/train_img/"+train_csv_name+"_train/{}_{}".format(unique_loc_train[unique][0]
                                                                                     ,unique_loc_train[unique][1])):
                os.makedirs(save_dir+"/train_img/"+train_csv_name+"_train/{}_{}".format(unique_loc_train[unique][0],
                                                                                unique_loc_train[unique][1]))
            for k in range(len(rssi)):
                plt.imsave(save_dir+"/train_img/"+train_csv_name+"_train/{}_{}/{}.png".format(unique_loc_train[unique][0],
                                                                                      unique_loc_train[unique][1],k),
                           rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
            for unique in range(len(unique_loc_valid)):
                rssi = valid[(valid["x"] == unique_loc_valid[unique][0]) & 
                             (valid["y"] == unique_loc_valid[unique][1])]
                rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
                if not os.path.isdir(save_dir+"/valid_img/"+train_csv_name+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                         unique_loc_valid[unique][1])):
                    os.makedirs(save_dir+"/valid_img/"+train_csv_name+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                    unique_loc_valid[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/valid_img/"+train_csv_name+"_valid/{}_{}/{}.png".format(unique_loc_valid[unique][0],
                                                                                          unique_loc_valid[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
            
     
        unique_loc = test_df.groupby(["x","y"]).size().reset_index().rename(columns={0:'count'})
        unique_loc = np.array(unique_loc.iloc[:,:2])
        for unique in range(len(unique_loc)): #by coord
            rssi = test_df[(test_df["x"] == unique_loc[unique][0]) & (test_df["y"] == unique_loc[unique][1])]
            rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
  
            if not os.path.isdir(save_dir+"/test_img/"+train_csv_name+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                os.makedirs(save_dir+"/test_img/"+train_csv_name+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
            for k in range(len(rssi)):
                plt.imsave(save_dir+"/test_img/"+train_csv_name+"_test/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                           rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
    else:
        unique_loc = train_df.groupby(["x","y"]).size().reset_index().rename(columns={0:'count'})
        unique_loc = np.array(unique_loc.iloc[:,:2])
        for unique in range(len(unique_loc)): #by coord
            rssi = train_df[(train_df["x"] == unique_loc[unique][0]) & (train_df["y"] == unique_loc[unique][1])]
            rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
  
            if not os.path.isdir(save_dir+"/train_only/"+train_csv_name+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                os.makedirs(save_dir+"/train_only/"+train_csv_name+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
            for k in range(len(rssi)):
                plt.imsave(save_dir+"/train_only/"+train_csv_name+"/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                           rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
            
            
            
            
    # print(unique_loc[][])
    
        
if __name__ == "__main__":

### Generate images
    ### Input Parameters ###
    train_file = ['trainingData_F1Sa', 'trainingData_F1Sb', 'trainingData_F2Sa', 'trainingData_F2Sb']
    test_file = ['testingData_F1Sa', 'testingData_F1Sb','testingData_F2Sa','testingData_F2Sb']
    csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/N4/csv_files/"
    img_dim = 18
    img_save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/N4/images/"
    csv_save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/N4/csv_files/n4_split/"
    
    #Find max rssi value
    max_val= -110
    for i in range(len(train_file)):
        train_csv_dir = csv_dir+train_file[i]+".csv"
        test_csv_dir = csv_dir+test_file[i]+".csv"
        curr_max = find_max(train_csv_dir, test_csv_dir)
        if curr_max > max_val:
            max_val = curr_max
    
    #Generate images
    for i in range(len(train_file)):
        train_csv_dir = csv_dir+train_file[i]+".csv"
        test_csv_dir = csv_dir+test_file[i]+".csv"
        
        generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_save_dir, True, max_val, train_file[i])
        generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_save_dir, False, max_val, train_file[i])


    



















