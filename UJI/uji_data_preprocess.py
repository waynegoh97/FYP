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
import shutil

def uji_id(train_csv_dir):
    '''
    Purpose: Getting all the Floor and Building number from training dataset (e.g. b0f0)

    Parameters
    ----------
    train_csv_dir : string
        directory of training dataset

    Returns
    -------
    loc : list
        contains string of FID
    unique_loc : 2d list
        contains [[buildingID, floorID]]

    '''
    df = pd.read_csv(train_csv_dir, header=0)
    unique = df.groupby(["BUILDINGID","FLOOR"]).size().reset_index().rename(columns={0:'count'})
    unique_loc = np.array(unique.iloc[:,0:2])
    loc = []
    for i in range(len(unique_loc)):
        loc.append("b"+str(unique_loc[i][0]) + "f" + str(unique_loc[i][1]))
    return loc, unique_loc 

def uji_test_csv(test_dir, save_dir):
    '''
    Purpose: Create new csv files according to FID for test data

    Parameters
    ----------
    test_dir : string
        directory of testing csv file
    save_dir : string
        directory of csv files to be saved

    '''
    df = pd.read_csv(test_dir, header=0)
    unique = df.groupby(["BUILDINGID","FLOOR"]).size().reset_index().rename(columns={0:'count'})
    unique_loc = np.array(unique.iloc[:,0:2])
    for i in range(len(unique_loc)):
        curr_df = df[(df['BUILDINGID'] == unique_loc[i][0]) & (df['FLOOR'] == unique_loc[i][1])]
        curr_df = curr_df.iloc[:,:524]
        curr_df = curr_df.replace(100,-110)
        curr_file_name = "b"+str(unique_loc[i][0])+"f"+str(unique_loc[i][1])+"_test.csv"
        curr_df.to_csv(save_dir+curr_file_name, index = False)



def generate_image(train_csv_dir, test_csv_dir, img_dim, save_dir, csv_dir,max_rssi, split = False):
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
    train_df = pd.read_csv(train_csv_dir, header = 0)
    test_df = pd.read_csv(test_csv_dir, header = 0)
    fid, fid_csv = uji_id(train_csv_dir)
    train_df = train_df.replace(100,-110)
    test_df = test_df.replace(100,-110)
    ### Create directory path to save images
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    ### Create extra columns to fit the image dimension
    img_size_diff = (img_dim*img_dim)-520
    for diff in range(img_size_diff):
        train_df.insert(520,"P"+str(diff), -110)
        test_df.insert(520,"P"+str(diff), -110)
        
    print("Size before removing: ", len(train_df))
    ### Removing small label samples 
# =============================================================================
#     for i in range(len(fid_csv)):
#         curr_df = train_df[(train_df['BUILDINGID'] == fid_csv[i][0]) & (train_df['FLOOR'] == fid_csv[i][1])]
#         unique_loc = curr_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
#         temp = unique_loc.index[(unique_loc["count"] < 4)]
#         if (len(temp) != 0):
#             temp_loc = []
#             for t in range(len(temp)):
#                 temp_loc.append(unique_loc.iloc[temp[t],:2].tolist())
#                 train_df = train_df.drop(train_df[(train_df["BUILDINGID"] == fid_csv[i][0]) & 
#                                                   (train_df["FLOOR"] == fid_csv[i][1]) & 
#                                                   (train_df["LONGITUDE"] == temp_loc[t][0]) & 
#                                                   (train_df["LATITUDE"] == temp_loc[t][1])].index)
#                 print("Removing: ", fid[i], temp_loc[t][0], temp_loc[t][1])
#                 print("Size after removing: ", len(train_df))
# =============================================================================
                
    if (split == True):
        for i in range(len(fid_csv)):
            curr_df = train_df[(train_df['BUILDINGID'] == fid_csv[i][0]) & (train_df['FLOOR'] == fid_csv[i][1])]
            train, valid = train_test_split(curr_df, test_size = 0.2)
            train.to_csv(csv_dir+"/"+fid[i]+"_train.csv", index=False)
            valid.to_csv(csv_dir+"/"+fid[i]+"_valid.csv", index=False)
            train = train.iloc[:,:-7]
            valid = valid.iloc[:,:-7]
            
            unique_loc_train = train.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
            unique_loc_train = np.array(unique_loc_train.iloc[:,:2])
            unique_loc_valid = valid.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
            unique_loc_valid = np.array(unique_loc_valid.iloc[:,:2])
            
            for unique in range(len(unique_loc_train)):
                rssi = train[(train["LONGITUDE"] == unique_loc_train[unique][0]) & 
                             (train["LATITUDE"] == unique_loc_train[unique][1])]
                rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
                if not os.path.isdir(save_dir+"/train_img/"+fid[i]+"_train/{}_{}".format(unique_loc_train[unique][0]
                                                                                         ,unique_loc_train[unique][1])):
                    os.makedirs(save_dir+"/train_img/"+fid[i]+"_train/{}_{}".format(unique_loc_train[unique][0],
                                                                                    unique_loc_train[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/train_img/"+fid[i]+"_train/{}_{}/{}.png".format(unique_loc_train[unique][0],
                                                                                          unique_loc_train[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
            for unique in range(len(unique_loc_valid)):
                rssi = valid[(valid["LONGITUDE"] == unique_loc_valid[unique][0]) & 
                             (valid["LATITUDE"] == unique_loc_valid[unique][1])]
                rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
                if not os.path.isdir(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                         unique_loc_valid[unique][1])):
                    os.makedirs(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                    unique_loc_valid[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}/{}.png".format(unique_loc_valid[unique][0],
                                                                                          unique_loc_valid[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
            
        for i in range(len(fid_csv)):
            curr_df = test_df[(test_df['BUILDINGID'] == fid_csv[i][0]) & (test_df['FLOOR'] == fid_csv[i][1])] #by fid
            curr_df = curr_df.iloc[:,:-7]
            unique_loc = curr_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
            unique_loc = np.array(unique_loc.iloc[:,:2])
            for unique in range(len(unique_loc)): #by coord
                rssi = curr_df[(curr_df["LONGITUDE"] == unique_loc[unique][0]) & (curr_df["LATITUDE"] == unique_loc[unique][1])]
                rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
      
                if not os.path.isdir(save_dir+"/test_img/"+fid[i]+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                    os.makedirs(save_dir+"/test_img/"+fid[i]+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/test_img/"+fid[i]+"_test/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
    else:
        for i in range(len(fid_csv)):
            curr_df = train_df[(train_df['BUILDINGID'] == fid_csv[i][0]) & (train_df['FLOOR'] == fid_csv[i][1])] #by fid
            curr_df = curr_df.iloc[:,:-7]
            unique_loc = curr_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
            unique_loc = np.array(unique_loc.iloc[:,:2])
            for unique in range(len(unique_loc)): #by coord
                rssi = curr_df[(curr_df["LONGITUDE"] == unique_loc[unique][0]) & (curr_df["LATITUDE"] == unique_loc[unique][1])]
                rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)
      
                if not os.path.isdir(save_dir+"/train_only/"+fid[i]+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                    os.makedirs(save_dir+"/train_only/"+fid[i]+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/train_only/"+fid[i]+"/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                
            
            
            
    # print(unique_loc[][])
    
def max_rssi(train_csv_dir,test_csv_dir):
    ### Finding max rssi value
    train_df = pd.read_csv(train_csv_dir, header = 0)
    test_df = pd.read_csv(test_csv_dir, header = 0)
    fid, fid_csv = uji_id(train_csv_dir)
    train_df = train_df.replace(100,-110)
    test_df = test_df.replace(100,-110)
    train_rssi = train_df.iloc[:,:520].to_numpy()
    test_rssi = test_df.iloc[:,:520].to_numpy()
    if train_rssi.max() > test_rssi.max():
        max_rssi = train_rssi.max()
    else:
        max_rssi = test_rssi.max()
    return max_rssi

def dirich_image(csv_dir, fid, img_dim, max_rssi, save_dir):
    train_df = pd.read_csv(csv_dir+fid+".csv", header=0)
    
    #Inserting padding to create a 23x23 image dim
    img_size_diff = (img_dim*img_dim)-520
    for diff in range(img_size_diff):
        train_df.insert(520,"P"+str(diff), -110)


    unique_loc = train_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
    unique_loc = np.array(unique_loc.iloc[:,:2])
    for unique in range(len(unique_loc)): #by coord
        rssi = train_df[(train_df["LONGITUDE"] == unique_loc[unique][0]) & (train_df["LATITUDE"] == unique_loc[unique][1])]
        rssi = np.array(rssi.iloc[:,:-2]).reshape(-1,img_dim, img_dim)

        if not os.path.isdir(save_dir+"/dirichlet/"+fid+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
            os.makedirs(save_dir+"/dirichlet/"+fid+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
        for k in range(len(rssi)):
            plt.imsave(save_dir+"/dirichlet/"+fid+"/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                       rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")

def train_split(csv, csv_dir):
    '''
    Purpose: Split training dataset into unique building and floor

    Parameters
    ----------
    csv : string
        training dataset csv
    csv_dir : string
        directory of csv to be saved

    Returns
    -------
    None.

    '''
    df = pd.read_csv(csv,header=0)
    for bid in range(2):
        for fid in range(4):
            temp = df[(df["BUILDINGID"] == bid) & (df["FLOOR"] == fid)]
            temp.to_csv(csv_dir+"b"+str(bid)+"f"+str(fid)+"_train.csv", index=False)
    for fid in range(5):
        temp = df[(df["BUILDINGID"] == 2) & (df["FLOOR"] == fid)]
        temp.to_csv(csv_dir+"b2f"+str(fid)+"_train.csv", index=False)
        
def combined_img():
    floor_id = ['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3','b2f0','b2f1', 'b2f2', 'b2f3', 'b2f4']
    for k in range(len(floor_id)):
        ori_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/UJI/images/gan+/train_only/" + floor_id[k]+"/" #original image folder
        aug_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/UJI/images/gan+/dirichlet/"+floor_id[k]+"/" #augmented image folder
        new_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/UJI/images/gan+/ori_dirich/"+floor_id[k]+"/" #new data image path
        label_name = os.listdir(ori_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for l in range(len(label_name)):
            if not os.path.exists(new_dir+label_name[l]):
                os.makedirs(new_dir+label_name[l])
                
        for i in range(len(label_name)):
            img_name = os.listdir(ori_dir+label_name[i])
            for img in img_name:
                shutil.copy(ori_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/'+img)
            if os.path.exists(aug_dir + label_name[i]):
                img_name = os.listdir(aug_dir + label_name[i])
                for img in img_name:
                    shutil.copy(aug_dir + label_name[i] + "/" + img, new_dir+label_name[i]+'/dirich_'+img)
    
        
if __name__ == "__main__":
    
    
# =============================================================================
# # =============================================================================
# #     1. Split original csv file into multiple csv files (by floor and building ID)
# # =============================================================================
# # =============================================================================
# 
# ### For testing dataset
#     ### Input Parameters ###
#     test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/UJI/csv_files/UJI-testData.csv"
#     save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/UJI/csv_files/uji_split/"
# # =============================================================================
#     uji_test_csv(test_csv_dir, save_dir)
#     
# # =============================================================================
# #   2. Generate images for wgan_gp (also splits training dataset into train and validation csv files e.g. b0f0_train.csv, b0f0_valid.csv)
# # =============================================================================
#     ### Input Parameters ###
#     train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
#     test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-testData.csv"
#     img_dim = 23
#     img_save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/UJI/images/gan+/"
#     csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/UJI/csv_files/uji_split"
#     # =============================================================================
#     max_rssi = max_rssi(train_csv_dir, test_csv_dir)
#     
#     generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_dir, max_rssi,True) #for localisation
#     generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_dir, max_rssi,False) #for augmentation (as it should use all training dataset)
# =============================================================================

# =============================================================================
#   3. Generate dirichlet images
# =============================================================================
    ### Input Parameters ###
# =============================================================================
#     csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/dirichlet/"
#     fid = ['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3','b2f0','b2f1', 'b2f2', 'b2f3', 'b2f4']
#     train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
#     test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-testData.csv"
#     img_dim = 23
#     save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/UJI/images/gan+/"
#     # =============================================================================
#     max_rssi = max_rssi(train_csv_dir, test_csv_dir)
#     for i in range(len(fid)):
#         dirich_image(csv_dir, fid[i], img_dim, max_rssi, save_dir)
# =============================================================================

# =============================================================================
#   4. Create new folder with images for GAN+/WGAN-GP training
# =============================================================================
    combined_img()
### for splitting training dataset
# =============================================================================
#     csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
#     csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/train_split/"
#     train_split(csv,csv_dir)
# 
# =============================================================================

















