import os.path
import shutil
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess(csv_file, fid, bid, num_ap):
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
    bfid = df[(df["FLOOR"] == fid) & (df["BUILDINGID"] == bid)]
    bfid = bfid.replace(100, -110)
    unique_loc = bfid.groupby(["LONGITUDE", "LATITUDE"]).size().reset_index().rename(columns={0: 'count'})
    labels = np.array(unique_loc.iloc[:,0:2])
    data_map = {} #keys = (longitude, latitude), access by sample_map[labels[0][0],labels[0][1]]
    test_map = {}
    for label in labels:
        rp = bfid[(bfid["LONGITUDE"] == label[0]) & (bfid["LATITUDE"] == label[1])]
        rp = np.array(rp.iloc[:,:num_ap])
        data_map[label[0],label[1]] = rp
    return data_map

def max_rssi(train_csv_dir,test_csv_dir):
    '''
    Find max rssi for image generation
    Parameters
    ----------
    train_csv_dir: string path of training dataset
    test_csv_dir: string path of testing dataset

    Returns: int of max rssi
    -------

    '''
    ### Finding max rssi value
    train_df = pd.read_csv(train_csv_dir, header = 0)
    test_df = pd.read_csv(test_csv_dir, header = 0)
    train_df = train_df.replace(100,-110)
    test_df = test_df.replace(100,-110)
    train_rssi = train_df.iloc[:,:520].to_numpy()
    test_rssi = test_df.iloc[:,:520].to_numpy()
    if train_rssi.max() > test_rssi.max():
        max_rssi = train_rssi.max()
    else:
        max_rssi = test_rssi.max()
    return max_rssi

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

def train_valid_split(train_csv, fid, bid, save_path):
    '''
    Split training dataset into train and validation data (save as new csv files)
    Parameters
    ----------
    train_csv: original training dataset
    fid: floor id
    bid: building id
    save_path: string path of new csv file

    Returns
    -------

    '''
    df = pd.read_csv(train_csv,header=0)
    bfid = df[(df["FLOOR"] == fid) & (df["BUILDINGID"] == bid)]
    bfid = bfid.replace(100, -110)
    train, valid = train_test_split(bfid, test_size=0.2)
    train.to_csv(save_path + "/b" + str(bid) + "f" + str(fid) + "_train.csv", index=False)
    valid.to_csv(save_path + "/b" + str(bid) + "f" + str(fid) + "_valid.csv", index=False)

def preprocess_dirichlet(csv_file, fid, bid, num_ap):
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
    unique_loc = df.groupby(["LONGITUDE", "LATITUDE"]).size().reset_index().rename(columns={0: 'count'})
    labels = np.array(unique_loc.iloc[:,0:2])
    data_map = {} #keys = (longitude, latitude), access by sample_map[labels[0][0],labels[0][1]]
    for label in labels:
        rp = df[(df["LONGITUDE"] == label[0]) & (df["LATITUDE"] == label[1])]
        rp = np.array(rp.iloc[:,:num_ap])
        data_map[label[0],label[1]] = rp
    return data_map

if __name__ == "__main__":

    ###### Generating original data as images ######
    # train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    # test_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-testData.csv"
    # bid = 0
    # fid = 0
    # num_ap = 520
    # img_dim = 23
    # train_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/original/train"
    # test_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/original/test"
    # #Get training dataset dict
    # train_dict = preprocess(train_csv, fid, bid, num_ap)
    # #Get testing dataset dict
    # test_dict = preprocess(test_csv, fid, bid, num_ap)
    #
    # #Generate images
    # max_rssi = max_rssi(train_csv, test_csv)
    # train_path = train_path +"/b"+str(bid)+"f"+str(fid)
    # test_path = test_path +"/b"+str(bid)+"f"+str(fid)
    # #generate_image(train_dict, img_dim, num_ap, max_rssi, train_path)
    # generate_image(test_dict, img_dim, num_ap, max_rssi, test_path)
    ##################################################################################################################

    ###### Generating training and validation csv dataset ######
    # train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    # bid = 2
    # fid = 4
    # save_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/"
    # train_valid_split(train_csv, fid, bid, save_path)
    ##################################################################################################################

    ###### Generating training and validation image dataset ######
    # fid = 0
    # bid = 0
    # num_ap = 520
    # img_dim = 23
    # train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/original/b"+str(bid)+"f"+str(fid)+"_train.csv"
    # valid_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/original/b"+str(bid)+"f"+str(fid)+"_valid.csv"
    # train_csv_overall = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    # test_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-testData.csv"
    #
    # train_dict = preprocess(train_csv, fid, bid, num_ap)
    # valid_dict = preprocess(valid_csv, fid, bid, num_ap)
    #
    # train_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/original/train"
    # valid_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/original/valid"
    #
    # max_rssi = max_rssi(train_csv_overall, test_csv)
    # train_path = train_path +"/b"+str(bid)+"f"+str(fid)
    # valid_path = valid_path +"/b"+str(bid)+"f"+str(fid)
    # # generate_image(train_dict, img_dim, num_ap, max_rssi, train_path)
    # generate_image(valid_dict, img_dim, num_ap, max_rssi, valid_path)
    ##################################################################################################################

    ###### Generating Dirichlet images ######
    dir_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/rssi-based/b2f4.csv"
    train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv"
    test_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-testData.csv"
    fid = 4
    bid = 2
    num_ap = 520
    img_dim = 23

    dir_dict = preprocess_dirichlet(dir_csv, fid, bid, num_ap)
    dir_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/rssi-based/"

    max_rssi = max_rssi(train_csv, test_csv)
    print(max_rssi)
    # dir_path = dir_path +"/b"+str(bid)+"f"+str(fid)
    # generate_image(dir_dict, img_dim, num_ap, max_rssi, dir_path)


    ###### Combining image folders ######
    # folder1 = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/ori_rssi-based/b2f4/"
    # folder2 = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/images/rssi-based/b2f4/"
    #
    # labels = os.listdir(folder1)
    # for i in range(len(labels)):
    #     init_img_name = os.listdir(folder1+labels[i])
    #     new_len = 150 -  len(init_img_name)
    #     img_name = os.listdir(folder2+labels[i])
    #     #rand = random.sample(range(400), new_len)
    #
    #     for img in range(new_len):
    #         #shutil.copy(folder2 + labels[i] + "/" + img_name[rand[img]], folder1 + labels[i] + '/wgan_' + img_name[rand[img]])
    #         shutil.copy(folder2 + labels[i] + "/" + img_name[img], folder1 + labels[i] + '/rssi_' + img_name[img])

    ###### Find most sample for the building ######
    # csv_file = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/personal/csv_files/UJI-trainingData.csv"
    # bid = 0
    # fid = [0,1,2,3]
    # df = pd.read_csv(csv_file,header=0)
    # floor_max = []
    # most_sample = 0
    # for i in range(len(fid)):
    #     bfid = df[(df["FLOOR"] == fid[i])&(df["BUILDINGID"] == bid)]
    #     unique_loc = bfid.groupby(["LONGITUDE", "LATITUDE"]).size().reset_index().rename(columns={0: 'count'})
    #     max_id = unique_loc["count"].idxmax()
    #     floor_max.append(list(unique_loc.iloc[max_id,:]))
    # for i in range(len(floor_max)):
    #     if floor_max[i][2] > most_sample:
    #         most_sample = floor_max[i][2]
    #         print("Most sample floor: {}, Coordinate: {}, {}".format(i,floor_max[i][0],floor_max[i][1]))
