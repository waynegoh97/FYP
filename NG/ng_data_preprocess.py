import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def ng_coord_reorg_train(csv_dir, save_dir, discard_dir, grid_size):
    ''' 
    Purpose: Reorganise the latitude and longitude to 2m x 2m grid so that there are lesser unique locations (
        for training dataset)
    Parameters
    ----------
    csv_dir : string
        training dataset directory
    save_dir : string
        directory of new training csv to be saved at
    discard_dir : string
        directory of discarded csv to be saved (discarded csv because labels contains < 4 samples)
    grid_size : int
        size in meters

    Returns
    -------
    None.
    '''
    df = pd.read_csv(csv_dir, header=0)
    #Organise df into unique floors
    unique = pd.unique(df["FLOOR"])
    # unique = [-1]
    new_df = pd.DataFrame()
    discard = pd.DataFrame()
    print("Grid size: ", grid_size)
    grid_size = float(grid_size/2)
    for i in unique:
        curr_df = df[(df["FLOOR"] == i)]
        
        min_lat = curr_df["LATITUDE"].min()
        min_long = curr_df["LONGITUDE"].min()
        max_lat = curr_df["LATITUDE"].max()
        max_long = curr_df["LONGITUDE"].max()
        unique_loc = curr_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
        unique_loc = np.array(unique_loc.iloc[:,:2])
        print("Old unique location count: {}".format(len(unique_loc)))
        
        curr_lat = min_lat
        curr_long = min_long
        coord_arr = np.array(curr_df[['LATITUDE','LONGITUDE']])
        
        new_coord = []
        while curr_long < max_long:
            while curr_lat < max_lat:
                curr_lat += grid_size
                new_coord.append([curr_lat, curr_long])
            curr_lat = min_lat
            curr_long+= grid_size
    
        lat = []
        long = []
        del_rows = []
        for coord in range(len(coord_arr)):    
            curr_coord = coord_arr[coord]
            for k in range(len(new_coord)):
                if ((curr_coord[0] < new_coord[k][0] + grid_size) & (curr_coord[0] > new_coord[k][0] - grid_size)):
                    if ((curr_coord[1] < new_coord[k][1]+grid_size) & (curr_coord[1] > new_coord[k][1]-grid_size)):
                        lat.append(new_coord[k][0])
                        long.append(new_coord[k][1])
                        break
                if k == (len(new_coord) - 1):
                    del_rows.append(coord) #contains [row index]
        print("Floor {}\tOld count: {} \t New count: {}".format(i, len(coord_arr), len(lat)))
        print("Deleted rows: ", len(del_rows))
        curr_df = curr_df.drop(curr_df.index[del_rows])
  
        curr_df["NEW_LAT"] = lat
        curr_df["NEW_LONG"] = long
        new_df = new_df.append(curr_df, ignore_index=True)
        unique_loc = curr_df.groupby(["NEW_LAT","NEW_LONG"]).size().reset_index().rename(columns={0:'count'})
        discard = discard.append(unique_loc[(unique_loc['count'] < 4)], ignore_index=True)
        
        print((unique_loc['count'] < 4).value_counts())
        print("New unique location count: {}".format(len(unique_loc)))
    discard=new_df[new_df.set_index(['NEW_LAT','NEW_LONG']).index.isin(discard.set_index(['NEW_LAT','NEW_LONG']).index)]
    discard.to_csv(discard_dir, index=False)
    new_df = pd.concat([new_df,discard]).drop_duplicates(keep=False)
    new_df.to_csv(save_dir, index=False)

def ng_coord_reorg_test(csv_dir, save_dir, grid_size):
    ''' 
    Purpose: Reorganise the latitude and longitude to 2m x 2m grid so that there are lesser unique locations (
        for testing dataset)
    Parameters
    ----------
    csv_dir : string
        training dataset directory
    save_dir : string
        directory of new training csv to be saved at
    grid_size : int
        size in meters

    Returns
    -------
    None.
    '''
    df = pd.read_csv(csv_dir, header=0)
    #Organise df into unique floors
    unique = pd.unique(df["FLOOR"])
    # unique = [-1]
    new_df = pd.DataFrame()
    discard = pd.DataFrame()
    print("Grid size: ", grid_size)
    grid_size = float(grid_size/2)
    for i in unique:
        curr_df = df[(df["FLOOR"] == i)]
        
        min_lat = curr_df["LATITUDE"].min()
        min_long = curr_df["LONGITUDE"].min()
        max_lat = curr_df["LATITUDE"].max()
        max_long = curr_df["LONGITUDE"].max()
        unique_loc = curr_df.groupby(["LONGITUDE","LATITUDE"]).size().reset_index().rename(columns={0:'count'})
        unique_loc = np.array(unique_loc.iloc[:,:2])
        print("Old unique location count: {}".format(len(unique_loc)))
        
        curr_lat = min_lat
        curr_long = min_long
        coord_arr = np.array(curr_df[['LATITUDE','LONGITUDE']])
        
        new_coord = []
        while curr_long < max_long:
            while curr_lat < max_lat:
                curr_lat += grid_size
                new_coord.append([curr_lat, curr_long])
            curr_lat = min_lat
            curr_long+= grid_size
    
        lat = []
        long = []
        del_rows = []
        for coord in range(len(coord_arr)):    
            curr_coord = coord_arr[coord]
            for k in range(len(new_coord)):
                if ((curr_coord[0] < new_coord[k][0] + grid_size) & (curr_coord[0] > new_coord[k][0] - grid_size)):
                    if ((curr_coord[1] < new_coord[k][1]+grid_size) & (curr_coord[1] > new_coord[k][1]-grid_size)):
                        lat.append(new_coord[k][0])
                        long.append(new_coord[k][1])
                        break
                if k == (len(new_coord) - 1):
                    del_rows.append(coord) #contains [row index]
        print("Floor {}\tOld count: {} \t New count: {}".format(i, len(coord_arr), len(lat)))
        print("Deleted rows: ", len(del_rows))
        curr_df = curr_df.drop(curr_df.index[del_rows])
  
        curr_df["NEW_LAT"] = lat
        curr_df["NEW_LONG"] = long
        new_df = new_df.append(curr_df, ignore_index=True)
        unique_loc = curr_df.groupby(["NEW_LAT","NEW_LONG"]).size().reset_index().rename(columns={0:'count'})

        print("New unique location count: {}".format(len(unique_loc)))
    
    new_df.to_csv(save_dir, index=False)
    
def plot(train_csv_dir):
    df = pd.read_csv(train_csv_dir, header=0)
    x = np.array(df["NEW_LAT"])
    y = np.array(df["NEW_LONG"])
    plt.xlabel("LATITUDE")
    plt.ylabel("LONGITUDE")
    plt.title("National Gallery Coordinates")
    
    plt.plot(x, y, 'o', color='red');
    
def ng_split_floor(csv_dir, datatype, save_path):
    '''
    Purpose: split training or testing csv file into separate floors (for data augmentation)

    Parameters
    ----------
    csv_dir : string
        directory of training csv file
    datatype : string
        either "_train" or "_test"
    save_path : string
        directory of new csv file to be saved

    Returns
    -------
    None.

    '''
    df = pd.read_csv(csv_dir, header=0)
    unique = np.unique(df["FLOOR"])
    for i in unique:
        curr_df = df[df["FLOOR"] == i]
        curr_df.to_csv(save_path + '/floor' +str(i)+ datatype + ".csv", index = False)

def ng_max_sample(train_csv_dir, save_dir):
    '''
    Purpose: Find the floor containing max sample from the training dataset, and save it as a new csv file

    Returns
    -------
    None.

    '''
    df = pd.read_csv(train_csv_dir, header = 0)
    unique_loc = df.groupby(["NEW_LAT","NEW_LONG"]).size().reset_index().rename(columns={0:'count'})
    new_max = max(unique_loc["count"])
    
    coord = np.array(unique_loc[unique_loc["count"] == new_max]).flatten()
    new_df = df[(df["NEW_LAT"] == coord[0]) & (df["NEW_LONG"] == coord[1])]
    print(new_df[["NEW_LAT", "NEW_LONG", "FLOOR"]])
    new_df.to_csv(save_dir, index=False)
    
def generate_image(train_csv_dir, test_csv_dir, img_dim, save_dir, csv_dir, split = False):
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
    ### Finding max rssi value
    train_df = pd.read_csv(train_csv_dir, header = 0)
    test_df = pd.read_csv(test_csv_dir, header = 0)
    fid_csv = np.unique(train_df["FLOOR"])
    fid = []
    for i in fid_csv:
        fid.append("floor"+str(i))
    train_rssi = train_df.iloc[:,:345].to_numpy() #345 APs in NG dataset
    test_rssi = test_df.iloc[:,:345].to_numpy()
    if train_rssi.max() > test_rssi.max():
        max_rssi = train_rssi.max()
    else:
        max_rssi = test_rssi.max()
    
    ### Create directory path to save images
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    ### Create extra columns to fit the image dimension
    img_size_diff = (img_dim*img_dim)-345
    for diff in range(img_size_diff):
        train_df.insert(345,"P"+str(diff), -110)
        test_df.insert(345,"P"+str(diff), -110)
        
    print("Size before removing: ", len(train_df))
    ### Removing small label samples 
    for i in range(len(fid_csv)):
        curr_df = train_df[(train_df['FLOOR'] == fid_csv[i])]
        unique_loc = curr_df.groupby(["NEW_LONG","NEW_LAT"]).size().reset_index().rename(columns={0:'count'})
                
    if (split == True):
        for i in range(len(fid_csv)):
            curr_df = train_df[(train_df['FLOOR'] == fid_csv[i])]
            train, valid = train_test_split(curr_df, test_size = 0.2)
            train.to_csv(csv_dir+"/"+fid[i]+"_train.csv", index=False)
            valid.to_csv(csv_dir+"/"+fid[i]+"_valid.csv", index=False)
            
            unique_loc_train = train.groupby(["NEW_LONG","NEW_LAT"]).size().reset_index().rename(columns={0:'count'})
            unique_loc_train = np.array(unique_loc_train.iloc[:,:2])
            unique_loc_valid = valid.groupby(["NEW_LONG","NEW_LAT"]).size().reset_index().rename(columns={0:'count'})
            unique_loc_valid = np.array(unique_loc_valid.iloc[:,:2])
            
            for unique in range(len(unique_loc_train)):
                rssi = train[(train["NEW_LONG"] == unique_loc_train[unique][0]) & 
                             (train["NEW_LAT"] == unique_loc_train[unique][1])]
                rssi = np.array(rssi.iloc[:,:(img_dim*img_dim)]).reshape(-1,img_dim, img_dim)
                if not os.path.isdir(save_dir+"/train_img/"+fid[i]+"_train/{}_{}".format(unique_loc_train[unique][0]
                                                                                         ,unique_loc_train[unique][1])):
                    os.makedirs(save_dir+"/train_img/"+fid[i]+"_train/{}_{}".format(unique_loc_train[unique][0],
                                                                                    unique_loc_train[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/train_img/"+fid[i]+"_train/{}_{}/{}.png".format(unique_loc_train[unique][0],
                                                                                          unique_loc_train[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
            for unique in range(len(unique_loc_valid)):
                rssi = valid[(valid["NEW_LONG"] == unique_loc_valid[unique][0]) & 
                             (valid["NEW_LAT"] == unique_loc_valid[unique][1])]
                rssi = np.array(rssi.iloc[:,:(img_dim*img_dim)]).reshape(-1,img_dim, img_dim)
                if not os.path.isdir(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                         unique_loc_valid[unique][1])):
                    os.makedirs(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}".format(unique_loc_valid[unique][0],
                                                                                    unique_loc_valid[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/valid_img/"+fid[i]+"_valid/{}_{}/{}.png".format(unique_loc_valid[unique][0],
                                                                                          unique_loc_valid[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
            
        for i in range(len(fid_csv)):
            curr_df = test_df[(test_df['FLOOR'] == fid_csv[i])] #by fid
            unique_loc = curr_df.groupby(["NEW_LONG","NEW_LAT"]).size().reset_index().rename(columns={0:'count'})
            unique_loc = np.array(unique_loc.iloc[:,:2])
            for unique in range(len(unique_loc)): #by coord
                rssi = curr_df[(curr_df["NEW_LONG"] == unique_loc[unique][0]) & (curr_df["NEW_LAT"] == unique_loc[unique][1])]
                rssi = np.array(rssi.iloc[:,:(img_dim*img_dim)]).reshape(-1,img_dim, img_dim)
      
                if not os.path.isdir(save_dir+"/test_img/"+fid[i]+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                    os.makedirs(save_dir+"/test_img/"+fid[i]+"_test/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/test_img/"+fid[i]+"_test/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")
                    
    else:
        for i in range(len(fid_csv)):
            curr_df = train_df[(train_df['FLOOR'] == fid_csv[i])] #by fid
            unique_loc = curr_df.groupby(["NEW_LONG","NEW_LAT"]).size().reset_index().rename(columns={0:'count'})
            unique_loc = np.array(unique_loc.iloc[:,:2])
            for unique in range(len(unique_loc)): #by coord
                rssi = curr_df[(curr_df["NEW_LONG"] == unique_loc[unique][0]) & (curr_df["NEW_LAT"] == unique_loc[unique][1])]
                rssi = np.array(rssi.iloc[:,:(img_dim*img_dim)]).reshape(-1,img_dim, img_dim)
      
                if not os.path.isdir(save_dir+"/train_only/"+fid[i]+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1])):
                    os.makedirs(save_dir+"/train_only/"+fid[i]+"/{}_{}".format(unique_loc[unique][0],unique_loc[unique][1]))
                for k in range(len(rssi)):
                    plt.imsave(save_dir+"/train_only/"+fid[i]+"/{}_{}/{}.png".format(unique_loc[unique][0],unique_loc[unique][1],k),
                               rssi[k], vmin = -110, vmax = max_rssi, cmap="gray")

if __name__ == '__main__':
# =============================================================================
#     1. Clean up csv file by grid (also removing grid with samples < 4)
# =============================================================================
    train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_train.csv"
    new_train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_train_modified_2m.csv"
    test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_test.csv"
    new_test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_test_modified_2m.csv"
    discard = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/discard_2m.csv"
    grid_size = 2
    #Grid size: 1m (1218 labels usable, 310 labels less than 4 samples, 43 rows not within grid)
    #Grid size: 1.5m (911 labels usable, 139 labels less than 4 samples, 43 rows not within grid)
    #Grid size: 2m (729 labels usabale, 78 labels less than 4 samples, 43 rows not within grid)
    #Grid size: 2.5m (610 labels usabale, 44 labels less than 4 samples, 43 rows not within grid)
    #Grid size: 3m (31 labels less than 4 samples, 43 rows not within grid)
    ng_coord_reorg_train(train_csv_dir, new_train_csv_dir, discard, grid_size)
    ng_coord_reorg_test(test_csv_dir, new_test_csv_dir, grid_size)
# =============================================================================
#     2. Split csv data into different floors
# =============================================================================

    test_datatype = "_test"
    save_path = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/ng_split"
    new_test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_test_modified_2m.csv"
    max_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/ng_split/WGAN/max_sample.csv"
    ng_split_floor(new_test_csv_dir, test_datatype, save_path)

    ng_max_sample(new_train_csv_dir, max_dir) #print details of most sample (manually take from train_only image folder)
    
# =============================================================================
#     plot(new_train_csv_dir)
#     plot(discard)
# =============================================================================
# =============================================================================
#   3. Generating images
# =============================================================================
    train_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_train_modified_2m.csv"
    test_csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/combined_test_modified_2m.csv"
    img_dim = 19
    img_save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/images/"
    csv_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/csv_files/ng_split/WGAN/"
    
    generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_dir, True) #for localisation
    generate_image(train_csv_dir, test_csv_dir, img_dim, img_save_dir, csv_dir, False) #for augmentation (as it should use all training dataset)








    
