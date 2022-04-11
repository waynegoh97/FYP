import csv

import numpy as np
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch
import sys

sys.path.append("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP/NG/")
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
    origin = np.amin(train_label, axis=0)

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
    train_path, train_label, _ = data_and_label(train_dir)
    valid_path, valid_label, _ = data_and_label(valid_dir)
    test_path, test_label, _ = data_and_label(test_dir)

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
                                    transforms.ToTensor(), transforms.Normalize((mean), (std))])
    # , transforms.Normalize((mean),(std))
    data_set = ImageDataset(img_path, label, transform)

    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0,
                            pin_memory=True)
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


def resnet18():
    '''

    Returns
    -------
    resnet18 : model
        structure of the CNN model

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
          '\nMinimum error:', np.amin(error_NN), '\nMaximum error:', np.amax(error_NN), '\nVariance:', np.var(error_NN))
    result = [np.mean(error_NN), np.amin(error_NN), np.amax(error_NN), np.var(error_NN)]
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
    num_compare = int(len(keys) / len(scenarios))

    for i in range(len(naming)):
        plt.figure(figsize=(10, 7))
        for k in range(num_compare):
            plt.plot(result[keys[key_count]], plot_color[k], label=keys[key_count])

            key_count += 1
        plt.xlabel('Model state')
        plt.ylabel(naming[i])
        plt.title(title_name)
        plt.legend(frameon=False)





if __name__ == '__main__':
##### Parameters #####
##### Code for single runs #####
    batch_size = 32
    lr = 0.0003 #test for 0.001, 0.0001, 0.0003
    epochs = 300
    patience = 100 #old used 30
    bfid = 'F1Sa'#['F1Sa']#["b0f0","b0f1", "b0f2", "b0f3", "b1f0","b1f1", "b1f2", "b1f3","b2f0","b2f1", "b2f2", "b2f3", "b2f4"]
    pred = []
    result = []
    # b = 'floor2'
    # train_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/WGAN-GP+/localisation_300/"+b+"/"
    # valid_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/valid_img/"+b+"_valid/"
    # test_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/test_img/"+b+"_test/30048.562039859924_30334.7167331308/"
    # _, train_label, _ = data_and_label(train_img_dir)
    # print(np.amin(train_label,axis=0))

    for b in bfid:
        for i in range(5):
            print(b,i)
            state_name = "0.0003_"+str(i)+"_"+b+"_train.pt"
            save_dir = "/home/SEANGLIDET/n4/model_state/original/"+b+"/"
            train_img_dir = "/home/SEANGLIDET/n4/images/original/train/" + bfid + "/"
            valid_img_dir = "/home/SEANGLIDET/n4/images/original/valid/" + bfid + "/"
            test_img_dir = "/home/SEANGLIDET/n4/images/original/test/" + bfid + "/"
            # save_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/NG/final_localisation/extendedGAN+300/"+b+"/"
            # train_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/WGAN-GP+/localisation_300/"+b+"/"
            # valid_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/valid_img/"+b+"_valid/"
            # test_img_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/test_img/"+b+"_test/"
            # bsize = len(os.listdir(test_img_dir))


            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model = resnet18()
            ##### Normalize data input and output. Load data. #####
            trainloader, validloader, testloader = normalize_input_and_load(train_img_dir, valid_img_dir, test_img_dir, batch_size)
            train_model(trainloader, validloader, model, lr, epochs, patience, state_name, save_dir)
    #         _, _, testloader = normalize_input_and_load(train_img_dir, valid_img_dir, test_img_dir,bsize)
    #         for test_input, test_output in testloader:
    #             print(test_input.shape)
    #
    #         predict_output, label = predict_result(state_name,model,testloader, save_dir)
    #         _, train_label, _ = data_and_label(train_img_dir)
    #         origin = np.amin(train_label,axis=0)
    #         info = [b,i,"extendedGAN+300"]
    #         info.extend(prediction_error(predict_output, label))
    #         result.append(info)
    #         for size in range(len(predict_output)):
    #             pred.append([b,i,(predict_output[size][0]+origin[0]), (predict_output[size][1]+origin[1]), (label[size][0]+origin[0]), (label[size][1]+origin[1])])
    # pred_df = pd.DataFrame(pred, columns=['FID','TRAIN_NUM', 'PREDICTED_LONGITUDE', 'PREDICTED_LATITUDE', 'ACTUAL_LONGITUDE', 'ACTUAL_LATITUDE'])
    # pred_df.to_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/final_results/pred_extendedGAN+300.csv", index=False)
    # rdf = pd.DataFrame(result, columns = ['BFID', 'TRAIN_NUM', 'CASE', 'MEAN', 'MIN', 'MAX', 'VAR'])
    # rdf.to_csv("C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/final_results/extendedGAN+300_results.csv",index=False)

            # result.insert(0,state_name)
            # result.insert(0,i)
            # result.insert(0,b)
            # with open(r'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/results/rssi-based_results.csv', 'a', newline = '') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(result)
    ##### Prediction Error #####
# =============================================================================
#     num_train = 5
#     batch_size = 24 #must check from test folder with smallest number of labels
#     result = [] ### order: mean, min, max, var
#     state_details = []### order: lr, train_num, fid, case
#     pred = []
#
#     lr = 0.0003
#     case_name = ['_mix']#['_train', '_wgan', '_original_wgan', '_mix']
#     # train_csv = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/UJI_python/csv_files/UJI-trainingData.csv"
#     save_path = 'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/model_state/NG/final_localisation/mix/'
#     floor_id = ['floor-1','floor1','floor2']#['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3', 'b2f0', 'b2f1', 'b2f2', 'b2f3', 'b2f4']
#
#
#     train_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/train_img/"
#     valid_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/valid_img/"
#     test_dir = "C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/image_dataset/NG/images/test_img/"
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
#                 curr_train = train_dir + fid + "_mix"
#                 save_dir = save_path#save_dir = save_path + fid +"/"
#
#                 #######
#                 _, train_label, _ = data_and_label(curr_train)
#
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
#     df.to_csv('C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/final_results/uji_mix_results.csv', index=False)
#     pred_df.to_csv('C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/NG/csv_files/final_results/mix_pred.csv', index=False)

# =============================================================================



















