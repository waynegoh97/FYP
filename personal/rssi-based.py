import math
import random

import numpy as np
import pandas as pd


def permutation(size, fid, bid, save_path):  # size is the number of permutations performed, filename as a string
    # Find number of unique RP
    train_df = pd.read_csv(
        'C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/csv_dataset/UJI/csv_files/UJI-trainingData.csv',
        header=0)
    train_df = train_df[(train_df["BUILDINGID"] == bid) & (train_df["FLOOR"] == fid)]
    train_df = train_df.replace(100, -110)
    unique = train_df.groupby(["LATITUDE", "LONGITUDE"]).size().reset_index().rename(columns={0: 'count'})
    loc = np.asarray(unique.iloc[:, 0:2])

    col_name = []
    for ap in range(520):
        col_name.append("AP" + str(ap + 1))
    col_name.append("LATITUDE")
    col_name.append("LONGITUDE")
    df = pd.DataFrame(columns=col_name)
    for i in range(len(loc)):  # for each RP
        overall = []

        rp_samples = np.array(train_df[(train_df.LATITUDE == loc[i][0]) & (train_df.LONGITUDE == loc[i][1])].iloc[:,:520])  # Curr RP samples
        dsize = size - len(rp_samples)
        nsize = int(dsize/len(rp_samples))+1
        for sam in range(len(rp_samples)):
            gen_samples = []
            for s in range(520):
                temp = []
                if rp_samples[sam][s] != -110:
                    col = [row[s] for row in rp_samples]
                    for k in range(nsize):
                        temp.append(int(random.choices(col)[0]))
                else:
                    for k in range(nsize):
                        temp.append(-110)
                gen_samples.append(temp)

            gen_samples = np.array(gen_samples).T.reshape(-1, 520)
            coord = np.full((len(gen_samples), 2), [loc[i][0], loc[i][1]])
            gen_samples = np.concatenate((gen_samples, coord), axis=1).tolist()
            overall.append(gen_samples)
        overall = [e for sl in overall for e in sl]
        #print(len(overall))
        if len(overall) > size:
            more =len(overall) - size
            for m in range(more):
                overall.pop(random.randint(0, len(overall) - 1 - m))
        tdf = pd.DataFrame(overall, columns=col_name)
        df = df.append(tdf, ignore_index=True)

    df.to_csv(save_path + "/" + "b" + str(bid) + "f" + str(fid) + ".csv", index=False)

permutation(200,4,2,"C:/Users/noxtu/LnF_FYP2122S1_Goh-Yun-Bo-Wayne/FYP_data/uji_data/csv_files/rssi-based/")