# make None signal data set
import random

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# read index
num = 12
Signalindex = pd.read_csv(f'DFB_VDC_100pointsSignal_V2index_{num}.csv')
Signalindex.columns = ['group','index']

# create an empty SignalDataSet
# NonSignalData = np.empty((0,100))
NonSignalData_Dataframe = pd.read_csv('DFB_VDC_V2_NonDustSignalSet.csv')
NonSignalData_Dataframe.drop(columns=['Unnamed: 0'], inplace=True)
# print(NonSignalData_Dataframe.head())
NonSignalData = NonSignalData_Dataframe.to_numpy()
# print(SignalData)
# write in data
# num = None Signal data numbers
num = 2000
# group and index
totalgroups = 62
totalIndex = 999  # 999+1 = 1000 pictures
l = 0
while l < num:
    # random group and index number
    group = random.randint(0,totalgroups)
    index = random.randint(0,totalIndex)
    # check if a Dust Signal
    if group in Signalindex['group'].values:
        if index in Signalindex['index'].loc[group==Signalindex['group']].values:
            print(f'({group,index}) is in SignalIndex. Continue the {l} term.')
            continue

    # read csv
    data_V2 = pd.read_csv(f'DFB_VDC_100000points_V2/DFB_VDC_100000points_V2_{group}.csv')
    data_V2.drop(columns=['Unnamed: 0'], inplace=True)
    data_V2.columns = ['Time','V']

    Npoints = 100
    start = index*Npoints
    end = (index+1)*Npoints
    # add a row
    NonSignalData = np.append(NonSignalData,data_V2['V'].iloc[start:end].to_numpy().reshape(-1,100),axis=0)

    l += 1

print('The NonSignalDataSet Dimension is ',NonSignalData.shape)
# save as csv
SignalDataSet = pd.DataFrame(NonSignalData)
SignalDataSet.to_csv('DFB_VDC_V2_NonDustSignalSet.csv')