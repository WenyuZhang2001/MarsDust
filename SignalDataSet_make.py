# make signal data set
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# read index
Num = 12
indexValue = pd.read_csv(f'DFB_VDC_100pointsSignal_V2index_{Num}.csv')
indexValue.columns = ['group','index']

# create an empty SignalDataSet
# SignalData = np.empty((0,100))
# print(SignalData)
# write in data
# read signal data file
signaldata_Dataframe = pd.read_csv('DFB_VDC_V2_DustSignalSet.csv')
signaldata_Dataframe.drop(columns=['Unnamed: 0'], inplace=True)
print(signaldata_Dataframe.head())
SignalData = signaldata_Dataframe.to_numpy()
for l in range(len(indexValue['group'])):
    group, index = (indexValue['group'].iloc[l], indexValue['index'].iloc[l])
    # data_V3 = pd.read_csv(f'DFB_VDC_100000points_V3/DFB_VDC_100000points_{group}_V3.csv')
    data_V2 = pd.read_csv(f'DFB_VDC_100000points_V2/DFB_VDC_100000points_V2_{group}.csv')
    data_V2.drop(columns=['Unnamed: 0'], inplace=True)
    # data_V3.drop(columns=['Unnamed: 0'], inplace=True)

    data_V2.columns = ['Time','V']

    Npoints = 100
    start = index*Npoints
    end = (index+1)*Npoints
    # add a row
    SignalData = np.append(SignalData,data_V2['V'].iloc[start:end].to_numpy().reshape(-1,100),axis=0)

print('The SignalDataSet Dimension is ',SignalData.shape)
# save as csv
SignalDataSet = pd.DataFrame(SignalData)
SignalDataSet.to_csv('DFB_VDC_V2_DustSignalSet.csv')