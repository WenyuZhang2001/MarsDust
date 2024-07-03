import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

# date
year = 2015
month = 11
day_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


for day in day_list:
    # get all file names
    path = f'Data/maven.lpw.calibrated_{year}_{month}'
    files = os.listdir(path)
    file_name = ''
    for f in files:
        if len(f) < 20:
            continue
        elif int(f[-14:-12]) == day and int(f[-16:-14]) == month and f[-4:]=='.cdf':
            file_name = f

    if file_name == '':
        continue
    print('file_name = ',file_name)
    # read data

    # check file name in the dir

    cdf = pycdf.CDF(path+'/'+file_name)
    # print(cdf.keys())

    # '''
    # KeysView(<CDF:
    # data: CDF_FLOAT
    # ddata: CDF_FLOAT
    # epoch: CDF_TIME_TT2000
    # flag: CDF_FLOAT
    # info: CDF_FLOAT
    # time_unix: CDF_DOUBLE
    # >)
    # '''
    # # make Dataframe, get data and time
    SignalData = pd.DataFrame(cdf['data'][...],columns=['data'])
    SignalData['epoch'] = cdf['epoch'][...]
    # print(SignalData.head().to_string())
    # make signal data set
    def SignalDataSetMake(SignalData,Npoints=1500):
        '''

        :param SignalData: input Signal data
        :param Npoints: Signal length to make
        :return: data set (in DataFrame)
        '''
        TotalNumber = len(SignalData['data'])
        # make 1500 length signal data
        SignalNumbers = int(TotalNumber/Npoints)
        SignalDataValues = np.empty((0,Npoints))
        SignalTime = pd.DataFrame(columns=['StartTime','EndTime'])
        for index in range(SignalNumbers):
            # Npoints = 1500 length signal
            start = index * Npoints
            end = (index + 1) * Npoints
            SignalDataValues = np.append(SignalDataValues,SignalData['data'].iloc[start:end].to_numpy().reshape(-1,Npoints),axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                SignalTime = SignalTime.append({'StartTime':SignalData['epoch'].iloc[start],
                                            'EndTime':SignalData['epoch'].iloc[end-1]},ignore_index=True)
            # show the process
            if index%10==0:
                print(f'{year:0>4d}{month:0>2d}{day:0>2d}'+'Current processing index = ',index)

        SignalDataSet = pd.DataFrame(SignalDataValues)
        SignalDataSet = pd.concat([SignalTime,SignalDataSet],axis=1)
        return SignalDataSet

    SignalDataSet = SignalDataSetMake(SignalData)
    # print(SignalDataSet.head().to_string())
    SignalDataSet.to_csv(f'MAVEN_SignalDataSet/mvn_lpw_l2_we12burstmf_{year:0>4d}{month:0>2d}{day:0>2d}_SignalSet.csv')