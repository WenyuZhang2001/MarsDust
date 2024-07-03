import numpy as np
import matplotlib as mpl
import pandas as pd
import pylab as plt
import h5py
from scipy.io import loadmat
from sklearn.decomposition import PCA
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Conv1D,MaxPooling1D,Flatten
from keras import losses
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler

# load model
model = tf.keras.models.load_model('../dust_signal_test/MAVEN_DustSIgnalIdentify_version5')

# load Signal Data set
year = 2015
month_list = range(1,13)
day_list = range(1,32)
for month in month_list:
    for day in day_list:
        # load file, if no file , continue
        try:
            SignalData = pd.read_csv(f'MAVEN_SignalDataSet/mvn_lpw_l2_we12burstmf_'
                                 f'{year:0>4d}{month:0>2d}{day:0>2d}_SignalSet.csv')
        except:
            print(f'No file: MAVEN_SignalDataSet/'
                  f'mvn_lpw_l2_we12burstmf_{year:0>4d}{month:0>2d}{day:0>2d}_SignalSet.csv')
            continue

        SignalNumbers = len(SignalData['StartTime'])
        print('SignalNumbers = ', SignalNumbers)
        Signal = SignalData.iloc[:,-1500:]
        scaler = MinMaxScaler()
        Signal = scaler.fit_transform(Signal.T)
        Signal = Signal.T
        Probability = model.predict(Signal,verbose=1)
        Prob = pd.DataFrame(Probability,columns=['Probability'])
        SignalData = pd.concat([Prob,SignalData],axis=1)
        SignalData.to_csv(f'MAVEN_SignalProbability_model5/mvn_lpw_l2_we12burstmf_'
                          f'{year:0>4d}{month:0>2d}{day:0>2d}_SignalProbability.csv')