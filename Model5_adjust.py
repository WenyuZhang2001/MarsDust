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

# load NonSignal data index
NonSignalDataIndex = pd.read_csv('MAVEN_DustImpactSignal/model5_Adjust_DustSignal/NonSignalIndex.csv',
                                 header=None)
NonSignalDataIndex.columns = ['Year','Month','Day','index']

Npoints = 1500
NonSignalDataSet = np.empty((0,Npoints))
for i in range(len(NonSignalDataIndex['Year'])):
    Year = NonSignalDataIndex['Year'].iloc[i]
    Month = NonSignalDataIndex['Month'].iloc[i]
    Day = NonSignalDataIndex['Day'].iloc[i]
    index = NonSignalDataIndex['index'].iloc[i]

    DustSignal = pd.read_csv('MAVEN_DustImpactSignal/model5_Adjust_DustSignal/mvn_lpw_l2_we12burstmf_'
                             f'{Year:0>4d}{Month:0>2d}{Day:0>2d}_DustImpactSignal.csv')
    print(Year,Month,Day,index)
    NonSignalDataSet = np.append(NonSignalDataSet,DustSignal.iloc[index,-Npoints:].to_numpy().reshape(-1,Npoints),axis=0)

# load Dust Impact Signal
Total_number = 0
DustImpactSignal = np.empty((0,Npoints))
Date = pd.read_csv('MAVEN_DustImpactSignal/DataDate.csv',header=None)
Date.columns = ['Year','Month','Day']

for i in range(len(Date['Year'])):
    Year = Date['Year'].iloc[i]
    Month = Date['Month'].iloc[i]
    Day = Date['Day'].iloc[i]
    DustSignal = pd.read_csv('MAVEN_DustImpactSignal/model5_Adjust_DustSignal/mvn_lpw_l2_we12burstmf_'
                             f'{Year:0>4d}{Month:0>2d}{Day:0>2d}_DustImpactSignal.csv')

    Temp_Data = NonSignalDataIndex.loc[NonSignalDataIndex['Year'] == Year]
    Temp_Data = Temp_Data.loc[Temp_Data['Month'] == Month]
    Temp_Data = Temp_Data.loc[Temp_Data['Day'] == Day]

    for j in range(len(DustSignal['Probability'])):
        Total_number += 1
        if j in Temp_Data['index'].values:
            continue

        DustImpactSignal = np.append(DustImpactSignal,DustSignal.iloc[j,-Npoints:].to_numpy().reshape(-1,Npoints),axis=0)

print('Total Signal number',Total_number)
print('DustImpactSignal.shape = ',DustImpactSignal.shape)
print('NonSignalDataSet.shape = ',NonSignalDataSet.shape)

# Data Pre Processing
scaler = MinMaxScaler()
DustWaveForm = scaler.fit_transform(DustImpactSignal.T)
DustWaveForm = DustWaveForm.T
NonDustWaveform = scaler.fit_transform(NonSignalDataSet.T)
NonDustWaveform = NonDustWaveform.T

# label
nDtrian = DustWaveForm.shape[0]; nNtrian = NonDustWaveform.shape[0];
iDtrian = np.random.randint(1,len(DustWaveForm[:,0]),nDtrian)
iNtrian = np.random.randint(1,len(NonDustWaveform[:,0]),nNtrian)
Trian = np.vstack((DustWaveForm[iDtrian,:],NonDustWaveform[iNtrian,:]))
# make label set 1 is true and 0 is false
Label_trian = np.vstack((np.ones((nDtrian,1)),np.zeros((nNtrian,1))))
print("Trian.shape",Trian.shape,"Label.shape",Label_trian.shape)

#  Trian  and Test set index shuffle
# index = np.random.permutation(Trian[:,0].size)
# print(Trian[:,0].size)
# Trian = Trian[index]
# Label_trian = Label_trian[index]

# model train
history = model.fit(Trian,Label_trian,validation_split=0.15,batch_size=24,epochs=200)
model.save('dust_signal_test/MAVEN_DustSignalIdentify_version5_Adjust')

def plot_training_history(history):
    # print(history.history.keys())

    fig = plt.figure()
    fig.set_size_inches(10, 5);

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    # plt.show()

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

plot_training_history(history)
mpl.rcParams['figure.figsize'] = (4,5)
plt.show()