import numpy as np
import matplotlib as mpl
import pylab as plt
import h5py
from scipy.io import loadmat
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Conv1D,MaxPooling1D,Flatten
from keras import losses
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# read data
Dustdata = pd.read_csv('DFB_VDC_V2_DustSignalSet.csv',index_col=0)
NonDustdata = pd.read_csv('DFB_VDC_V2_NonDustSignalSet.csv',index_col=0)
print(Dustdata.shape)
# to numpy ndarray
Dustdata = Dustdata.to_numpy()
NonDustdata = NonDustdata.to_numpy()

# preprocessing
Scaler = MinMaxScaler()
Dustdata = Scaler.fit_transform(Dustdata.T)
NonDustdata = Scaler.fit_transform(NonDustdata.T)
DustWaveForm = Dustdata.T
NonDustWaveform = NonDustdata.T

def draw_signal_graph(DustWaveForm,NonDustWaveform):
    nplot = 8 # number of plot

    irndD = np.random.randint(0,len(DustWaveForm[:,0]),nplot) #random choose index
    iD = DustWaveForm[irndD]
    irndN = np.random.randint(0,len(NonDustWaveform[:,0]),nplot)
    iN = NonDustWaveform[irndN]
    print("plot iN.shape,iD.shape",iN.shape,iD.shape)
    fig, axes = plt.subplots(nplot,2);
    fig.set_size_inches(12,8);
    for iplot in range(1, nplot + 1):

        ax = axes[iplot - 1, 0]
        ax.plot(iD[iplot-1], color='r')
        ax.yaxis.set_ticks([])

        ax = axes[iplot - 1, 1]
        ax.plot(iN[iplot-1], color='k')
        ax.yaxis.set_ticks([])

    axes[0, 0].title.set_text('Dust Signal Examples')
    axes[0, 1].title.set_text('Noise Signal Examples')
    axes[-1, 0].set_xlabel('No. of samples')
    axes[-1, 1].set_xlabel('No. of samples')
    plt.tight_layout()
    plt.show()

draw_signal_graph(DustWaveForm,NonDustWaveform)

# make trian set and test set
nDtrian =int(len(DustWaveForm)*0.7); nNtrian =int(len(NonDustWaveform)*0.7)
iDtrian = np.random.randint(0,len(DustWaveForm[:,0]),nDtrian)
iNtrian = np.random.randint(0,len(NonDustWaveform[:,0]),nNtrian)
Trian = np.vstack((DustWaveForm[iDtrian,:],NonDustWaveform[iNtrian,:]))
# make label set 1 is true and 0 is false
Label_trian = np.vstack((np.ones((nDtrian,1)),np.zeros((nNtrian,1))))
print("Trian.shape",Trian.shape,"Label.shape",Label_trian.shape)

pDustwaveform = np.delete(DustWaveForm,iDtrian,axis=0)
pNonDustwaveform = np.delete(NonDustWaveform,iNtrian,axis=0)
nDtest = len(DustWaveForm)-nDtrian; nNtest = len(NonDustWaveform)-nNtrian
iDtest = np.random.randint(0,len(pDustwaveform[:,0]),nDtest)
iNtest = np.random.randint(0,len(pNonDustwaveform[:,0]),nNtest)
Test = np.vstack((pDustwaveform[iDtest,:],pNonDustwaveform[iNtest,:]))
# make label set 1 is true and 0 is false
Label_test = np.vstack((np.ones((nDtest,1)),np.zeros((nNtest,1))))
print("Test.shape",Test.shape,"Label.shape",Label_test.shape)

#  Trian  and Test set index shuffle
index = np.random.permutation(Trian[:,0].size)
print(Trian[:,0].size)
Trian = Trian[index]
Label_trian = Label_trian[index]
index = np.random.permutation(Test[:,0].size)
print(Test[:,0].size)
Test = Test[index]
Label_test = Label_test[index]

# set Conv1D model
model = Sequential()
model.add(Conv1D(16,5,input_shape=(100,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.15))

model.add(Conv1D(28,5))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv1D(16,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv1D(12,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(9))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(6))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=losses.binary_crossentropy,\
              optimizer='Adam',metrics=['accuracy'])
print(model.summary())
# tf.keras.utils.plot_model(model,to_file='model.jpg')
# trian the model

history = model.fit(Trian,Label_trian,validation_split=0.1,batch_size=9,epochs=200)
# plot trianing history
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

print("Model accuracy on test set")
score = model.evaluate(Test,Label_test)

# plot PR and RE
def get_precall(y_prob, y_test, prPrimeVect):
    y_test = y_test.flatten()

    pr = []
    re = []
    ac = []

    for p_min in prPrimeVect:
        lgc_predQ = (y_prob >= p_min)
        lgc_testQ = (y_test == 1)

        ntp = np.sum(lgc_predQ & lgc_testQ)
        ntn = np.sum(~lgc_predQ & ~lgc_testQ)
        nfp = np.sum(lgc_predQ & ~lgc_testQ)
        nfn = np.sum(~lgc_predQ & lgc_testQ)
        precision = np.true_divide(ntp, (ntp + nfp))
        recall = np.true_divide(ntp, (ntp + nfn))
        #accuracy = np.true_divide((ntp + ntn), (ntp + ntn + nfp + nfn))
        pr.append(precision)
        re.append(recall)
        #ac.append(accuracy)

    return pr, re#, ac

Y_prob = model.predict(Test,verbose=1)

def plotPrRe(Y_prob,Label_test):

    prPrime = np.arange(0, 1.0, 0.1)
    npr     = prPrime.shape[0]
    pr, re = get_precall(Y_prob[:,0],Label_test,prPrime)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p3 = plt.scatter(re,pr, c=prPrime, cmap='jet', marker='v', edgecolor='k')
    #ax.annotate(str(ntl), xy=(re1[-1]-.005, pr1[-1]-.005))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.plot([0,1.1],[1,1],c='k',linewidth=0.5)
    plt.plot([1,1],[0,1.1],c='k',linewidth=0.5)
    # plt.xlim((0.75, 1.002))
    # plt.ylim((0.75, 1.002))
    # plt.legend('Model')
    cbar = plt.colorbar();
    cbar.set_label('Probability threshold');
    plt.tight_layout()
    plt.show()

plotPrRe(Y_prob,Label_test)

