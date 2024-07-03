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


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dustdata = loadmat('dust_waveform.mat')
nondustdata = loadmat('non_dust_waveform2.mat')
indistingguishdata = loadmat('non_dust_waveform1.mat')

# print(type(dustdata))
# print(dustdata.keys(),nondustdata.keys(),indistingguishdata.keys())
DustWaveForm = dustdata['dust_waveform']
IndistingguishableWaveForm = indistingguishdata['undis_waveform']
NonDustWaveform = nondustdata['non_dust_waveform']
DustWaveForm = np.delete(DustWaveForm,-1,axis=0)
IndistingguishableWaveForm = np.delete(IndistingguishableWaveForm,-1,axis=0)
NonDustWaveform = np.delete(NonDustWaveform,-1,axis=0)
print("DustWaveForm.shape",DustWaveForm.shape)
print("NonDustWaveform.shape",NonDustWaveform.shape)
print("IndistingguishableWaveForm.shape",IndistingguishableWaveForm.shape,"\n")
print(type(DustWaveForm))


# data preprocessing
scaler = MinMaxScaler()
DustWaveForm = scaler.fit_transform(DustWaveForm)
IndistingguishableWaveForm = scaler.fit_transform(IndistingguishableWaveForm)
NonDustWaveform = scaler.fit_transform(NonDustWaveform)
DustWaveForm = DustWaveForm.T
IndistingguishableWaveForm = IndistingguishableWaveForm.T
NonDustWaveform = NonDustWaveform.T

# draw signal gragh
def draw_signal_graph(DustWaveForm,NonDustWaveform,IndistingguishableWaveForm):
    nplot = 8 # number of plot

    irndD = np.random.randint(1,len(DustWaveForm[:,0]),nplot) #random choose index
    iD = DustWaveForm[irndD]
    irndN = np.random.randint(1,len(NonDustWaveform[:,0]),nplot)
    iN = NonDustWaveform[irndN]
    irndI = np.random.randint(1,len(IndistingguishableWaveForm[:,0]),nplot)
    iI = IndistingguishableWaveForm[irndI]
    print("plot iN.shape,iD.shape",iN.shape,iD.shape)
    fig, axes = plt.subplots(nplot,3);
    fig.set_size_inches(12,8);
    for iplot in range(1, nplot + 1):

        ax = axes[iplot - 1, 0]
        ax.plot(iD[iplot-1], color='r')
        ax.yaxis.set_ticks([])

        ax = axes[iplot - 1, 1]
        ax.plot(iN[iplot-1], color='k')
        ax.yaxis.set_ticks([])

        ax = axes[iplot - 1, 2]
        ax.plot(iI[iplot - 1], color='b')
        ax.yaxis.set_ticks([])

    axes[0, 0].title.set_text('Dust Signal Examples')
    axes[0, 1].title.set_text('Noise Signal Examples')
    axes[0, 2].title.set_text('Indistinguishable Signal Examples')
    axes[-1, 0].set_xlabel('No. of samples')
    axes[-1, 1].set_xlabel('No. of samples')
    axes[-1, 2].set_xlabel('No. of samples');
    plt.show()

draw_signal_graph(DustWaveForm,NonDustWaveform,IndistingguishableWaveForm)

# make trian set and test set
nDtrian = 3000; nNtrian = 34620; nItrian = 3000;
iDtrian = np.random.randint(1,len(DustWaveForm[:,0]),nDtrian)
iNtrian = np.random.randint(1,len(NonDustWaveform[:,0]),nNtrian)
iItrian = np.random.randint(1,len(IndistingguishableWaveForm[:,0]),nItrian)
Trian = np.vstack((DustWaveForm[iDtrian,:],NonDustWaveform[iNtrian,:],IndistingguishableWaveForm[iItrian,:]))
# make label set 1 is true and 0 is false
Label_trian = np.vstack((np.ones((nDtrian,1)),np.zeros((nNtrian+nItrian,1))))
print("Trian.shape",Trian.shape,"Label.shape",Label_trian.shape)

pDustwaveform = np.delete(DustWaveForm,iDtrian,axis=0)
pNonDustwaveform = np.delete(NonDustWaveform,iNtrian,axis=0)
pIndistingguishableWaveForm = np.delete(IndistingguishableWaveForm,iItrian,axis=0)
nDtest = 1000; nNtest = 1000;nItrian = 1000;
iDtest = np.random.randint(1,len(pDustwaveform[:,0]),nDtest)
iNtest = np.random.randint(1,len(pNonDustwaveform[:,0]),nNtest)
iItrian = np.random.randint(1,len(pIndistingguishableWaveForm[:,0]),nItrian)
Test = np.vstack((pDustwaveform[iDtest,:],pNonDustwaveform[iNtest,:],pIndistingguishableWaveForm[iItrian,:]))
# make label set 1 is true and 0 is false
Label_test = np.vstack((np.ones((nDtest,1)),np.zeros((nNtest+nItrian,1))))
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
model.add(Conv1D(16,5,input_shape=(1500,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(28,5)) # 8,5
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv1D(12,3)) # 8,3
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv1D(8,3)) #4,3
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(4))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=losses.binary_crossentropy,\
              optimizer='Adam',metrics=['accuracy'])
print(model.summary())

# trian the model

history = model.fit(Trian,Label_trian,validation_split=0.1,batch_size=128,epochs=150)
model.save('MAVEN_DustSignalIdentify_version5')
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
plt.plot([0,2],[1,1],c='k',linewidth=0.5)
plt.plot([1,1],[0,2],c='k',linewidth=0.5)
plt.xlim((0.75, 1.002))
plt.ylim((0.75, 1.002))
# plt.legend('Model')
cbar = plt.colorbar();
cbar.set_label('Probability threshold');
plt.tight_layout()
plt.show()