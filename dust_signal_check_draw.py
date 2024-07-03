import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler



dustdata = loadmat('dust_waveform.mat')
DustWaveForm = dustdata['dust_waveform']
DustWaveForm = np.delete(DustWaveForm,-1,axis=0)
scaler = MinMaxScaler()
DustWaveForm = scaler.fit_transform(DustWaveForm)
DustWaveForm = DustWaveForm.T

nplot = 20
irndD = np.random.randint(1,len(DustWaveForm[:,0]),nplot) #random choose index
iD = DustWaveForm[irndD]
fig, axes = plt.subplots(nplot,1)
fig.set_size_inches(9,9)
for iplot in range(nplot):
    ax = axes[iplot]
    ax.plot(iD[iplot], color='r')
    ax.yaxis.set_ticks([])
    plt.tight_layout()

plt.show()