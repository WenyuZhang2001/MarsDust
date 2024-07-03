import pandas as pd
import matplotlib.pyplot as plt

Dustdata = pd.read_csv('DFB_VDC_V2_DustSignalSet.csv',index_col=0)
NonDustdata = pd.read_csv('DFB_VDC_V2_NonDustSignalSet.csv',index_col=0)

plt.figure()
plt.plot(Dustdata.loc[10])
plt.show()
