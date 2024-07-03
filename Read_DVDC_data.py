# read DFB DVDC txt document
import pandas as pd
import matplotlib.pyplot as plt
DFB_VDC_data = pd.read_table(r'C:\Users\12938\Downloads\PSP_FLD_L2_DFB_WF_VDC_131817.txt',
                           comment='#',header=None,skiprows=90,sep='\s+',engine='python')
DFB_VDC_data.drop(axis=0,inplace=True,index=[0,1])
DFB_VDC_data.columns = ['Time','T','V']
# print(DFB_VDC_data['V'].describe())
# print(DFB_VDC_data.head())
# print(DFB_VDC_data.loc[DFB_VDC_data['Time']=='EPOCH'])
# print(DFB_VDC_data.iloc[12656212:12656215,:])

n = 126

for i in range(n):
    # cut data into 100000 points a csv
    start = i*100000 + 12656212
    end = (i+1)*100000 + 12656212
    data_100000 = DFB_VDC_data.iloc[start:end]
    data_100000.to_csv(f'DFB_VDC_100000points_V3/DFB_VDC_100000points_{i}_V3.csv')
    print(f'{i}',end='')