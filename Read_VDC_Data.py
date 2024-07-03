# read DFB VDC txt document
import pandas as pd
import matplotlib.pyplot as plt
DFB_VDC_data = pd.read_csv(r'C:\Users\12938\Downloads\PSP_FLD_L2_DFB_WF_VDC_127861.csv',
                           comment='#',engine='python')# skiprows=90
# DFB_VDC_data.drop(axis=0,inplace=True,index=[0,1])
# DFB_VDC_data.columns = ['Time','T','V']
# print(DFB_VDC_data['V'].describe())
# print(DFB_VDC_data.head())
# print(DFB_VDC_data.describe())
# total 25312423 points
n  = int(len(DFB_VDC_data)/100000/2)

for i in range(n):
    # cut data into 100000 points a csv
    start = i*100000
    end = (i+1)*100000
    data_100000 = DFB_VDC_data.iloc[start:end]
    data_100000.to_csv(f'DFB_VDC_100000points_V2/DFB_VDC_100000points_V2_{i}.csv')
    print(f'{i}',end='')

# print(DFB_VDC_data.loc[DFB_VDC_data['V2_V']=='V3_V'])
# V3 read and save
# V3_index_start = DFB_VDC_data.loc[DFB_VDC_data['V2_V']=='V3_V'].index.to_list()[0] + 1
# for i in range(n):
#     # cut data into 100000 points a csv
#     start = i*100000+V3_index_start
#     end = (i+1)*100000+V3_index_start
#     data_100000 = DFB_VDC_data.iloc[start:end]
#     data_100000.to_csv(f'DFB_VDC_100000points_V3/DFB_VDC_100000points_V3_{i}.csv')
#     print(f'{i}',end='')