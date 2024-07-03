# read and draw 100000 points data
import pandas as pd
import matplotlib.pyplot as plt
import os

N = 2
# read index dustsignal V2
indexValue = pd.read_csv(f'DFB_VDC_100pointsSignal_V2index_{N}.csv')
indexValue.columns = ['group','index']

# draw
for l in range(len(indexValue['group'])):
    j, i = (indexValue['group'].iloc[l], indexValue['index'].iloc[l])
    data_V3 = pd.read_csv(f'DFB_VDC_100000points_V3/DFB_VDC_100000points_V3_{j}.csv')
    data_V2 = pd.read_csv(f'DFB_VDC_100000points_V2/DFB_VDC_100000points_V2_{j}.csv')
    data_V2.drop(columns=['Unnamed: 0'], inplace=True)
    data_V3.drop(columns=['Unnamed: 0'], inplace=True)

    # simple operations
    def timeoperation(test_data):
        # change time columns to standard time format
        test_data.columns = ['Time', 'V']
        test_data['Time'] = pd.to_datetime(test_data['Time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        test_data['V'] = test_data['V'] * 1000
        test_data.columns = ['Time', 'mV']
        return test_data
    data_V2 = timeoperation(data_V2)
    data_V3 = timeoperation(data_V3)
    # get the max and min value of this group, to set ylim
    # print(test_data.describe())
    # print(test_data.head())
    # check if the dir exist, if not, set up
    try:
        os.mkdir(f'DFB_VDC_100points_V2V3drawing_{N}')
    except:
        print('File already exist')

    # draw
    # Npoints is the points in the pictures
    Npoints = 100
    # picture numbers = 100000/Npoints

    plt.figure()
    # set start and end point index
    start = i * Npoints
    end = (i + 1) * Npoints
    max_mV = max(data_V2['mV'].iloc[start:end].max(), data_V3['mV'].iloc[start:end].max())
    min_mV = min(data_V2['mV'].iloc[start:end].min(), data_V3['mV'].iloc[start:end].min())
    # draw
    plt.plot(data_V2['Time'].iloc[start:end], data_V2['mV'].iloc[start:end],c='b',label='V2')
    plt.plot(data_V3['Time'].iloc[start:end], data_V3['mV'].iloc[start:end],c='y',label='V3')
    plt.title(f'DFB_VDC_100points_{j}_{i}_V2V3\nStartTime{data_V2["Time"].iloc[start]}')
    plt.ylabel('E (mV)')
    plt.xlabel('Time (min:s.millisec)')
    plt.legend()
    plt.tight_layout()
    # set ylim
    plt.ylim([int(min_mV-100),int(max_mV+100)])
    # plt.show()
    plt.savefig(f'DFB_VDC_100points_V2V3drawing_{N}/DFB_VDC_100points_{j}_{i}_V2V3.jpg')
    plt.close()