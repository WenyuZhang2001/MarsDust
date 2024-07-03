# read and draw 100000 points data
import pandas as pd
import matplotlib.pyplot as plt
import os

temp = True
j = 0 # group number
statues = 1 # initial statues, to ask if continue
while temp:
    # ask if continue the program
    if j%100 == 0:
        statues = input('input num : 1/0 ')

    if statues == '0':
        break

    # load data
    test_data = pd.read_csv(f'DFB_VDC_100000points_V2/DFB_VDC_100000points_V2_{j}.csv')
    test_data.drop(columns=['Unnamed: 0'], inplace=True)
    # simple operations
    # test_data.columns = ['Day', 'Time', 'V']
    # test_data['Time'] = test_data['Time'].str.replace('.', ',', 1)
    # test_data['Time'] = test_data['Time'].str.replace('.', '')
    # test_data['Time'] = test_data['Time'].str.replace(',', '.')
    # test_data['Time'] = test_data['Day'] + test_data['Time']
    # test_data['Time'] = pd.to_datetime(test_data['Time'], format='%d-%m-%Y%H:%M:%S.%f')
    # test_data['V'] = test_data['V'] * 1000  # convert to mV
    # test_data.columns = ['Day', 'Time', 'mV']
    test_data.columns = ['Time','V']
    test_data['Time'] = pd.to_datetime(test_data['Time'],format='%Y-%m-%dT%H:%M:%S.%fZ')
    test_data['V'] = test_data['V'] * 1000
    test_data.columns = ['Time','mV']
    # get the max and min value of this group, to set ylim
    max_mV = test_data['mV'].max()
    min_mV = test_data['mV'].min()
    # print(test_data.describe())
    # print(test_data.head())
    # check if the dir exist, if not, set up
    try:
        os.mkdir(f'DFB_VDC_100000points_V2_drawing/{j}')
    except:
        print('File already exist')

    # draw
    # Npoints is the points in the pictures
    Npoints = 100
    # picture numbers = 100000/Npoints
    for i in range(int(100000 / Npoints)):
        plt.figure()
        print(f'Total {i} pictures,group {j}')
        # set star and end point index
        start = i * Npoints
        end = (i + 1) * Npoints
        # draw
        plt.plot(test_data['Time'].iloc[start:end], test_data['mV'].iloc[start:end],color='red')
        plt.title(f'DFB_VDC_100000points_{j}_{i}')
        plt.ylabel('E (mV)')
        plt.xlabel('Time (min:s.millisec)')
        plt.tight_layout()
        # set ylim
        plt.ylim([int(min_mV-100),int(max_mV+100)])
        # plt.show()
        plt.savefig(f'DFB_VDC_100000points_V2_drawing/{j}/DFB_VDC_100points_{j}_{i}.jpg')
        plt.close()
    j = j + 1