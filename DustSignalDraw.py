import pandas as pd
import matplotlib.pyplot as plt

# Draw dust signals
# load
# year_month_dict = {'2018':[1,2,3,6,7,8,9,10,11,12],
#                    '2019':[1,3,5,7,8,9,10,12],
#                    '2020':[1,4,5,6,7,8,9,10,11,12],
#                    '2021':[1,2,3,4,6,7,8,9],
#                    '2022':[1,2,7,10,11]}
year_month_dict = {'2015':[1,2,3,4,5,6,7,8,9,10,11,12]}

day_list = range(1,32)

for year in year_month_dict.keys():
    for month in year_month_dict[year]:
        for day in day_list:
            year = int(year)
            try:
                DustSignal = pd.read_csv('MAVEN_DustImpactSignal/model5_DustSignal/mvn_lpw_l2_we12burstmf_'
                                     f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal.csv')
            except:
                print(f'No file: MAVEN_DustImpactSignal/'
                      f'model5_DustSignal/mvn_lpw_l2_we12burstmf_{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal.csv')
                continue


            SignalNumbers = len(DustSignal)
            SignalLength = 1500
            for i in range(SignalNumbers):
                StartTime = DustSignal['StartTime'].iloc[i]
                Time = pd.date_range(DustSignal['StartTime'].iloc[i],DustSignal['EndTime'].iloc[i],periods=SignalLength)
                plt.figure(figsize=(16,9))
                plt.plot(Time,DustSignal.iloc[i,-SignalLength:])
                plt.ylabel('E (mV/m)')
                plt.xlabel(f'Time (StartTime {StartTime})')
                plt.title('MAVEN Dust Impact Signal')
                plt.tight_layout()
                plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}/{year:0>4d}{month:0>2d}{day:0>2d}__{i}__DustImpactSignal.jpg',dpi=200)
                plt.close()
                # plt.show()