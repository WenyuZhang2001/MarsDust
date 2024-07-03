import pandas as pd
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# date
# year_month_dict = {'2019':[1,3,5,7,8,9,10,12],
#                    '2020':[1,4,5,6,7,8,9,10,11,12],
#                    '2021':[1,2,3,4,6,7,8,9],
#                    '2022':[1,2,7,10,11]}

# year_month_dict = {'2015':[1,2,3,4,5,6,7,8,9,10,11,12]}
# year_month_dict = {'2021':[1]}
year_month_dict = {'2018':[1,2,3,6,7,8,9,10,11,12]}
day_list = range(1,32)
distance_divide = 300 # km
distance_min = 3400 # km
distance_max = 8000# km
for year in year_month_dict.keys():
    for month in year_month_dict[year]:
        for day in day_list:
            year = int(year)
            try:
                # read Insitu data and Signal Data
                Insitu_Data = pd.read_table('MAVEN_Insitu/MAVEN_Insitu_'
                                   f'{year:0>4d}_{month:0>2d}/mvn_kp_insitu_{year:0>4d}{month:0>2d}{day:0>2d}_v18_r03.tab',
                                    comment='#', header=None, sep='\s+')

                SignalData = pd.read_csv(f'MAVEN_SignalProbability_model5/mvn_lpw_l2_we12burstmf_'
                                 f'{year:0>4d}{month:0>2d}{day:0>2d}_SignalProbability.csv')
                SignalData.drop(columns=['Unnamed: 0.1','Unnamed: 0'], inplace=True)
            except:
                # print('No file on this day: ',year,month,day)
                continue

            print('Processing Date: ',year,month,day)
            '''
                            number start from 1
                            196 local time
                            193 SPICE:Spacecraft GEO Longitude
                            194 SPICE:Spacecraft GEO Latitude
                            190-192 SPICE:Spacecraft MSO X, Y, Z
                            0 UTC Time

                            '''
            Insitu_Data = Insitu_Data[[0,195,193,192,189,190,191]]
            Insitu_Data.columns = ['UTC_Time','LocalTime','GEO_Latitude','GEO_Longitude','MSO_X','MSO_Y','MSO_Z']
            # change UTC Time format
            Insitu_Data['UTC_Time'] = pd.to_datetime(Insitu_Data['UTC_Time'],format='%Y-%m-%dT%H:%M:%S')
            # calculate distance
            Insitu_Data['distance'] = (Insitu_Data['MSO_X']**2 + Insitu_Data['MSO_Y']**2
                                       + Insitu_Data['MSO_Z']**2)**0.5

            # Signal Data Time format
            SignalData['StartTime'] = pd.to_datetime(SignalData['StartTime'],format='%Y-%m-%d %H:%M:%S.%f')
            SignalData['EndTime'] = pd.to_datetime(SignalData['EndTime'],format='%Y-%m-%d %H:%M:%S.%f')
            #
            SignalNumbers = len(SignalData)

            Insitu_Data_Match = pd.DataFrame()

            for i in range(SignalNumbers):
                # Match Data
                # time = datetime.strptime(SignalData['StartTime'].iloc[i], '%Y-%m-%d %H:%M:%S.%f')
                InsituData_temp = Insitu_Data.loc[abs((Insitu_Data['UTC_Time'] - SignalData['StartTime'].iloc[i]))
                                                  < timedelta(seconds=8)].head(1)
                # add
                Insitu_Data_Match = pd.concat([InsituData_temp,Insitu_Data_Match],ignore_index=True)

            Signal_Insitu_Data = pd.concat([Insitu_Data_Match,SignalData[['StartTime','EndTime']]],axis=1)
            Signal_Insitu_Data['TimeDuration'] = Signal_Insitu_Data['EndTime']-Signal_Insitu_Data['StartTime']

            print(Signal_Insitu_Data.head().to_string())

            # local time distribution
            LocalTime_Distribution = Signal_Insitu_Data.groupby([pd.cut(Signal_Insitu_Data['LocalTime'],np.arange(0,25,1)),
                                                                 pd.cut(Signal_Insitu_Data['distance'], np.arange(distance_min,distance_max,distance_divide))])

            LocalTime_Duration_Distribution = LocalTime_Distribution['TimeDuration'].sum()
            LocalTime_Duration_Distribution.to_csv('MAVEN_LocalTime_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                          f'{year:0>4d}{month:0>2d}{day:0>2d}_LocalTime_Duration_Distribution.csv')

            # spatial La and Long distribution
            Latitude_Longitude_Distribution = Signal_Insitu_Data.groupby([pd.cut(Signal_Insitu_Data['GEO_Longitude'],np.arange(0,360+10,10)),
                                                                          pd.cut(Signal_Insitu_Data['GEO_Latitude'],np.arange(-90,90+10,10)),
                                                                          pd.cut(Signal_Insitu_Data['distance'], np.arange(distance_min,distance_max,distance_divide))])
            Latitude_Longitude_Distribution = Latitude_Longitude_Distribution['TimeDuration'].sum()
            Latitude_Longitude_Distribution.to_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Longitude_Duration_Distribution.csv')

            # long distribution
            Longitude_Distribution = Signal_Insitu_Data.groupby([pd.cut(Signal_Insitu_Data['GEO_Longitude'],np.arange(0,360+10,10)),
                                                                 pd.cut(Signal_Insitu_Data['distance'], np.arange(distance_min,distance_max,distance_divide))])
            Longitude_Distribution = Longitude_Distribution['TimeDuration'].sum()
            Longitude_Distribution.to_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Longitude_Duration_Distribution.csv')

            # latitude distribution
            Latitude_Distribution = Signal_Insitu_Data.groupby([pd.cut(Signal_Insitu_Data['GEO_Latitude'], np.arange(-90,90+10, 10)),
                                                                pd.cut(Signal_Insitu_Data['distance'], np.arange(distance_min, distance_max, distance_divide))])
            Latitude_Distribution = Latitude_Distribution['TimeDuration'].sum()
            Latitude_Distribution.to_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Duration_Distribution.csv')
            # MSO distribution
            MSO_Distribution = Signal_Insitu_Data.groupby([pd.cut(Signal_Insitu_Data['MSO_X'], np.arange(-10000, 10000, 1000)),
                                                           pd.cut(Signal_Insitu_Data['MSO_Y'],np.arange(-10000, 10000, 1000)),
                                                           pd.cut(Signal_Insitu_Data['MSO_Z'], np.arange(-10000, 10000, 1000))])
            MSO_Distribution = MSO_Distribution['TimeDuration'].sum()
            MSO_Distribution.to_csv('MAVEN_MSO_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_MSO_Duration_Distribution.csv')

            # Processing Dust Impact Signal
            try:
                DustImpactSignal = pd.read_csv('MAVEN_DustImpactSignal/model5_DustSignal_Insitu/mvn_lpw_l2_we12burstmf_'
                              f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal_Insitu.csv')
            except:
                print('No dust Impact Signal on day: ',year,month,day)
                continue

            DustImpactSignal['distance'] = (DustImpactSignal['MSO_X'] ** 2 + DustImpactSignal['MSO_Y'] ** 2 +
                                            DustImpactSignal['MSO_Z'] ** 2) ** 0.5

            # Local Time
            Dust_LocalTime = DustImpactSignal[['StartTime']].groupby(
                                                            [pd.cut(DustImpactSignal['LocalTime'], np.arange(0, 25, 1)),
                                                             pd.cut(DustImpactSignal['distance'], np.arange(distance_min, distance_max, distance_divide))])

            Dust_LocalTime = Dust_LocalTime.count()
            Dust_LocalTime.rename(columns={'StartTime':'Count'},inplace=True)
            Dust_LocalTime.to_csv('MAVEN_DustImpactSignal/MAVEN_LocalTime_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_LocalTime_Duration_Distribution.csv')

            # Latitude Longitude
            Dust_Latitude_Longitude = DustImpactSignal[['StartTime']].groupby([pd.cut(DustImpactSignal['GEO_Longitude'],np.arange(0,360+10,10)),
                                                                               pd.cut(DustImpactSignal['GEO_Latitude'],np.arange(-90,90+10,10)),
                                                                               pd.cut(DustImpactSignal['distance'], np.arange(distance_min,distance_max,distance_divide)),])
            Dust_Latitude_Longitude = Dust_Latitude_Longitude.count()
            Dust_Latitude_Longitude.rename(columns={'StartTime':'Count'},inplace=True)
            Dust_Latitude_Longitude.to_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Longitude_Duration_Distribution.csv')

            # longitude
            Dust_Longitude = DustImpactSignal[['StartTime']].groupby(
                                                                [pd.cut(DustImpactSignal['GEO_Longitude'], np.arange(0, 360 + 10, 10)),
                                                                 pd.cut(DustImpactSignal['distance'], np.arange(distance_min, distance_max, distance_divide))])
            Dust_Longitude = Dust_Longitude.count()
            Dust_Longitude.rename(columns={'StartTime':'Count'},inplace=True)
            Dust_Longitude.to_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_Longitude_Duration_Distribution.csv')

            # Latitude
            Dust_Latitude = DustImpactSignal[['StartTime']].groupby(
                                                            [pd.cut(DustImpactSignal['GEO_Latitude'], np.arange(-90, 90 + 10, 10)),
                                                             pd.cut(DustImpactSignal['distance'], np.arange(distance_min, distance_max,distance_divide))])
            Dust_Latitude = Dust_Latitude.count()
            Dust_Latitude.rename(columns={'StartTime':'Count'},inplace=True)
            Dust_Latitude.to_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                         f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Duration_Distribution.csv')

            # MSO
            Dust_MSO = DustImpactSignal[['StartTime']].groupby(
                                                    [pd.cut(DustImpactSignal['MSO_X'], np.arange(-10000, 10000, 1000)),
                                                    pd.cut(DustImpactSignal['MSO_Y'], np.arange(-10000, 10000, 1000)),
                                                    pd.cut(DustImpactSignal['MSO_Z'], np.arange(-10000, 10000, 1000))])
            Dust_MSO = Dust_MSO.count()
            Dust_MSO.rename(columns={'StartTime':'Count'},inplace=True)
            Dust_MSO.to_csv('MAVEN_DustImpactSignal/MAVEN_MSO_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                    f'{year:0>4d}{month:0>2d}{day:0>2d}_MSO_Duration_Distribution.csv')


