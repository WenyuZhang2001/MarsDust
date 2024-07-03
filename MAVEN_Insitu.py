import pandas as pd
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

# date
# year_list = [2018,2019,2020,2021,2022]
year_list = [2015]

month_list = range(1,13)
day_list = range(1,32)

for year in year_list:
    for month in month_list:
        for day in day_list:
            # load dust signal data and Insitu data

            try:
                DustSignal = pd.read_csv('MAVEN_DustImpactSignal/model5_DustSignal/mvn_lpw_l2_we12burstmf_'
                                     f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal.csv')

                Insitu = pd.read_table(f'MAVEN_Insitu/MAVEN_Insitu_'
                                       f'{year:0>4d}_{month:0>2d}/mvn_kp_insitu_{year:0>4d}{month:0>2d}{day:0>2d}_v18_r03.tab',
                                       comment='#', header=None, sep='\s+')

            except:
                print(f'No file on this day: {year:0>4d}{month:0>2d}{day:0>2d}')
                continue

            Insitu[0] = pd.to_datetime(Insitu[0], format='%Y-%m-%dT%H:%M:%S')
            SignalNumbers = len(DustSignal)
            if SignalNumbers == 0:
                continue
            SignalLength = 1500
            Insitu_Data_df = pd.DataFrame()

            for i in range(SignalNumbers):
                # get time from dust signal
                time = datetime.strptime(DustSignal['StartTime'].iloc[i],'%Y-%m-%d %H:%M:%S.%f')
                # compare time, get the first time timedelta in 8 seconds
                # Four- to eight-sec cadence, all parameters on uniform time grid
                # the Insitu time resolution is max 8s
                InsituData = Insitu.loc[abs((Insitu[0]-time))<timedelta(seconds=8)].head(1)
                # print(InsituData.to_string())
                '''
                number start from 1
                196 local time
                193 SPICE:Spacecraft GEO Longitude
                194 SPICE:Spacecraft GEO Latitude
                190-192 SPICE:Spacecraft MSO X, Y, Z
                
                '''

                Insitu_Data_df = pd.concat([Insitu_Data_df,InsituData[[195,193,192,189,190,191]]],ignore_index=True)

            Insitu_Data_df.columns = ['LocalTime','GEO_Latitude','GEO_Longitude','MSO_X','MSO_Y','MSO_Z']
            # print(Insitu_Data_df)
            # concat insitu and signal data
            DustSignal = pd.concat([Insitu_Data_df,DustSignal],axis=1)
            DustSignal.drop(columns=['Unnamed: 0'],inplace=True)
            print(DustSignal.iloc[:,:9].to_string())
            # save as csv
            DustSignal.to_csv('MAVEN_DustImpactSignal/model5_DustSignal_Insitu/mvn_lpw_l2_we12burstmf_'
                                  f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal_Insitu.csv')