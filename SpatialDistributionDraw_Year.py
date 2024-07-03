import pandas as pd
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# date
year_month_dict = {'2015':[1,2,3,4,5,6,7,8,9,10,11,12],
                   '2018':[1,2,3,6,7,8,9,10,11,12],
                   '2019':[1,3,5,7,8,9,10,12],
                   '2020':[1,4,5,6,7,8,9,10,11,12],
                   '2021':[1,2,3,4,6,7,8,9],
                   '2022':[1,2,7,10,11]}
# year_month_dict = {'2021':[1,2,3,4]}
# year_month_dict = {'2015':[1,2,3,4,5,6,7,8,9,10,11,12]}
# year_month_dict = {'2018':[1,2,3,6,7,8,9,10,11,12]}
# year_month_dict = {'2019':[1,3,5,7,8,9,10,12]}
# year_month_dict = {'2020':[1,4,5,6,7,8,9,10,11,12]}
# year_month_dict = {'2021':[1,2,3,4,6,7,8,9]}
# year_month_dict = {'2022':[1,2,7,10,11]}
day_list = range(1,32)
distance_divide = 300 # km
distance_min = 3400 # km
distance_max = 8000# km
# build empty dataframe/Series
# total durations
LocalTime_totalDuration = pd.Series(dtype=float)
Latitude_Longitude_totalDuration = pd.Series(dtype=float)
Longitude_totalDuration = pd.Series(dtype=float)
Latitude_totalDuration = pd.Series(dtype=float)
MSO_totalDuration = pd.Series(dtype=float)
# counts
LocalTime_Count = pd.Series(dtype=int)
Latitude_Longitude_Count = pd.Series(dtype=int)
Latitude_Count = pd.Series(dtype=int)
Longitude_Count = pd.Series(dtype=int)
MSO_Count = pd.Series(dtype=int)
# daily counts and duration
Daily_Data = pd.DataFrame(columns=['Year','Month','Day','Duration','Count'])
total_Days = 0
total_Days_Impact = 0
for year in year_month_dict.keys():
    for month in year_month_dict[year]:
        print('Month ', year, month,  ' processed')
        for day in day_list:
            year = int(year)
            try:
                # read data Local time, La,Long, MSO, duration
                LocalTime_Duration = pd.read_csv('MAVEN_LocalTime_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                 f'{year:0>4d}{month:0>2d}{day:0>2d}_LocalTime_Duration_Distribution.csv')
                Latitude_Longitude_Duration = pd.read_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Longitude_Duration_Distribution.csv')
                Longitude_Duration = pd.read_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Longitude_Duration_Distribution.csv')
                Latitude_Duration = pd.read_csv('MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Duration_Distribution.csv')
                MSO_Duration = pd.read_csv('MAVEN_MSO_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_MSO_Duration_Distribution.csv')


            except:
                # no duration and count today
                Daily_Data = pd.concat([Daily_Data,pd.DataFrame({'Year':year,'Month':month,'Day':day,'Duration':0,'Count':0},index=[0])]
                                       ,ignore_index=True)
                continue

            # Time Duration process

            LocalTime_Duration['TimeDuration'] = pd.to_timedelta(
                LocalTime_Duration['TimeDuration']).dt.total_seconds() / 3600.
            if LocalTime_Duration['TimeDuration'].max() > 24:
                print(f'Wrong timedelta on day: {year,month,day}, > 24h !')
                continue
            Latitude_Longitude_Duration['TimeDuration'] = pd.to_timedelta(
                Latitude_Longitude_Duration['TimeDuration']).dt.total_seconds() / 3600.
            Longitude_Duration['TimeDuration'] = pd.to_timedelta(
                Longitude_Duration['TimeDuration']).dt.total_seconds() / 3600.
            Latitude_Duration['TimeDuration'] = pd.to_timedelta(
                Latitude_Duration['TimeDuration']).dt.total_seconds() / 3600.
            MSO_Duration['TimeDuration'] = pd.to_timedelta(MSO_Duration['TimeDuration']).dt.total_seconds() / 3600.

            # Processing Duration
            if LocalTime_totalDuration.empty:
                LocalTime_totalDuration = LocalTime_Duration

            else:
                LocalTime_totalDuration = LocalTime_totalDuration+LocalTime_Duration


            if Latitude_Longitude_totalDuration.empty:
                Latitude_Longitude_totalDuration = Latitude_Longitude_Duration
            else:
                Latitude_Longitude_totalDuration = Latitude_Longitude_totalDuration+Latitude_Longitude_Duration

            if Longitude_totalDuration.empty:
                Longitude_totalDuration = Longitude_Duration
            else:
                Longitude_totalDuration = Longitude_totalDuration+Longitude_Duration

            if Latitude_totalDuration.empty:
                Latitude_totalDuration = Latitude_Duration
            else:
                Latitude_totalDuration = Latitude_totalDuration+Latitude_Duration

            if MSO_totalDuration.empty:
                MSO_totalDuration = MSO_Duration
            else:
                MSO_totalDuration = MSO_totalDuration+MSO_Duration
            total_Days += 1

            # Processing Dust signal counts
            try:
                Dust_LocalTime = pd.read_csv(
                    'MAVEN_DustImpactSignal/MAVEN_LocalTime_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                    f'{year:0>4d}{month:0>2d}{day:0>2d}_LocalTime_Duration_Distribution.csv')
                Dust_Latitude_Longitude = pd.read_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                                   f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Longitude_Duration_Distribution.csv')
                Dust_Longitude = pd.read_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                          f'{year:0>4d}{month:0>2d}{day:0>2d}_Longitude_Duration_Distribution.csv')
                Dust_Latitude = pd.read_csv('MAVEN_DustImpactSignal/MAVEN_Latitude_Longitude_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                         f'{year:0>4d}{month:0>2d}{day:0>2d}_Latitude_Duration_Distribution.csv')
                Dust_MSO = pd.read_csv('MAVEN_DustImpactSignal/MAVEN_MSO_Duration_Distribution/mvn_lpw_l2_we12burstmf_'
                                    f'{year:0>4d}{month:0>2d}{day:0>2d}_MSO_Duration_Distribution.csv')
            except:
                # no count today
                Daily_Data = pd.concat([Daily_Data,pd.DataFrame({'Year': year, 'Month': month, 'Day': day,
                                                'Duration': LocalTime_Duration['TimeDuration'].sum(), 'Count': 0},index=[0])],ignore_index=True)
                continue


            if LocalTime_Count.empty:
                LocalTime_Count = Dust_LocalTime

            else:
                LocalTime_Count = Dust_LocalTime+LocalTime_Count


            if Latitude_Longitude_Count.empty:
                Latitude_Longitude_Count = Dust_Latitude_Longitude
            else:
                Latitude_Longitude_Count = Dust_Latitude_Longitude+Latitude_Longitude_Count

            if Longitude_Count.empty:
                Longitude_Count = Dust_Longitude
            else:
                Longitude_Count = Dust_Longitude+Longitude_Count

            if Latitude_Count.empty:
                Latitude_Count = Dust_Latitude
            else:
                Latitude_Count = Dust_Latitude+Latitude_Count

            if MSO_Count.empty:
                MSO_Count = Dust_MSO
            else:
                MSO_Count = Dust_MSO+MSO_Count
            # Daily counts
            Daily_Data = pd.concat([Daily_Data,pd.DataFrame({'Year': year, 'Month': month, 'Day': day,
                                            'Duration': LocalTime_Duration['TimeDuration'].sum(),
                                            'Count': Dust_LocalTime['Count'].sum()},index=[0])],ignore_index=True)
            total_Days_Impact += 1
            # print('Day ',year,month,day,' processed')


# basic values
Mars_radius = 3396.2
print('MarsRadius ',Mars_radius)
print('Total Days = ',total_Days)
print('Total Days have Impact = ',total_Days_Impact)

# local time drawing
# notice that the data is cut in LocalTime, distance
Localtime_theta = np.arange(0.5,24.5,1)/24*2*np.pi
distance = np.arange(distance_min,distance_max-distance_divide,distance_divide)
time, d = np.meshgrid(Localtime_theta,distance)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(time,d,LocalTime_totalDuration['TimeDuration'].to_numpy().reshape(len(Localtime_theta),len(distance)).T
                   ,cmap='OrRd')
              #cmap='bone',vmin=LocalTime_Duration['TimeDuration'].min(),vmax=int(LocalTime_Duration['TimeDuration'].max()))
ax.set_rmax(distance_max-distance_divide)
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[0,np.pi/2,np.pi,3*np.pi/2],labels=['0h','6h','12h','18h'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label('Duration (hour)')
plt.title('LocalTime Duration\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_LocalTime_Duration.jpg',dpi=400)
plt.show()

# Local Time Impact Rate
ImpactRate_LocalTime = LocalTime_Count['Count']/LocalTime_totalDuration['TimeDuration']
ImpactRate_LocalTime.fillna(0,inplace=True)
# print(ImpactRate_LocalTime.describe())
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(time,d,ImpactRate_LocalTime.to_numpy().reshape(len(Localtime_theta),len(distance)).T,
                   cmap='OrRd',vmax=10)
ax.set_rmax(distance_max-distance_divide)
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[0,np.pi/2,np.pi,3*np.pi/2],labels=['0h','6h','12h','18h'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label(label='Impact Rate (number/hour)')
plt.title('LocalTime ImpactRate Distribution\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_LocalTime_ImpactRate_Distribution.jpg',dpi=400)
plt.show()

# draw Local Time with impact rate
LocalTime_rate = LocalTime_Count['Count'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=1)/\
              LocalTime_totalDuration['TimeDuration'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=1)

shft = 0.24
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(np.arange(24)+shft,LocalTime_rate,shft*2,color='orange')
ax1.set_ylabel('Impact Rate (number/hour)',color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_xticks(range(0,24),[str(r) for r in range(0,24)])
ax1.set_xlabel('Local Time (24h)\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
ax2 =ax1.twinx()
ax2.bar(np.arange(24)+shft*3,LocalTime_Count['Count'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=1),shft*2,color='royalblue')
ax2.set_ylabel('Impact Numbers',color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')
plt.title('Local Time Impact Rate & Numbers')
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_LocalTime_ImpactRate_Numbers.jpg',dpi=400)
plt.show()

# draw distance to impact rate
Height_Rate = LocalTime_Count['Count'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=0)/\
              LocalTime_totalDuration['TimeDuration'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=0)

shft = 0.24
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(np.arange(15)+shft,Height_Rate,shft*2,color='seagreen')
ax1.set_ylabel('Impact Rate (number/hour)',color='seagreen')
ax1.tick_params(axis='y', labelcolor='seagreen')
ax1.set_xticks(range(15),[str(int(r)) for r in distance-Mars_radius-3])
ax1.set_xlabel('Height to Mars Surface (km)\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
ax2 =ax1.twinx()
ax2.bar(np.arange(15)+shft*3,LocalTime_Count['Count'].to_numpy().reshape(len(Localtime_theta),len(distance)).sum(axis=0),shft*2,color='slateblue')
ax2.set_ylabel('Impact Numbers',color='slateblue')
ax2.tick_params(axis='y', labelcolor='slateblue')
plt.title('Height Impact Rate & Numbers')
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_Height_Impact_Rate_Numbers.jpg',dpi=400)
plt.show()

# draw local time count distribution
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(time,d,LocalTime_Count['Count'].to_numpy().reshape(len(Localtime_theta),len(distance)).T,
                   cmap='OrRd',vmax=40)
ax.set_rmax(distance_max-distance_divide)
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[0,np.pi/2,np.pi,3*np.pi/2],labels=['0h','6h','12h','18h'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label(label='Impact Numbers')
plt.title('LocalTime ImpactNumber Distribution\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_LocalTime_ImpactNumber_Distribution.jpg',dpi=400)
plt.show()

# Draw latitude Impact rate distribution
ImpactRate_Latitude = Latitude_Count['Count']/Latitude_totalDuration['TimeDuration']
ImpactRate_Latitude.fillna(0,inplace=True)
Latitude_range = np.arange(-85,95,10)/360*2*np.pi
Latitude, d = np.meshgrid(Latitude_range,distance)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(Latitude,d,ImpactRate_Latitude.to_numpy().reshape(len(Latitude_range),len(distance)).T,cmap='OrRd',vmax=2)
ax.set_rmax(distance_max-distance_divide)
ax.set_xlim([-np.pi/2,np.pi/2])
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[-np.pi/2,-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3,np.pi/2],labels=['-90','-60','-30','0','30','60','90'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label(label='Impact Rate (number/hour)')
plt.title('Latitude ImpactRate Distribution\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_Latitude_ImpactRate_Distribution.jpg',dpi=400)
plt.show()

# draw latitude count number distribution
Latitude_range = np.arange(-85,95,10)/360*2*np.pi
Latitude, d = np.meshgrid(Latitude_range,distance)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(Latitude,d,Latitude_Count['Count'].to_numpy().reshape(len(Latitude_range),len(distance)).T,cmap='OrRd')
ax.set_rmax(distance_max-distance_divide)
ax.set_xlim([-np.pi/2,np.pi/2])
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[-np.pi/2,-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3,np.pi/2],labels=['-90','-60','-30','0','30','60','90'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label(label='Impact Numbers')
plt.title('Latitude ImpactNumbers Distribution\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_Latitude_ImpactNumbers_Distribution.jpg',dpi=400)
plt.show()

# draw latitude duration
Latitude_range = np.arange(-85,95,10)/360*2*np.pi
Latitude, d = np.meshgrid(Latitude_range,distance)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
im = ax.pcolormesh(Latitude,d,Latitude_totalDuration['TimeDuration'].to_numpy().reshape(len(Latitude_range),len(distance)).T,cmap='OrRd')
ax.set_rmax(distance_max-distance_divide)
ax.set_xlim([-np.pi/2,np.pi/2])
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['Mars','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[-np.pi/2,-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3,np.pi/2],labels=['-90','-60','-30','0','30','60','90'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=1,transform=ax.transData._b))
cbar = fig.colorbar(im)
cbar.set_label(label='Duration (hours)')
plt.title('Latitude Duration Distribution\n'+'Year: '+''.join([str(y)+' ' for y in year_month_dict.keys()]))
# save to each year file folders
if len(year_month_dict.keys()) == 1:
    year = int([year for year in year_month_dict.keys()][0])
    plt.savefig(f'MAVEN_DustImpactSignal/model5_DustSignalPicture/{year}_Distribution/{year:0>4d}_Latitude_Duration_Distribution.jpg',dpi=400)
plt.show()