import pandas as pd
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# date
year_month_dict = {'2018':[1,2,3,6,7,8,9,10,11,12],
                   '2019':[1,3,5,7,8,9,10,12],
                   '2020':[1,4,5,6,7,8,9,10,11,12],
                   '2021':[1,2,3,4,6,7,8,9],
                   '2022':[1,2,7,10,11]}
day_list = range(1,32)

DustSignal_Insitu = pd.DataFrame()

for year in year_month_dict.keys():
    for month in year_month_dict[year]:
        for day in day_list:
            year = int(year)
            try:
                data = pd.read_csv('MAVEN_DustImpactSignal/model5_DustSignal_Insitu/mvn_lpw_l2_we12burstmf_'
                              f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal_Insitu.csv')
            except:
                # print('No file on this day: ',year,month,day)
                continue

            DustSignal_Insitu = pd.concat([DustSignal_Insitu,data])

print('DustSignal_Insitu.shape ',DustSignal_Insitu.shape)
distance = (DustSignal_Insitu['MSO_X']**2 + DustSignal_Insitu['MSO_Y']**2 + DustSignal_Insitu['MSO_Z']**2)**0.5
distance.columns = ['distance']
DustSignal_Insitu = pd.concat([distance,DustSignal_Insitu],axis=1)
DustSignal_Insitu.drop(columns=['Unnamed: 0'],inplace=True)
DustSignal_Insitu.rename(columns={ 0:'distance'},inplace=True)
print(DustSignal_Insitu.head().to_string())
# print(DustSignal_Insitu.keys())

# total impact numbers =

# basic values

Mars_radius = 3396.2


fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection='3d')
ax.scatter(DustSignal_Insitu['MSO_X'],DustSignal_Insitu['MSO_Y'],DustSignal_Insitu['MSO_Z'],c='b',s=0.5)
ax.set_xlim([-8000,8000])
ax.set_ylim([-8000,8000])
ax.set_zlim([-8000,8000])
ax.set_xlabel('MSO_X (km)')
ax.set_ylabel('MSO_Y (km)')
ax.set_zlabel('MSO_Z (km)')
# draw sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = Mars_radius * np.outer(np.cos(u), np.sin(v))
y = Mars_radius * np.outer(np.sin(u), np.sin(v))
z = Mars_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x,y,z,color='orange',alpha=0.3)
# ax.plot_surface(x*1.5,y*1.5,z*1.5,color='moccasin',alpha=0.3)
# ax.plot_surface(x*2,y*2,z*2,color='oldlace',alpha=0.3)
ax.view_init(25,25)
# plt.title(f'MSO X Y Z\n date {year_month_dict} ')
plt.title(f'MSO X Y Z ')
plt.show()

# MSO X , Z
figure, axes = plt.subplots(figsize=(9,9))
Mars_circle = plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=0,alpha=0.5)
plt.scatter(DustSignal_Insitu['MSO_X'],DustSignal_Insitu['MSO_Z'])
plt.xlim([-8000,8000])
plt.ylim([-8000,8000])
# plt.xlabel(f'MSO_X (km)\n date {year_month_dict}')
plt.xlabel(f'MSO_X (km)')
plt.ylabel('MSO_Z (km)')
axes.add_artist(Mars_circle)
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*1.5 ,color='royalblue',zorder=0,fill=False,label='1.5 Mars Radius'))
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*2 ,color='black',zorder=0,fill=False,label='2 Mars Radius'))
plt.legend()
plt.grid(True)
plt.title('MSO_Z to MSO_X ')
# plt.tight_layout()
plt.show()

# MSO X , Y
figure, axes = plt.subplots(figsize=(9,9))
Mars_circle = plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=0,alpha=0.5)
plt.scatter(DustSignal_Insitu['MSO_X'],DustSignal_Insitu['MSO_Y'])
plt.xlim([-8000,8000])
plt.ylim([-8000,8000])
# plt.xlabel(f'MSO_X (km) \n date {year_month_dict}')
plt.xlabel('MSO_X (km)')
plt.ylabel('MSO_Y (km)')
axes.add_artist(Mars_circle)
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*1.5 ,color='royalblue',zorder=0,fill=False,label='1.5 Mars Radius'))
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*2 ,color='black',zorder=0,fill=False,label='2 Mars Radius'))
plt.legend()
plt.grid(True)
plt.title('MSO_Y to MSO_X ')
# plt.tight_layout()
plt.show()

# MSO X , Z
figure, axes = plt.subplots(figsize=(9,9))
Mars_circle = plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=0,alpha=0.5)
plt.scatter(DustSignal_Insitu['MSO_Y'],DustSignal_Insitu['MSO_Z'])
plt.xlim([-8000,8000])
plt.ylim([-8000,8000])
plt.xlabel(f'MSO_Y (km)\n date {year_month_dict}')
plt.ylabel('MSO_Z (km)')
axes.add_artist(Mars_circle)
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*1.5 ,color='royalblue',zorder=0,fill=False,label='1.5 Mars Radius'))
axes.add_artist(plt.Circle(( 0 , 0 ), Mars_radius*2 ,color='black',zorder=0,fill=False,label='2 Mars Radius'))
plt.legend()
plt.grid(True)
plt.title('MSO_Z to MSO_Y ')
# plt.tight_layout()
plt.show()


# draw local time and distance to Mars
theta = DustSignal_Insitu['LocalTime']/24 * 2*np.pi
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))
ax.scatter(theta, DustSignal_Insitu['distance'])
ax.set_rmax(8000)
ax.set_rticks([Mars_radius,Mars_radius*1.5,Mars_radius*2],labels=['R','1.5R','2R'])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_xticks(ticks=[0,np.pi/2,np.pi,3*np.pi/2],labels=['0h','6h','12h','18h'])
ax.grid(True)
ax.add_artist(plt.Circle(( 0 , 0 ), Mars_radius ,color='moccasin',zorder=0,transform=ax.transData._b))
# ax.set_title(f'Local Time Distribution\n date {year_month_dict}')
ax.set_title(f'Local Time Distribution')
plt.show()

# count dust numbers in an altitude range
count = DustSignal_Insitu.groupby(pd.cut(DustSignal_Insitu['distance'], np.arange(3400,8000,50))).count()
print(count.distance.values)
plt.figure(figsize=(10,10))
plt.scatter(np.arange(3400,7950,50)-Mars_radius,count.distance.values)
# plt.xlabel(f'Height (km, to Mars surface) \n date {year_month_dict}')
plt.xlabel(f'Height (km, to Mars surface)')
plt.ylabel('Impact Count')
plt.title('Impact Count to Height ')
plt.show()