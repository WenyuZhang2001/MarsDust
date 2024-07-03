import numpy as np
import pandas as pd
month = 1
day = 1
year =2021
print('%.4d%.2d%.2d'%(year,month,day))
print(f'{year:0>4d}{month:0>2d}{day:0>2d}')
print(np.arange(0,370,10))
print(pd.DataFrame({'year':2000,'month':1,'day':1},index=[0]))
print( len(np.arange(-85,95,10)))

year_month_dict = {'2018':[1,2,3,6,7,8,9,10,11,12],
                   '2019':[1,3,5,7,8,9,10,12],
                   '2020':[1,4,5,6,7,8,9,10,11,12],
                   '2021':[1,2,3,4,6,7,8,9],
                   '2022':[1,2,7,10,11]}

# year_month_dict = {'2015':[1,2,3,4,5,6,7,8,9,10,11,12]}
# print(year_month_dict.keys())
# year = [year for year in year_month_dict.keys()][0]
print(year)
print(len(year_month_dict.keys()))