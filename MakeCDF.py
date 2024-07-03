import pandas as pd
import cdflib
from spacepy import pycdf
import datetime
import numpy as np
time = [datetime.datetime(2000,10,1,1,val) for val in range(60)]
data = np.random.random_sample(len(time))
cdf = pycdf.CDF('MyCDF.cdf','')
cdf['Epoch'] = time
cdf['data'] = data
cdf.attrs['Auther'] = 'WY Z'
cdf.attrs['CreateDate'] = datetime.datetime.now()
cdf['data'].attrs['units'] = 'MeV'
cdf.close()
