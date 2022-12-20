# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:26:54 2021

@author: user
"""

import pandas as pd
import numpy as np
import csv
import haversine as hs
from haversine import Unit
import math
import numpy as np
import matplotlib.pyplot as plt

'''Read the weather station and temperature data for the year 1986. The station data identifies the station
and its GPS location. The temperature data records the temperature at a particular station on a particular
day.
 '''
df_st = pd.read_csv("stations.csv")
df_tp = pd.read_csv("temp.csv")



'''Filter and clean up the data: Some temperatures are missing, some stations identifiers may be null
(missing), and some GPS coordinates (e.g., 0.0 / 0.0) are clearly invalid. You can ignore the WBAN
identifier and just focus on the STATION identifier.
'''

dff1=df_st.dropna()

dft=df_tp.dropna()


print(dft)

m,n=dff1.shape


dd={}
dk={}
data={}
dlt={}
dlg={}
sid={}
dkc={}
c=1

          
df=pd.DataFrame(dff1)

df=df.reset_index()
df=df.drop(['index'],axis=1)
print('******************Processing*********************')
'''Identify all weather stations within 100 km of Cape Canaveral. Calculate the distance using the Haversine
distance function which takes into account the curvature of the Earth.   (Note: not all of the stations
necessarily recorded a temperature on any given day.)  
'''

for i in range(0,m):
    
       dd[i]=hs.haversine([28.396837,-80.60605659],[df.iloc[i,1],df.iloc[i,2]],unit=Unit.METERS)
       dk[i]=dd[i]/100000
       
       if dk[i]>100:
          
          
          df.iloc[i,1]=0
          df.iloc[i,2]=0
          df.iloc[i,0]=0
          


d=df.replace(0, np.nan)
d=d.dropna()
d=d.reset_index()
d=d.drop(['index'],axis=1)


x=dft.SID.isin(d.SID).astype(int)

dft.insert(4, 'NBS', x.values)
dft.insert(5, 'LT',d['LT'])
dft.insert(6, 'LG',d['LG'])
dd=dft.replace(0, np.nan)
dd=dd.dropna()
dd=dd.reset_index()
dd=dd.drop(['index'],axis=1)
#print(dd)
dd.to_csv('merged2.csv')



def idwr(x, y, z, xi, yi):
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d=hs.haversine([28.396837,-80.60605659],[x[s],y[s]])
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u=sumsup/suminf
        xyzi = [xi[p], yi[p],u]
        lstxyzi.append(xyzi)
    return(lstxyzi)
x=np.array(dd.iloc[:,5])
y=np.array(dd.iloc[:,6])
z=np.array(dd.iloc[:,3])
''' 
Use inverse distance weighting (https://gisgeography.com/inverse-distance-weighting-idw-interpolation/)
with p=1 to estimate the temperature at Cape Canaveral on January 28, 1986.   The idea here is that in 
estimating the temperature at Cape Canaveral, we want to give more weight to temperature readings
from stations that are closer to the site of the shuttle launch.
'''
xyz=idwr(x,y,z,[28.396837],[-80.60605659])
print(xyz)


plt.figure()
plt.imshow(xyz, extent=(x.min(), x.max(), y.max(), y.min()))
plt.scatter(x,y,c=z)
plt.colorbar()
plt.title('Inverse Distance Weighting (IDW) Interpolation')
plt.show()

'''Plot the temperature at Cape Canaveral for every day in January 1986.   '''

x1=np.array(dd.iloc[:,1])
y1=np.array(dd.iloc[:,2])
plt.figure()
plt.scatter(x1,y1,c=z)
plt.colorbar()
plt.title('Temperature for every day in 1986')
plt.show()

''' 
Plot the NumPy array as an image plot.  Repeat for both January 28
th
, 1986 and February 1
, 1986. Note
how the temperatures in Florida had warmed up considerably after just four days.   

st'''


d_28_jan = dd[dd.M == 1]
print(d_28_jan)
d_28_jan1 = d_28_jan[d_28_jan.D == 28]

print(d_28_jan1)
xx=np.array(d_28_jan1.iloc[:,5])
yy=np.array(d_28_jan1.iloc[:,6])
zz=np.array(d_28_jan1.iloc[:,3])
plt.figure()
plt.imshow(xyz, extent=(xx.min(), xx.max(), yy.max(), yy.min()))
plt.scatter(xx,yy,c=zz)
plt.colorbar()
plt.title('Temperature plot on 28th Jan 1986')
plt.show()


d_1_feb = dd[dd.M == 2]
print(d_1_feb)
d_1_feb = d_1_feb[d_1_feb.D == 1]

print(d_28_jan1)
xx1=np.array(d_1_feb.iloc[:,5])
yy1=np.array(d_1_feb.iloc[:,6])
zz1=np.array(d_1_feb.iloc[:,3])
plt.figure()
plt.imshow(xyz, extent=(xx1.min(), xx1.max(), yy1.max(), yy1.min()))
plt.scatter(xx1,yy1,c=zz1)
plt.colorbar()
plt.title('Temperature plot on 1st Feb 1986')
plt.show()
