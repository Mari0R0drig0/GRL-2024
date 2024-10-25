import numpy as np
import math

from pathlib import Path
from netCDF4 import Dataset as ncdf
from netCDF4 import num2date, date2num 
from datetime import date, timedelta, datetime

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from windspharm.examples import example_data_path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point


def seasonal_value(nlat, nlon, nyr, field, nmonth):
  """Compute the seasonal value of the field.
  nlat: number of latitudes
  nlon: number of longitudes
  nyr: number of years
  field: field where we compute the seasonal anomalies
  nmonth: months that define the season"""
    
    value = np.zeros([nyr, nlat, nlon])
    for i in range (nyr):
        tmp = field[12*(i)+nmonth[0]:12*(i)+nmonth[-1],:,:] #select season (previous or next)
        value[i,:,:] = np.mean(tmp,axis=0) #season field mean
    
    return value

def t_test(nlat,nlon,field,indices_1,indices_2,pval,equal_variances):
    pvalue = np.empty([nlat,nlon])
    test = np.empty([nlat,nlon])
    for i in range (nlat):
        for j in range (nlon):
            pvalue = stats.ttest_ind(field[indices_1,i,j],field[indices_2,i,j],axis=0,equal_var=equal_variances,nan_policy='propagate').pvalue
            if (pvalue<=pval): test[i,j] = 1
            else: test[i,j] = 0
    return test

# El Niño DJF QBO W JJASON
# ONI NOAA 0.5 K: 1957, 1963, 1969, 1982, 1986, 1987, 1997, 2002, 2004, 2006, 2009, 2015 (12)
#indices_W = np.array([1957, 1963, 1969, 1982, 1986, 1987, 1997, 2002, 2004, 2006, 2009, 2015]) - 1950

# El Niño DJF QBO E JJASON
# ONI NOAA 0.5 K: 1951, 1953, 1958, 1965, 1968, 1972, 1976, 1977, 1979, 1991, 1994, 2014, 2018 (13)
#indices_E = np.array([1951, 1953, 1958, 1965, 1968, 1972, 1976, 1977, 1979, 1991, 1994, 2014, 2018]) - 1950

# QBO JJA
# W: 1953 1955 1964 1967 1969 1973 1976 1978 1983 1985 1986 1988 1993 1995 1997 1999 2000 2002 2004 2008 2009 2011 2017 2019 (24)
indices_W = np.array([1953, 1955, 1964, 1967, 1969, 1973, 1976, 1978, 1983, 1985, 1986, 1988, 1993, 1995, 1997, 1999, 2000, 2002, 2004, 2008, 2009, 2011, 2017, 2019]) - 1950

# E: 1952 1954 1956 1963 1968 1970 1977 1982 1984 1987 1992 1994 1996 1998 2001 2010 2016 2018 2020 (19)
indices_E = np.array([1952, 1954, 1956, 1963, 1968, 1970, 1977, 1982, 1984, 1987, 1992, 1994, 1996, 1998, 2001, 2010, 2016, 2018, 2020]) - 1950

n = 10 # pressure level: 10 = 100 hPa
# Zonal wind
ncu = ncdf(example_data_path('/home/data/obs/ERA5/Ulevs_1000-001_mon.era5.nc'), 'r')
uwnd_tmp = ncu.variables['u'][:,n,:,:]
lons = ncu.variables['lon'][:]
lats = ncu.variables['lat'][:]
lev = ncu.variables['level'][n]
ncu.close()
# Meridional wind
ncv = ncdf(example_data_path('/home/data/obs/ERA5/Vlevs_1000-001_mon.era5.nc'), 'r')
vwnd_tmp = ncv.variables['v'][:,n,:,:]
ncv.close()

# Dimensions
nlats = lats.shape[0]
nlons = lons.shape[0]
nyr = uwnd_tmp.shape[0]//12

# Wind decomposition into irrotational and non-divergent components with windspharm
uwnd, uwnd_info = prep_data(uwnd_tmp, 'tyx')
vwnd, vwnd_info = prep_data(vwnd_tmp, 'tyx')
lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)

w = VectorWind(uwnd, vwnd)

sf, vp = w.sfvp()
vp = recover_data(vp, uwnd_info)

uchi, vchi, upsi, vpsi = w.helmholtz()
uchi = recover_data(uchi, uwnd_info)
vchi = recover_data(vchi, uwnd_info)

# Seasonal value
nmonth = [5,8]
VP = seasonal_value(nlats,nlons,nyr,vp,nmonth)
U = seasonal_value(nlats,nlons,nyr,uchi,nmonth)
V = seasonal_value(nlats,nlons,nyr,vchi,nmonth)

# QBO composite
VP_W = np.mean(VP[indices_W,:,:], axis = 0)
U_W = np.mean(U[indices_W,:,:], axis = 0)
V_W = np.mean(V[indices_W,:,:], axis = 0)

VP_E = np.mean(VP[indices_E,:,:], axis = 0)
U_E = np.mean(U[indices_E,:,:], axis = 0)
V_E = np.mean(V[indices_E,:,:], axis = 0)

# Differences
VP_diff = VP_W - VP_E
U_diff = U_W - U_E
V_diff = V_W - V_E

# Statistical test
## VP
pval = 0.05
test_VP = t_test(nlats,nlons,VP,indices_W,indices_E,pval,True)

## U and V
test_U = t_test(nlats,nlons,U,indices_W,indices_E,pval,True)
test_V = t_test(nlats,nlons,V,indices_W,indices_E,pval,True)

# Significant differences
## VP
VP_diff_test = np.copy(VP_diff)
for i in range (nlats):
    for j in range (nlons):
        if (test_VP[i,j]==0): 
            VP_diff_test[i,j] = np.nan

## U and V
U_diff_test = np.copy(U_diff)
V_diff_test = np.copy(V_diff)
for i in range (nlats):
    for j in range (nlons):
        if (test_U[i,j]==0 and test_V[i,j]==0): 
            U_diff_test[i,j] = np.nan
            V_diff_test[i,j] = np.nan

# add a cyclic point for plotting purposes).
from cartopy.util import add_cyclic_point
U_plot, lons_c = add_cyclic_point(U_diff_test, lons)
V_plot, lons_c = add_cyclic_point(V_diff_test, lons)

# Plot
lon2d, lat2d=np.meshgrid(lons,lats)

fig = plt.figure(figsize=(28,10)) #set figure size
ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines() 

ax.set_xticks([0,30,60,90,120,150,180,210,240,270,300,330], crs=ccrs.PlateCarree())
ax.set_yticks([-60,-45,-30,-15,0,15,30,45,60,75], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.tick_params(axis='both', which='major', length=1, width=1, direction='out', size=12, labelsize=28)
ax.set_extent([0, -180, -60, 60], crs=ccrs.PlateCarree())

# 100 hPa    
clevsA=np.arange(-2.5,0,0.5)
clevsB=np.arange(0.5,3.,0.5)
# 700 hPa
#clevsA=np.arange(-1.0,0,0.2)
#clevsB=np.arange(0.2,1.2,0.2)
clevsAB = np.concatenate((clevsA,clevsB))

cf1 = ax.contourf(lon2d, lat2d, VP_diff_test*1e-6, clevsAB, extend = 'both', transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)

clevs2 = np.arange(-10,10.5,0.5)
ax.contour(lon2d, lat2d, VP_diff*1e-6, clevs2, transform=ccrs.PlateCarree(), colors='black')

n=2
Q2 =ax.quiver(lons_c[::n],lats[::n+1],U_plot[::n+1,::n],V_plot[::n+1,::n],transform=ccrs.PlateCarree(),units='width',scale=20,headwidth=3.0,headlength=5.0,headaxislength=4.5,pivot='mid',color='black',alpha=1)
qk2 = ax.quiverkey(Q2,0.8,0.9,1,r'',labelpos='E',coordinates='figure', fontproperties={'size':22})

# 100 hPa
cb = fig.colorbar(cf1, orientation='horizontal', shrink=0.5, ticks = np.concatenate(([-2.5,-1.5,-0.5],[0.5,1.5,2.5])), pad=0.11)
# 700 hPa
#cb = fig.colorbar(cf1, orientation='horizontal', shrink=0.5, ticks = np.concatenate(([-1,-0.6,-0.2],[0.2,0.6,1])), pad=0.11)
cb.set_label('$10^{-6} m²/s$', fontsize=24)
cb.ax.tick_params(labelsize=24)

#Display the plot
plt.show()
