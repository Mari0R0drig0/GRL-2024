import numpy as np
import math
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ncdf
from netCDF4 import num2date, date2num 
from datetime import date, timedelta, datetime
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import MultipleLocator, MaxNLocator
from sklearn.linear_model import LinearRegression

def lat_average(data, lat):
    """Latitudinal average.
    data: field where we apply the latitudinal average
    lat:latitudes where we apply the latitudinal average"""
    nlat_dom = np.size(data,0)
    nlon_dom = np.size(data,1)
    data_ave = np.empty([nlon_dom])
    for i in range (nlon_dom):
        wgts_dom= np.cos(np.deg2rad(lat))
        weight = wgts_dom/np.nansum(wgts_dom)
        data_tmp = np.zeros(nlat_dom)
        for j in range (nlat_dom):
            data_tmp[j] = data[j,i]*weight[j]
        data_ave[i] = np.nansum(data_tmp, axis=0)
    return data_ave

def seasonal_anomalies(nlat, nlon, nyr, field, nmonth):
    """Seasonal detrended anomalies.
    nlat: number of latitudes
    nlon: number of longitudes
    nyr: number of years
    field: field where we compute the seasonal anomalies
    nmonth: months that define the season"""

    value = np.zeros([nyr, nlat, nlon])
    anom = np.zeros([nyr, nlat, nlon])
    anomdetrend = np.zeros([nyr, nlat, nlon])
    for i in range (nyr):
        tmp=field[12*(i)+nmonth[0]:12*(i)+nmonth[-1],:,:]
        value[i,:,:]=np.mean(tmp,axis=0)

    clim = np.mean(value[:30,:,:],axis=0)

    for i in range (0, nyr, 1):
        anom[i,:,:]=value[i,:,:]-clim   

    for k in range (nlat):
        for p in range (nlon):
            anomalies = anom[:,k,p]
            x = np.arange(nyr) 
            model = LinearRegression()
            model.fit(x[:, np.newaxis], anomalies[:,np.newaxis])
            linearreg = model.predict(x[:, np.newaxis])[:,0]
            anomdetrend[:,k,p] = anomalies - linearreg

    return anomdetrend

# El Niño events with anomalies exceeding 1 K, based on NOAA's ONI for DJF
indices_Nino = np.array([1957, 1963, 1965, 1968, 1972, 1982, 1986, 1991, 1994, 1997, 2002, 2009, 2015]) - 1950

# Read SST data
folder = Path('/home//data/obs/HadISST')
file_to_open = folder / 'HadISST.nc'
              
file = ncdf(file_to_open, "r")
timevar = file.variables["time"]
dates = num2date(timevar[:], units=timevar.units)
lat = file.variables['lat'][:]
lon = file.variables['lon'][:]
              
# Select the period
ini=int(np.where(dates==datetime(1950, 1, 16, 12, 0, 0))[0])
fin=int(np.where(dates==datetime(2023, 1, 16, 12, 0, 0))[0])
              
SST = file.variables['sst'][ini:fin,:,:]
file.close()

# Seasonal detrended anomalies
nyr_obs = SST.shape[0]//12
nlat = lat.shape[0]
nlon = lon.shape[0]
nmonth = [11,14]
SSTa = seasonal_anomalies(nlat, nlon, nyr_obs, SST, nmonth)

# Equatorial latitudes
lat_step = lat[1]-lat[0]
lat1 = -5
lat2 = 5
pos1_lat = int(np.where(np.logical_and(lat>lat1, lat<(lat1+lat_step)))[0]+1)
pos2_lat = int(np.where(np.logical_and(lat>lat2, lat<(lat2+lat_step)))[0])
print(lat[pos1_lat:pos2_lat+1])

# Latitudinal average
SST_OBS = np.empty([nyr_obs,nlon])
for i in range (nyr_obs):
    SST_OBS[i,:] = lat_average(SSTa[i,pos1_lat:pos2_lat+1,:],lat[pos1_lat:pos2_lat+1])

# Figure
figure, ax = plt.subplots(1,1, sharex=True, sharey=False,figsize=(16,10))

SST_Nino_OBS = SST_OBS[indices_Nino,:]

plt.plot(lon, SST_Nino_OBS[0,:], color='red', linewidth=0.75, label = 'EN')
for i in range (SST_Nino_OBS.shape[0]):
    plt.plot(lon, SST_Nino_OBS[i,:], color='red', linewidth=0.75)

plt.plot(lon, SST_OBS[1982-1950,:], color='firebrick', linewidth = 3, label = '1982/83')
plt.plot(lon, SST_OBS[1997-1950,:], color='orange', linewidth = 3, label = '1997/98')
plt.plot(lon, SST_OBS[2015-1950,:], color='magenta', linewidth = 3, label = '2015/16')

plt.xlim(150, 278)
plt.ylim(-1,4.1)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
plt.hlines(0,0,360, colors = 'black')

ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(axis='y', which='major', length=16, width=4, direction='out', labelsize=30)
ax.tick_params(axis='y', which='minor', length=8, width=1, direction='out', labelsize=30)

ax.tick_params(axis='x', which='major', length=16, width=4, direction='out', labelsize=30)
ax.tick_params(axis='x', which='minor', length=8, width=1, direction='out', labelsize=30)
ax.set_xticklabels(labels=['140E','160E','180','160W','140W','120W','100W'], fontsize=30)

legend=plt.legend(loc="upper left", fontsize=28, ncol=1)
legend.get_frame().set_facecolor('white')

plt.grid(True, which='major', alpha=0.5)

plt.vlines(x=190, ymin=-1.0, ymax=-0.6, linewidth=4, color='black')
plt.vlines(x=240, ymin=-1.0, ymax=-0.6, linewidth=4, color='black')
plt.text(208,-0.8,"Nino3.4", fontsize=32, color='black')

plt.ylabel('K', fontsize=32)

plt.text(205,-1.85,'Longitude',fontsize=28)

plt.subplots_adjust(top=0.95,bottom=0.14,left=0.05,right=0.95,hspace=0.15,wspace=0.2)

plotname = 'Spaghetti_plot_DJF_SST_OBS_ENDJF' # Figure name
plotsDir = '/home/Plots/' # directory to save the Figure

#plt.savefig(plotsDir + plotname+ '.eps', bbox_inches='tight')
plt.show()
