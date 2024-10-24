import numpy as np
import math
from netCDF4 import Dataset as ncdf
from netCDF4 import num2date, date2num 
from datetime import date, timedelta, datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path

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

def seasonal_anomalies_OBS(nlat, nlon, nyr, field, nmonth):
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
    tmp=field[12*(i)+nmonth[0]:12*(i)+nmonth[-1],:,:] #select season (previous or next)
    value[i,:,:]=np.mean(tmp,axis=0) #season field mean
  
  clim = np.mean(value,axis=0) #season field climatology
  
  for i in range (0, nyr, 1):
    anom[i,:,:]=value[i,:,:]-clim   
  
  for k in range (nlat):
    for p in range (nlon):
      anomalies = anom[:,k,p]
      x = np.arange(nyr) 
      model = LinearRegression()
      model.fit(x[:, np.newaxis], anomalies[:,np.newaxis])
      linearreg = model.predict(x[:, np.newaxis])[:,0]
      anomdetrend[:,k,p] = anomalies - linearreg # quito la tendencia para cada punto
  
  return anomdetrend

# QBO composite years
folder = Path("/home/mrodrigo/Documents/data/re-obs/ERA5/")
file_to_open = folder / "Ulevs_1000-001_mon.era5_195001_202105_interp.nc"

file = ncdf(file_to_open, "r")

timevar = file.variables["time"]
dates = num2date(timevar[:], units=timevar.units) 
lons = file.variables["lon"][:]
lats = file.variables["lat"][:]
lev = file.variables["level"][:]
n = 8 # 50 hPa vertical level
print('QBO level: '+str(lev[n]))

lat_step = lats[1]-lats[0]
pos1_lat=int(np.where(np.logical_and(lats<=-0.5, lats>=-0.5-lat_step))[0]) 
pos2_lat=int(np.where(np.logical_and(lats>=0.5, lats<=0.5+lat_step))[0])

# Select period
ini=int(np.where(dates==datetime(1950, 1, 1, 0, 0, 0, 0))[0])
fin=int(np.where(dates==datetime(2021, 1, 1, 0, 0, 0, 0))[0])

U = file.variables['u'][ini:fin,n,pos1_lat:pos2_lat+1,:]

file.close()

nlats = U.shape[1]
nlons = U.shape[2]
nmonth = [5,8]
nyr = U.shape[0]//12
U_anom = seasonal_anomalies_OBS(nlats, nlons, nyr, U, nmonth)

index = np.mean(U_anom, axis=(1,2))
index = index/np.std(index) # Standardized index

# QBO W and E years
indices_W = np.where(index >= 0.75)[0]
print('Indices Westerly: '+str(1950+indices_W), indices_W.shape[0])
indices_E = np.where(index <= -0.75)[0]
print('Indices Easterly: '+str(1950+indices_E), indices_E.shape[0])

# Zonal wind field
folder = Path("/home/mrodrigo/Documents/data/re-obs/ERA5/")
file_to_open = folder / "Ulevs_1000-001_mon.era5_195001_202105_interp.nc"

file = ncdf(file_to_open, "r")

timevar = file.variables["time"]
dates = num2date(timevar[:], units=timevar.units) 
lons = file.variables["lon"][:]
lats = file.variables["lat"][:]
lev = np.flip(file.variables["level"][:])

lat_step = lats[1]-lats[0]
pos1_lat=int(np.where(np.logical_and(lats<=-5.5, lats>=-5.5-lat_step))[0]) 
pos2_lat=int(np.where(np.logical_and(lats>=5.5, lats<=5.5+lat_step))[0])

uwnd = np.flip(file.variables['u'][ini:fin,:,pos1_lat+1:pos2_lat,:], axis=1)

file.close()

# Dimensions
nlon = uwnd.shape[3]
nlat = uwnd.shape[2]
nlev = uwnd.shape[1]
nyr = uwnd.shape[0]//12

# Seasonal values
Ua = np.empty([nyr,nlev,nlat,nlon])
nmonth = [5,8]
for i in range (nlev):
    Ua[:,i,:,:] = seasonal_value(nlat,nlon,nyr,uwnd[:,i,:,:],nmonth)

# Latitudinal mean
Ua_lat_mean = np.mean(Ua, axis=2)

# QBO composite
coeff_U_W = np.mean(Ua_lat_mean[indices_W,:,:], axis=0)
coeff_U_E = np.mean(Ua_lat_mean[indices_E,:,:], axis=0)

# Differences
coeff_U_diff = coeff_U_W - coeff_U_E

# Temperature field
folder = Path("/home/mrodrigo/Documents/data/re-obs/ERA5/")
file_to_open = folder / "Tlevs_1000-001_mon.era5_195001_202105_interp.nc"

file = ncdf(file_to_open, "r")

T = np.flip(file.variables['t'][ini:fin,:,pos1_lat+1:pos2_lat,:], axis=1)

file.close()

# Seasonal values
Ta = np.empty([nyr,nlev,nlat,nlon])
for i in range (nlev):
    Ta[:,i,:,:] = seasonal_value(nlat,nlon,nyr,T[:,i,:,:],nmonth)

# Latitudinal mean
Ta_lat_mean = np.mean(Ta, axis=2)

# Composite
coeff_T_W = np.mean(Ta_lat_mean[indices_W,:,:], axis=0)
coeff_T_E = np.mean(Ta_lat_mean[indices_E,:,:], axis=0)

# Differences
coeff_T_diff = coeff_T_W - coeff_T_E

# Student t-test
test_T = np.empty([nlev,nlon])
for i in range (nlev):
    for j in range (nlon):
        pvalue = stats.ttest_ind(Ta_lat_mean[indices_W,i,j],Ta_lat_mean[indices_E,i,j],axis=0,equal_var=True,nan_policy='propagate').pvalue
        if (pvalue<=0.05): test_T[i,j] = 1
        else: test_T[i,j] = 0
            
test_U = np.empty([nlev,nlon])
for i in range (nlev):
    for j in range (nlon):
        pvalue = stats.ttest_ind(Ua_lat_mean[indices_W,i,j],Ua_lat_mean[indices_E,i,j],axis=0,equal_var=True,nan_policy='propagate').pvalue
        if (pvalue<=0.05): test_U[i,j] = 1
        else: test_U[i,j] = 0

# Significant differences
coeff_T_diff_test = np.copy(coeff_T_diff)
for i in range (nlev):
    for j in range (nlon):
        if (test_T[i,j]==0): 
            coeff_T_diff_test[i,j] = np.nan
            
coeff_U_diff_test = np.copy(coeff_U_diff)
for i in range (nlev):
    for j in range (nlon):
        if (test_U[i,j]==0): 
            coeff_U_diff_test[i,j] = np.nan

# Plot
fig = plt.figure(figsize=(24,16)) #set figure size
ax=plt.axes()

# Zonal wind
clevsAB = np.array([-39,-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,-2,-1,1,1.5,2,2.5,3,6,9,12,15,18,21,24,27,30,33,36,39])

# Temperature
clevsA = np.arange(-2.5,0,0.5)
clevsB = np.arange(0.5,3.0,0.5)
clevsAB2 = np.concatenate((clevsA,clevsB))

Q2 = plt.contourf(lons, lev, coeff_T_diff_test, clevsAB2, extend='both', cmap=plt.cm.RdBu_r)

plt.contour(lons, lev, coeff_U_diff, clevsAB, colors='black', negative_linestyles = 'dashed')
plt.contour(lons, lev, coeff_U_diff_test, clevsAB, colors='black', linewidths=5)

cb=plt.colorbar(Q2, orientation='horizontal', ticks=[-2.5,-1.5,-0.5,0.5,1.5,2.5], shrink=0.65, pad=0.05)
cb.set_label('K', fontsize=24)
cb.ax.tick_params(labelsize=24)

plt.yscale('log')
plt.yticks(lev, fontsize=28)
ax.set_yticklabels(['1000','','','','','','','','','','','700','','','','500','','','','300','','','200','','','','100','70','50','30','20','10','7','5','3','2','1'])

plt.xticks(np.arange(0,390,30), fontsize=28)
ax.set_xticklabels(['0','30 E','60 E','90 E','120 E','150 E','180','150 W','120 W','90 W','60 W','30 W',''])

plt.xlim(0,357.5)
plt.ylim(1000,5)

#plt.savefig("/home/mrodrigo/Documents/DynVar_SNAP_23/Letter/Plots/Main/Eq_U_&_T_JJA_QBO_0.75std_50hPa_JJA_ERA5"+'.eps', bbox_inches='tight')

plt.show()
