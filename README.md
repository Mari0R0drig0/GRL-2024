# GRL-2024
This repository contains Python scripts for replicating the main figures in the paper "Quasi-Biennial Oscillation Influence on Tropical Convection and El Niño Variability," submitted to Geophysical Research Letters in October 2024. It includes three scripts:
  1. Spaghetti_plot_SST.py: Computes observed equatorial SST anomalies during El Niño events in DJF using the HadISST dataset. El Niño events are defined by a 1 K anomaly in the ONI DJF index, with super El Niño events highlighted.

  2. QBO_eq_impact_U_&_T.py: Computes the QBO signal in equatorial temperature (shading) and zonal wind (contours) using a Westerly (W) minus Easterly (E) composite in ERA5. This is done for all years and for El Niño events specifically. Contours are plotted at 0.5 m/s intervals, with values above 4 m/s plotted every 2 m/s. Only statistically significant differences in temperature (95% confidence level) are shaded, and zonal wind differences are bolded.

  3. QBO_div_wind.py: Computes the QBO signal in velocity potential and the divergent wind component at 100 hPa, using a Westerly (W) minus Easterly (E) composite in ERA5 for all years and for El Niño events specifically. Statistically significant differences (95% confidence level) in velocity potential and divergent wind are shaded and plotted, respectively.

The calculation of velocity potential and the irrotational (divergent) wind component in QBO_div_wind.py is based on the windspharm library: https://ajdawson.github.io/windspharm/latest/.

The outputs from these scripts correspond to the main figures presented in the paper.

All scripts were run using Python 3.10.12.
