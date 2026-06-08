
'''
Created on June 2026
@author Rosie Eade

- test pdf_distance.py: distance based scores on random normal datasets
  test sensitivity to number of bins (nbins)

'''

import os
from pathlib import Path

import scipy as sp
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

import pdf_distance

import importlib

# Can reimport libraries using importlib.reload() e.g. importlib.reload(pdf_distance) 

out_dir=Path('test_pdf_dist/')
os.makedirs(out_dir, exist_ok=True)

#nbins_arr=np.array([10, 100, 1000, 10000, 100000])

nbins_power=np.arange(26)*0.2+1
nbins_arr=np.zeros(len(nbins_power))
for ii in range(len(nbins_power)): nbins_arr[ii]=int(10**nbins_power[ii])
nbins_arr=nbins_arr.astype(int)

# -------------------------------

sample_size=10**7 # 10000000
arr0=np.random.normal(loc=0.0, scale=1.0, size=sample_size)
arr1=np.random.normal(loc=0.0, scale=0.5, size=sample_size)
arr2=np.random.normal(loc=1.0, scale=1.0, size=sample_size)
arr3=np.random.normal(loc=2.0, scale=1.0, size=sample_size)

print(nbins_arr)
print(nbins_arr.max())
print(sample_size)

printBins=False # True

# -------------------------------
# Wasserstein distance: approximate (on histogram data) vs full method (on orig sample data)

wd01=sp.stats.wasserstein_distance(arr0,arr1)
wd02=sp.stats.wasserstein_distance(arr0,arr2)
wd03=sp.stats.wasserstein_distance(arr0,arr3)

approx_wd01=np.zeros(len(nbins_arr))
approx_wd02=np.zeros(len(nbins_arr))
approx_wd03=np.zeros(len(nbins_arr))

for ii in range(len(nbins_arr)): approx_wd01[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr1, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='wdh', printBins=printBins)
for ii in range(len(nbins_arr)): approx_wd02[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr2, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='wdh', printBins=printBins)
for ii in range(len(nbins_arr)): approx_wd03[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr3, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='wdh', printBins=printBins)

filepath=os.path.join(out_dir, 'test_wdh0123.png')

fig, axes = plt.subplots(3, 1, figsize=(6, 9), layout='constrained', squeeze=False)
ax1=axes[0,0]
ax1.plot(nbins_arr, nbins_arr*0.0+wd01, color="black", linestyle='--', label='WD full')
ax1.plot(nbins_arr, approx_wd01, color='red', label='WD approx')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Wasserstein distance')
ax1.set_title('Norm(0,1) vs N(0,0.5)') 
ax1.legend()

ax1=axes[1,0]
ax1.plot(nbins_arr, nbins_arr*0.0+wd02, color="black", linestyle='--', label='WD full')
ax1.plot(nbins_arr, approx_wd02, color='red', label='WD approx')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Wasserstein distance')
ax1.set_title('Norm(0,1) vs N(1,1)') 
ax1.legend()

ax1=axes[2,0]
ax1.plot(nbins_arr, nbins_arr*0.0+wd03, color="black", linestyle='--', label='WD full')
ax1.plot(nbins_arr, approx_wd03, color='red', label='WD approx')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Wasserstein distance')
ax1.set_title('Norm(0,1) vs N(2,1)') 
ax1.legend()

plt.savefig(filepath)
plt.close('all')

# -------------------------------
# Jensen Shannon distance: approximate (on histogram data)

approx_js01=np.zeros(len(nbins_arr))
approx_js02=np.zeros(len(nbins_arr))
approx_js03=np.zeros(len(nbins_arr))

for ii in range(len(nbins_arr)): approx_js01[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr1, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='jsd', printBins=printBins)
for ii in range(len(nbins_arr)): approx_js02[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr2, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='jsd', printBins=printBins)
for ii in range(len(nbins_arr)): approx_js03[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr3, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='jsd', printBins=printBins)

filepath=os.path.join(out_dir, 'test_jsd0123.png')

fig, axes = plt.subplots(3, 1, figsize=(6, 9), layout='constrained', squeeze=False)

ax1=axes[0,0]
ax1.plot(nbins_arr, approx_js01, color='red', label='JSD')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Jensen Shannon distance')
ax1.set_title('Norm(0,1) vs N(0,0.5)') 
ax1.legend()

ax1=axes[1,0]
ax1.plot(nbins_arr, approx_js02, color='red', label='JSD')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Jensen Shannon distance')
ax1.set_title('Norm(0,1) vs N(1,1)') 
ax1.legend()

ax1=axes[2,0]
ax1.plot(nbins_arr, approx_js03, color='red', label='JSD')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Jensen Shannon distance')
ax1.set_title('Norm(0,1) vs N(2,1)') 
ax1.legend()

plt.savefig(filepath)
plt.close('all')

# -------------------------------
# Perkins skill score: approximate (on histogram data)

approx_ps01=np.zeros(len(nbins_arr))
approx_ps02=np.zeros(len(nbins_arr))
approx_ps03=np.zeros(len(nbins_arr))

for ii in range(len(nbins_arr)): approx_ps01[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr1, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='pss', printBins=printBins)
for ii in range(len(nbins_arr)): approx_ps02[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr2, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='pss', printBins=printBins)
for ii in range(len(nbins_arr)): approx_ps03[ii] = pdf_distance.calc_pdf_metric_nparray(arr0, arr3, axis=0, bins_arr=None, nbins=nbins_arr[ii], typedistance='pss', printBins=printBins)

filepath=os.path.join(out_dir, 'test_pss0123.png')

fig, axes = plt.subplots(3, 1, figsize=(6, 9), layout='constrained', squeeze=False)

ax1=axes[0,0]
ax1.plot(nbins_arr, approx_ps01, color='red', label='PSS')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Perkins skill score')
ax1.set_title('Norm(0,1) vs N(0,0.5)') 
ax1.legend()

ax1=axes[1,0]
ax1.plot(nbins_arr, approx_ps02, color='red', label='PSS')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Perkins skill score')
ax1.set_title('Norm(0,1) vs N(1,1)') 
ax1.legend()

ax1=axes[2,0]
ax1.plot(nbins_arr, approx_ps03, color='red', label='PSS')
ax1.set_xscale("log")
ax1.set_xlabel('No. Bins in histogram')
ax1.set_ylabel('Perkins skill score')
ax1.set_title('Norm(0,1) vs N(2,1)') 
ax1.legend()

plt.savefig(filepath)
plt.close('all')
