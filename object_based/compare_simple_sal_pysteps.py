'''
Example comparison of simple_sal to pysteps version:
https://pysteps.readthedocs.io/en/latest/generated/pysteps.verification.salscores.sal.html
* Will need to first install pysteps

- simple_sal generally consistent with pysteps sal
- but treats close objects differently
--- simple_sal retains all objects, regardless of proximity
--- pysteps sal appears to reject objects that are too close to another object

Copyright (c) 2026 Klima consulting
Author: Rosie Eade

'''

import numpy as np

import simple_sal as my_sal

from pysteps.verification.salscores import sal as pysteps_sal
from pysteps.verification.salscores import _sal_detect_objects as pysteps_sal_detect_objects

# Use same neighbour structure as pysteps
structure4=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# Set up example fields 

# Target
imageT=np.zeros([35, 45])
imageT[1:4,1:20]=1.0
imageT[2,10]=5.0
imageT[11:14,1:20]=1.0
imageT[12,10]=6.0
imageT[23:33,33:43]=1.0
imageT[27,37]=7.0


print('----------------------------------')
print('Example prediction 1: slightly shifted version of Target')
# Prediction
imageP1=np.zeros([35, 45])
imageP1[2:5,2:19]=2.0
imageP1[2,10]=5.0
imageP1[12:15,2:19]=2.0
imageP1[12,10]=6.0
imageP1[20:30,30:38]=2.0
imageP1[25,35]=7.0

mysalP1, myobjT1, myobjP1=my_sal.compute_sal(imageP1, imageT, eThreshFix=None, eThreshPrFix=None, thr_factor=1/15., thr_quantile=0.95, minFac='min', minsize=50, structure=structure4,printThresh=True)
pysalP1=pysteps_sal(imageP1,imageT)
pyobjP1 = pysteps_sal_detect_objects(imageP1, thr_factor=1/15., thr_quantile=0.95, tstorm_kwargs=None)
print('P1 my_sal SAL scores')
print(mysalP1)
print('P1 pysteps SAL scores')
print(pysalP1)
print('Example prediction 1: Identical SAL score values')
print('------------')
print('P1 my_sal SAL objects')
print(myobjP1)
print('P1 pysteps SAL objects')
print(pyobjP1)
print('Example prediction 1: Identical objects')
print('----------------------------------')

print('----------------------------------')
print('Example prediction 2: slightly shifted version of Target but with max locations very close')
# Prediction
imageP1=np.zeros([35, 45])
imageP1[1:4,1:20]=1.0
imageP1[2,10]=5.0
imageP1[10:13,1:20]=1.0
imageP1[11,10]=6.0
imageP1[20:30,30:40]=1.0
imageP1[25,35]=7.0

mysalP1, myobjT1, myobjP1=my_sal.compute_sal(imageP1, imageT, eThreshFix=None, eThreshPrFix=None, thr_factor=1/15., thr_quantile=0.95, minFac='min', minsize=50, structure=structure4,printThresh=True)
pysalP1=pysteps_sal(imageP1,imageT)
pyobjP1 = pysteps_sal_detect_objects(imageP1, thr_factor=1/15., thr_quantile=0.95, tstorm_kwargs=None)
print('P1 my_sal SAL scores')
print(mysalP1)
print('P1 pysteps SAL scores')
print(pysalP1)
print('Example prediction 2: Different SAL score values')
print('------------')
print('P1 my_sal SAL objects')
print(myobjP1)
print('P1 pysteps SAL objects')
print(pyobjP1)
print('Example prediction 2: Different objects:')
print(' pysteps ignores 2nd object as max value too close to 1st object')
print(' i.e. < 10 grid points, contrary to expectation that pysteps merges objects')
print('----------------------------------')

print('----------------------------------')
print('Example prediction 3: as prediction 2 but 2nd object shifted by 1 extra grid box')
# Prediction
imageP1=np.zeros([35, 45])
imageP1[1:4,1:20]=1.0
imageP1[2,10]=5.0
imageP1[11:14,1:20]=1.0
imageP1[12,10]=6.0
imageP1[20:30,30:40]=1.0
imageP1[25,35]=7.0

mysalP1, myobjT1, myobjP1=my_sal.compute_sal(imageP1, imageT, eThreshFix=None, eThreshPrFix=None, thr_factor=1/15., thr_quantile=0.95, minFac='min', minsize=50, structure=structure4,printThresh=True)
pysalP1=pysteps_sal(imageP1,imageT)
pyobjP1 = pysteps_sal_detect_objects(imageP1, thr_factor=1/15., thr_quantile=0.95, tstorm_kwargs=None)
print('P1 my_sal SAL scores')
print(mysalP1)
print('P1 pysteps SAL scores')
print(pysalP1)
print('Example prediction 3: Identical SAL score values')
print('------------')
print('P1 my_sal SAL objects')
print(myobjP1)
print('P1 pysteps SAL objects')
print(pyobjP1)
print('Example prediction 3: Identical objects')
print('----------------------------------')

