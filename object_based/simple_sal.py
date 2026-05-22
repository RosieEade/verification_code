"""
Code to compute SAL score for pairs of lat/lon gridded fields:
Prediction and Target (truth)

    compute_sal_xr()
    - compute SAL score for multiple timesteps of fields
      in an xarray Dataset

    compute_sal()
    - compute SAL score for single pair of fields
      in a numpy array
    - Option to also plot objects for single fields

Plotting code also provided for standard summary plots
of SAL scores computed over multiple timesteps

Author: Rosie Eade
 
"""

from pathlib import Path

import os
import numpy as np
import scipy as sp
import xarray as xr
import math

import cartopy.crs as ccrs
import cartopy.feature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# Computation Code
# -------------------------------------------------------------------------------------
def calculate_error_nparray(arr0, arr1, axis=0, typeerror='rmse'):
    """
    Calculate (rms) error between 2 N-dimensional numpy arrays
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : int, default=0
        Axis/axes along which to compute error
    typeerror : str, default='rmse'
        Type of error to compute, options ('rmse', 'bias', 'abias', 'ratio')
    
    Returns:
    --------
    numpy.ndarray
        Error. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).
    
    """

    if arr0.shape != arr1.shape:
        raise ValueError(f"Shape mismatch: {arr0.shape} vs {arr1.shape}")

    error_list=['bias', 'abias', 'rmse', 'ratio']
    if not typeerror in error_list:
        raise ValueError(f"typeerror not recognised: {typeerror}")
    
    # Compute Error over axis/axes specified
    # - subtract first then take mean so consistent method for bias and rmse
    # - better level of accuracy than taking means then subtract (slight difference)
    if typeerror=='bias': output_error = (arr1-arr0).mean(axis=axis)
    if typeerror=='abias': output_error = np.abs(arr1-arr0).mean(axis=axis)
    if typeerror=='rmse': output_error = np.sqrt(((arr1-arr0)**2).mean(axis=axis))
    if typeerror=='ratio': output_error = arr1.mean(axis=axis)/arr0.mean(axis=axis)
    
    return output_error

# -------------------------------------------------------------------------------------
def calculate_pearsoncorr_nparray(arr0, arr1, axis=0):
    """
    Calculate Pearson correlation between 2 N-dimensional numpy arrays
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : int or type of int, default=0
        Axis or tuple of axes over which to compute correlation
        None : if want to compute over all axes
    
    Returns:
    --------
    numpy.ndarray
        Pearson correlation coefficients. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).
    
    """

    if arr0.shape != arr1.shape:
        raise ValueError(f"Shape mismatch: {arr0.shape} vs {arr1.shape}")
    
    # Center the data over axis/axes specified
    arr0_centered = arr0 - arr0.mean(axis=axis, keepdims=True)
    arr1_centered = arr1 - arr1.mean(axis=axis, keepdims=True)
    
    # Compute correlation over axis/axes specified
    numerator = (arr0_centered * arr1_centered).sum(axis=axis)
    denominator = (arr0_centered**2).sum(axis=axis) * (arr1_centered**2).sum(axis=axis)
    denominator = np.sqrt(denominator)

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator!=0)
    
    return correlations

# -------------------------------------------------------------------------------------
def calc_3dradius_sal(sal_vals, percentile: float | None = None):
    """
    Compute the 3d radii based on the 3 elements of the SAL score.
    Based on Wernli et al., 2008
    https://journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml

    Output the given percentile for the 3d radii 
    Compute the 3d radii for all timesteps given, the compute the
    percentile (if not None)
    e.g. percentile=5: computes the radius that contains best 5% 
    of timesteps (smaller radius suggests better prediction)
    Added by R. Eade.

    Parameters:
    -----------
    sal_vals : xr.Dataset
        Dataset output from compute_sal_xr with s, a and l values vs time
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location
    percentile : float | None
        Given a float, compute that percentile from set of radii
        Given None, output all radii 

    Returns:
    --------
    numpy.ndarray
        Percentile of radii values
        or all values (if percentile==None)
    
    """
    if not isinstance(percentile, (int, float)): percentile=None
    
    if percentile:
        if not (0.0 <= percentile <= 100.0):
            print(f"percentile must be between 0 and 100, but given")
            print(f"{percentile} thus output all radii instead")
            percentile=None

    ntime=sal_vals.sal_s.shape[0]
    rad_all=np.zeros(ntime)
    for dd in range(ntime):
        rad_all[dd]=np.square(sal_vals.sal_s[dd].values)
        rad_all[dd]=rad_all[dd]+np.square(sal_vals.sal_a[dd].values)
        rad_all[dd]=rad_all[dd]+np.square(sal_vals.sal_l[dd].values)
        rad_all[dd]=np.sqrt(rad_all[dd])
    
    if percentile==None: return rad_all 
    else: return np.nanpercentile(rad_all,percentile)

# -------------------------------------------------------------------------------------
def compute_sal_xr(
    prediction_xr, 
    target_xr, 
    eThreshFix=None, 
    eThreshPrFix=None, 
    thr_quantile=None, 
    thr_factor=None, 
    minFac=None, 
    minsize=0, 
    structure=np.ones((3, 3), dtype=int)):

    '''
    Compute SAL score wrt given threhold(s)
    Based on Wernli et al., 2008 and 2009
    https://journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml
    https://journals.ametsoc.org/view/journals/wefo/24/6/2009waf2222271_1.xml
    Extended to allow the use of:
    - fixed thresholds independent of input data
    - different minimum object size thresholds
    - different structure for neighbour definitions    
    Added by R. Eade.

    Parameters:
    -----------
    prediction : numpy.ndarray
        Prediction field data as 3d [time, lat, lon] array
    target : xr.Dataset
        Target field data, same shape as prediction
    eThreshFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If eThreshFix value given, this overrides quantile based options
    eThreshPrFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If None, uses eThreshFix.
    thr_quantile : float | None
        Quantile value in [0.0, 1.0] used to compute threshold to define
        event, as Wernli et al. 2009 (they use thr_quantile=0.95)
    thr_factor : float | None
        Factor to reduce the quantile by, as Wernli et al. 2008 & 2009
        (they use thr_factor=1/15)
    minFac : float | str | None
        Option to mask data less than threshold=minFac before computing
        quantile, as Wernli et al. 2009 (they use 0.1 mm for precip)
        Special case: minFac='min' implies use min value of field,
        chosen to align with option in pysteps.
    minsize : int = 0
        Option to ignore options with size (no. grid points) < minsize
    structure : numpy.ndarray, dtype=int, shape [3, 3]
        This array defines what are classed as neighbouring grid points.
        2 Options:
        np.ones((3, 3), dtype=int) # Orthogonal and diagonal (default)
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) # Orthogonal only
    
    Returns:
    --------
    xr.Dataset
        Arrays of SAL scores for input target and prediction field pairs
        (np.nan if no objects found)
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location
        .sal_l1 = location_1
        .sal_l2 = location_2
        .sal_targ_num = number of objects found in target field
        .sal_pred_num = number of objects found in prediction field

    See Also
    --------

    scipy.ndimage.label : 
    Identify objects using sp.ndimage.label
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    
    Method checked against pysteps
    https://pysteps.readthedocs.io/en/stable/
    Using default pysteps parameters, compute_sal() has identical results
    except:
    - pysteps rejects objects that are within 10 pixels of another object
      (distance between location of peaks),
      i.e. doesn't appear to merge such objects as expected?
    - compute_sal keeps objects as separate, regardless of proximity

    '''
  
    if target_xr.values.shape != prediction_xr.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target_xr.values.shape},"
            f"Prediction {prediction_xr.values.shape})"
        )
        
    ntime=target_xr.values.shape[0]
    
    sal_s=[]
    sal_a=[]
    sal_l=[]
    sal_l1=[]
    sal_l2=[]
    sal_targ_n=[]
    sal_pred_n=[]
    
    for tcount in range(ntime):
        target_np=target_xr.values[tcount]
        prediction_np=prediction_xr.values[tcount]
                        
        sal_tmp, targ_obj, pred_obj = compute_sal(
            prediction_np, 
            target_np, 
            eThreshFix=eThreshFix, 
            eThreshPrFix=eThreshPrFix, 
            thr_factor=thr_factor, 
            thr_quantile=thr_quantile, 
            minFac=minFac, 
            minsize=minsize, 
            structure=structure)
        
        sal_s.append(sal_tmp.sal_s)
        sal_a.append(sal_tmp.sal_a)
        sal_l.append(sal_tmp.sal_l)
        sal_l1.append(sal_tmp.sal_l1)
        sal_l2.append(sal_tmp.sal_l2)
        sal_targ_n.append(sal_tmp.sal_targ_num)
        sal_pred_n.append(sal_tmp.sal_pred_num)

    sal_all = xr.Dataset({
            "sal_s": ("time", np.array(sal_s)),
            "sal_a": ("time", np.array(sal_a)),
            "sal_l": ("time", np.array(sal_l)),
            "sal_l1": ("time", np.array(sal_l1)),
            "sal_l2": ("time", np.array(sal_l2)),
            "sal_targ_num": ("time", np.array(sal_targ_n)),
            "sal_pred_num": ("time", np.array(sal_pred_n))})

    sal_all['time'] = target_xr['time']

    return sal_all

# -------------------------------------------------------------------------------------
def compute_sal(
    prediction, 
    target, 
    lat_values=None, 
    lon_values=None, 
    eThreshFix=None, 
    eThreshPrFix=None, 
    thr_quantile=None, 
    thr_factor=None, 
    minFac=None, 
    minsize=0, 
    structure=np.ones((3, 3), dtype=int), 
    filename=None, 
    plot_indiv=False, 
    field_levels=None, 
    printThresh=False):

    '''
    Compute SAL score wrt given threhold(s)
    Based on Wernli et al., 2008 and 2009
    https://journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml
    https://journals.ametsoc.org/view/journals/wefo/24/6/2009waf2222271_1.xml
    Extended to allow the use of:
    - fixed thresholds independent of input data
    - different minimum object size thresholds
    - different structure for neighbour definitions    
    Added by R. Eade.

    Parameters:
    -----------
    prediction : numpy.ndarray
        Prediction field data as 2d [lat, lon] array
    target : numpy.ndarray
        Target field data, same shape as prediction
    lat_values : numpy.ndarray
        1d array of latitude values matching data arrays
        (only needed if plotting options are used)
    lon_values : numpy.ndarray
        1d array of longitude values matching data arrays
        (only needed if plotting options are used)
    eThreshFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If eThreshFix value given, this overrides quantile based options
    eThreshPrFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If None, uses eThreshFix.
    thr_quantile : float | None
        Quantile value in [0.0, 1.0] used to compute threshold to define
        event, as Wernli et al. 2009 (they use thr_quantile=0.95)
    thr_factor : float | None
        Factor to reduce the quantile by, as Wernli et al. 2008 & 2009
        (they use thr_factor=1/15)
    minFac : float | str | None
        Option to mask data less than threshold=minFac before computing
        quantile, as Wernli et al. 2009 (they use 0.1 mm for precip)
        Special case: minFac='min' implies use min value of field,
        chosen to align with option in pysteps.
    minsize : int = 0
        Option to ignore options with size (no. grid points) < minsize
    structure : numpy.ndarray, dtype=int, shape [3, 3]
        This array defines what are classed as neighbouring grid points.
        2 Options:
        np.ones((3, 3), dtype=int) # Orthogonal and diagonal (default)
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) # Orthogonal only
    filename : str | None
        If not None: objects map saved to filename+'_objects.png':
        Target (Truth) shaded in colours, Prediction as contour lines
        (SAL scores in title)
    plot_indiv : boolean = False
        If True: 2 by 2 figure of maps of input fields and object fields
        saved to filename+'_fields_objects.png' (SAL scores in title)
    field_levels : numpy.ndarray | None
        Optional 1d array of levels to be used by contourf for input field
        plots if plot_indiv is True
    printThresh : boolean
        If True: print thresholds used
    
    Returns:
    --------
    xr.Dataset
        SAL scores for single input target and prediction field pair
        (np.nan if no objects found)
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location
        .sal_l1 = location_1
        .sal_l2 = location_2
        .sal_targ_num = number of objects found in target field
        .sal_pred_num = number of objects found in prediction field
    xr.Dataset | None
        Object information for target (None if no objects found)
        .obj_com_lat : object centre of mass lat. (units: no. grid boxes)
        .obj_com_lon : object centre of mass lon. (units: no. grid boxes)
        .obj_max : object max value
        .obj_size : object size (no grid points)
        .obj_labeled_array : object field, integer labels (shape==target)
        Sub-select single objects:
            pred_object_xarray.sel(object_num=6).obj_max
    xr.Dataset | None
        Object information for prediction (as for target)

    See Also
    --------

    scipy.ndimage.label : 
    Identify objects using sp.ndimage.label
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    
    Method checked against pysteps
    https://pysteps.readthedocs.io/en/stable/
    Using default pysteps parameters, compute_sal() has identical results
    except:
    - pysteps rejects objects that are within 10 pixels of another object
      (distance between location of peaks),
      i.e. doesn't appear to merge such objects as expected?
    - compute_sal keeps objects as separate, regardless of proximity

    Examples
    --------

    Example to sub-select single objects:
    >>> sal, targ, pred = compute_sal(target, prediction)
    >>> pred.sel(object_num=1).obj_max

    '''
    
    if target.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: target ({target.shape}, prediction {prediction.shape})"
        )

    # Code assumes that minsize is an integer so check (and convert)
    if minsize is None: minsize=0
    if isinstance(minsize, float): minsize=int(minsize)

    # If fixed target threshold defined, default to using this
    if eThreshFix is not None:
        # Use fixed thresholds
        eThresh=eThreshFix
        if eThreshPrFix is None: eThreshPr=eThresh
        if eThreshPrFix is not None: eThreshPr=eThreshPrFix

    # If no fixed target threshold defined, use quantile based thresholds
    # Set default values if not supplied
    if eThreshFix is None:
        if thr_quantile is None: thr_quantile=0.95
        if thr_factor is None: thr_factor=1/15.

        # Compute quantile-based thresholds from input data,
        # with option to first mask out very small values
        if isinstance(minFac, (int, float)):
            eThresh = thr_factor * np.nanquantile(
                target[target > minFac], thr_quantile)
            eThreshPr = thr_factor * np.nanquantile(
                prediction[prediction > minFac], thr_quantile)
        if isinstance(minFac, str):
            if minFac == 'min':
                eThresh = thr_factor * np.nanquantile(
                    target[target > np.nanmin(target)], thr_quantile)
                eThreshPr = thr_factor * np.nanquantile(
                    prediction[prediction > np.nanmin(prediction)], thr_quantile)
            else:
                minFac = None
        if minFac is None:
            eThresh = thr_factor * np.nanquantile(target, thr_quantile)
            eThreshPr = thr_factor * np.nanquantile(prediction, thr_quantile)


    if printThresh is True:
        print(f"eThresh={eThresh:.3f}")
        print(f"eThreshPr={eThreshPr:.3f}")

    # Setup empty xarray dataset for the case where no objects are found
    no_object_xr=xr.Dataset(
        {
            "sal_s": np.nan,
            "sal_a": np.nan,
            "sal_l": np.nan,
            "sal_l1": np.nan,
            "sal_l2": np.nan,
            "sal_targ_num": 0,
            "sal_pred_num": 0,
        })
    
    # If all data below event thresholds, then no objects can be found
    targ_max=target.max()
    pred_max=prediction.max()
    if targ_max<=eThresh or pred_max <=eThreshPr:
        print("No Objects Found: Event thresholds too large")
        return no_object_xr, None, None

    targ_masked: np.ndarray = (target > eThresh) & np.isfinite(target)
    targ_labeled_array, targ_num_features = sp.ndimage.label(
        targ_masked, structure=structure)

    pred_masked: np.ndarray = (prediction > eThreshPr) & np.isfinite(prediction)
    pred_labeled_array, pred_num_features = sp.ndimage.label(
        pred_masked, structure=structure)

    # Compute size of objects
    targSize=np.zeros(targ_num_features)
    for icount in range(targ_num_features): 
        targSize[icount]=targ_labeled_array[targ_labeled_array==icount+1].size
    predSize=np.zeros(pred_num_features)
    for icount in range(pred_num_features): 
        predSize[icount]=pred_labeled_array[pred_labeled_array==icount+1].size

    # Option to discard objects if too small and then renumber 
    # - assumes minsize is an integer
    ReCompSize=False
    targSizemin=targSize.min()
    targSizemax=targSize.max()
    predSizemin=predSize.min()
    predSizemax=predSize.max()
    # If all objects below size thresholds, then no objects can be found
    if targSizemax<minsize or predSizemax<minsize:
        print("No Objects Found: Objects too small")
        return no_object_xr, None, None
    

    if minsize>1 and targSizemin<minsize:
        unq_lab_orig=np.unique(targ_labeled_array)
        targ_labeled_tmp=targ_labeled_array.copy()
        for ocount in range(targ_num_features):
            if targSize[ocount]<minsize: 
                targ_labeled_tmp[targ_labeled_tmp==(ocount+1)]=0
        unq_lab_tmp=np.unique(targ_labeled_tmp)
        targ_num_features_tmp=len(unq_lab_tmp)-1
        for ocount in range(targ_num_features_tmp): 
            targ_labeled_tmp[targ_labeled_tmp==unq_lab_tmp[ocount+1]] = ocount+1
        targ_labeled_array=targ_labeled_tmp.copy()
        targ_num_features=targ_num_features_tmp
        ReCompSize=True

    if minsize>1 and predSizemin<minsize:
        unq_lab_orig=np.unique(pred_labeled_array)
        pred_labeled_tmp=pred_labeled_array.copy()
        for ocount in range(pred_num_features):
            if predSize[ocount]<minsize: 
                pred_labeled_tmp[pred_labeled_tmp==(ocount+1)]=0
        unq_lab_tmp=np.unique(pred_labeled_tmp)
        pred_num_features_tmp=len(unq_lab_tmp)-1
        for ocount in range(pred_num_features_tmp): 
            pred_labeled_tmp[pred_labeled_tmp==unq_lab_tmp[ocount+1]] = ocount+1
        pred_labeled_array=pred_labeled_tmp.copy()
        pred_num_features=pred_num_features_tmp
        ReCompSize=True

    # Re-Compute size of objects
    if ReCompSize:
        targSize=np.zeros(targ_num_features)
        for icount in range(targ_num_features): 
            targSize[icount]=targ_labeled_array[targ_labeled_array==icount+1].size
        predSize=np.zeros(pred_num_features)
        for icount in range(pred_num_features): 
            predSize[icount]=pred_labeled_array[pred_labeled_array==icount+1].size

        
    # Compute SAL score (as Wernli et al, 2008)
    
    # ----------------------------
    # - Amplitude
    #   Measure of total over whole domain
    sal_amplitude=2.0*(prediction.mean() - target.mean())
    sal_amplitude=sal_amplitude/(prediction.mean() + target.mean())

    # ----------------------------
    # - Location
    #   Measure of location of objects wrt whole domain
    # - based on centre of mass of fields and objects
    # scipy.ndimage.center_of_mass
    # docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html

    # --- location 1 based on centre of mass of total fields
    # taken from pysteps
    # Normalised difference of target and prediction centres of mass
    # - distances computed in number of grid points
    # - assumes small region so actual distances roughly equal?
    # centre_of_mass() assumes data non-negative and non-missing
    max_d = math.hypot(target.shape[0],target.shape[1]) # i.e. domain diagonal
    targ_shifted=target-np.nanmin(target)
    targCoM = sp.ndimage.center_of_mass(np.nan_to_num(targ_shifted,nan=0))
    pred_shifted=prediction-np.nanmin(prediction)
    predCoM = sp.ndimage.center_of_mass(np.nan_to_num(pred_shifted,nan=0))
    diffCoM = math.hypot(predCoM[1] - targCoM[1], predCoM[0] - targCoM[0])
    Loc1 = np.abs(diffCoM) / max_d

    # --- location 2 based on centre of mass of individual objects
    # - Compute total sum of values per object (sum of all grid points) [Rn]
    targTotal=np.zeros(targ_num_features)
    for icount in range(targ_num_features):
        targTotal[icount]=target[targ_labeled_array==icount+1].sum()
    predTotal=np.zeros(pred_num_features)
    for icount in range(pred_num_features):
        predTotal[icount]=prediction[pred_labeled_array==icount+1].sum()

    # Compute centre of mass of each object and distance from centre of its total field
    targ_distCoM=np.zeros(targ_num_features)
    targ_obj_CoM=[]
    for icount in range(targ_num_features):
        tmp_shifted=target.copy()-np.nanmin(target)
        tmp_shifted[targ_labeled_array != icount+1]=np.nan
        CoMtmp = sp.ndimage.center_of_mass(np.nan_to_num(tmp_shifted,nan=0))
        dist_tmp=math.hypot(CoMtmp[1] - targCoM[1], CoMtmp[0] - targCoM[0])
        targ_distCoM[icount]=dist_tmp
        targ_obj_CoM.append(CoMtmp)

    pred_distCoM=np.zeros(pred_num_features)
    pred_obj_CoM=[]
    for icount in range(pred_num_features):
        tmp_shifted=prediction.copy()-np.nanmin(prediction)
        tmp_shifted[pred_labeled_array != icount+1]=np.nan
        CoMtmp = sp.ndimage.center_of_mass(np.nan_to_num(tmp_shifted,nan=0))
        dist_tmp=math.hypot(CoMtmp[1] - predCoM[1], CoMtmp[0] - predCoM[0])
        pred_distCoM[icount]=dist_tmp
        pred_obj_CoM.append(CoMtmp)

    # Compute weighted average distance of each object from centre of its total field [r]
    targ_wadistCoM = (targTotal*np.abs(targ_distCoM)).sum()/targTotal.sum()
    pred_wadistCoM = (predTotal*np.abs(pred_distCoM)).sum()/predTotal.sum()

    Loc2=2*np.abs(pred_wadistCoM-targ_wadistCoM)/max_d
    
    sal_location = Loc1+Loc2

    # ----------------------------
    # - Structure
    #   Measure of relative 'volume' in objects
    
    # - Compute max value in each object
    targMax=np.zeros(targ_num_features)
    for icount in range(targ_num_features):
        targMax[icount]=target[targ_labeled_array==icount+1].max()
    predMax=np.zeros(pred_num_features)
    for icount in range(pred_num_features):
        predMax[icount]=prediction[pred_labeled_array==icount+1].max()
    
    targVOL=targTotal/targMax
    predVOL=predTotal/predMax
    
    targ_waVOL=(targTotal*targVOL).sum()/targTotal.sum()
    pred_waVOL=(predTotal*predVOL).sum()/predTotal.sum()

    sal_structure = 2*(pred_waVOL-targ_waVOL)/(pred_waVOL+targ_waVOL)    
    
    # ---------------------------- 
    
    sal_xarray=xr.Dataset(
        {
            "sal_s": sal_structure,
            "sal_a": sal_amplitude,
            "sal_l": sal_location,
            "sal_l1": Loc1,
            "sal_l2": Loc2,
            "sal_targ_num": targ_num_features,
            "sal_pred_num": pred_num_features,
        })

    targ_com_array = np.array(targ_obj_CoM)
    pred_com_array = np.array(pred_obj_CoM)
        
    targ_object_xarray=xr.Dataset(
        {
            "obj_com_lat": ("object_num", targ_com_array[:, 0]),
            "obj_com_lon": ("object_num", targ_com_array[:, 1]),
            "obj_max": ("object_num", targMax),
            "obj_size": ("object_num", targSize),
            "obj_labeled_array": (["lat", "lon"], targ_labeled_array),
        })

    pred_object_xarray=xr.Dataset(
        {
            "obj_com_lat": ("object_num", pred_com_array[:, 0]),
            "obj_com_lon": ("object_num", pred_com_array[:, 1]),
            "obj_max": ("object_num", predMax),
            "obj_size": ("object_num", predSize),
            "obj_labeled_array": (["lat", "lon"], pred_labeled_array),
        })
    
    # Option to plot map of original fields and map of objects
    # - Need lat and lon values to plot map
    if lat_values is None or lon_values is None: 
        if filename: print("No lat_values or lon_values supplied so no plot")
        filename=None

    if filename:
        # Re-mask out background for plotting shaded contours:
        #     values <= eThresh or eThreshPr
        #     values not in an object (where objects discarded if too small)
        targ_labeled_array_msk=targ_labeled_array.astype(np.float32)
        targ_labeled_array_msk[target <= eThresh]=0
        targ_labeled_array_msk = np.ma.masked_equal(targ_labeled_array_msk, 0)
        pred_labeled_array_msk=pred_labeled_array.astype(np.float32)
        pred_labeled_array_msk[prediction <= eThreshPr]=0
        pred_labeled_array_msk = np.ma.masked_equal(pred_labeled_array_msk, 0)

        max_num_features = np.array([targ_num_features, pred_num_features]).max()
        sal_txt = (f"S: {sal_structure:.3f}, A: {sal_amplitude:.3f}, "
                   f"L: {sal_location:.3f}, ")
        if max_num_features < 4:
            max_num_features = 4
        plot_sal_objects_map_contourf(
            pred_labeled_array, targ_labeled_array_msk,
            lat_values, lon_values, 
            filepath=filename + '_objects.png',
            title=sal_txt + ", Tr (sh) Pr (cn) Objects",
            cbartitle="",
            levels=np.arange(max_num_features + 1),
            color_map=plt.get_cmap('gist_ncar'),
            FIGWIDTH=8, FIGHEIGHT=6)

        if plot_indiv:
            plot_input_and_objects_map(
                prediction, target,
                pred_labeled_array_msk, targ_labeled_array_msk,
                lat_values, lon_values, 
                filepath=filename + '_fields_objects.png',
                title=sal_txt,
                cbartitle="Orig Data",
                levels_list=[field_levels, np.arange(max_num_features + 1)],
                color_map_list=[plt.get_cmap('Blues'), plt.get_cmap('gist_ncar')],
                FIGWIDTH=12, FIGHEIGHT=10)

    return sal_xarray, targ_object_xarray, pred_object_xarray
# -------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------
# Plotting Code
# -------------------------------------------------------------------------------------
def plot_sal_objects_map_contourf(
    pred_data,
    targ_data,
    lat,
    lon,
    filepath="filepath.png", 
    title="", 
    cbartitle="",
    levels=None, 
    color_map=plt.get_cmap('RdBu_r'), 
    FIGWIDTH=8, 
    FIGHEIGHT=6):

    """
    Plot SAL objects in prediction and target fields (single pair).
    Target (Truth) shaded in colours, Prediction as contour lines, and
    SAL scores in title.
    Added by R. Eade.

    Parameters:
    -----------
    pred_data : numpy.ndarray
        Prediction field of object labels as 2d [lat, lon] array
    targ_data : numpy.ndarray
        Target field of object labels, same shape as pred_data
    lat : numpy.ndarray
        1d array of latitude values matching pred_data
    lon : numpy.ndarray
        1d array of longitude values matching pred_data  
    filepath : str
        Objects map saved to filepath (including .png or equiv at end)
    levels : numpy.ndarray | None
        Optional 1d array of levels to be used by contourf for color
        levels e.g. np.arange(10) where there are 10 objects
    color_map : matplotlib Colormap
        Define colormap for output map plot shading
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs a map saved to filepath.
    
    """

    if targ_data.shape != pred_data.shape:
        raise ValueError(
            f"Shape mismatch: targ_data {targ_data.shape} vs "
            f"pred_data {pred_data.shape}")

    if lat.shape[0] != targ_data.shape[0]:
        raise ValueError(
            f"Shape mismatch: targ_data {targ_data.shape[0]} vs "
            f"lat {lat.shape[0]}")

    if lon.shape[0] != targ_data.shape[1]:
        raise ValueError(
            f"Shape mismatch: targ_data {targ_data.shape[1]} vs "
            f"lat {lon.shape[0]}")
    
    fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(lon, lat, targ_data, levels=levels, cmap=color_map)
    cs1 = ax.contour(lon, lat, pred_data, levels=levels, colors='darkgray')

    # Plot geographic features
    ax.coastlines(resolution='50m')  # '10m' for more detail
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)

    # Draw and label gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(cbartitle)
    
    plt.tight_layout()
    plt.savefig(filepath) # , dpi=150, bbox_inches='tight'
    plt.close()

# -------------------------------------------------------------------------------------
def plot_input_and_objects_map(
    pred_data, 
    targ_data, 
    pred_objs, 
    targ_objs, 
    lat, 
    lon, 
    filepath="filepath.png", 
    title="", 
    cbartitle="", 
    levels_list=[None,None], 
    color_map_list=[plt.get_cmap('Blues'),plt.get_cmap('gist_ncar')], 
    FIGWIDTH=8, 
    FIGHEIGHT=6):

    """
    Plot input prediction and target fields (single pair) and also the 
    SAL objects for each, with SAL scores in title.
    Added by R. Eade.

    Parameters:
    -----------
    pred_data : numpy.ndarray
        Prediction field data as 2d [lat, lon] array
    targ_data : numpy.ndarray
        Target field data, same shape as pred_data
    pred_objs : numpy.ndarray
        Prediction field of object labels as 2d [lat, lon] array
    targ_objs : numpy.ndarray
        Target field of object labels, same shape as pred_objs
    lat : numpy.ndarray
        1d array of latitude values matching data arrays
    lon : numpy.ndarray
        1d array of longitude values matching data arrays    
    filepath : str
        Field & object maps saved to filepath (including .png or equiv)
    title : str
        Optional string of text for title
    cbartitle : str
        Optional string of text for colorbar title
    levels : list of numpy.ndarray or None
        Optional list of 1d arrays of levels to be used by contourf
        [0] for field data (if None, use contourf default)
        [1] for object data e.g. np.arange(10) (None=>contourf default)
    color_map_list : list of matplotlib Colormap
        List defining colormaps for output map plot shading
        [0] for field data
        [1] for object data
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs a 2x2 matrix of maps saved to filepath.
    
    """

    if not (pred_data.shape == targ_data.shape == pred_objs.shape == targ_objs.shape):
        raise ValueError(
            f"All data arrays must have the same shape, got: "
            f"pred_data={pred_data.shape}, targ_data={targ_data.shape}, "
            f"pred_objs={pred_objs.shape}, targ_objs={targ_objs.shape}"
        )
    
    if lat.shape[0] != pred_data.shape[0]:
        raise ValueError(
            f"lat length must match data.shape[0], "
            f"got {lat.shape[0]} and {pred_data.shape[0]}"
        )
    
    if lon.shape[0] != pred_data.shape[1]:
        raise ValueError(
            f"lon length must match data.shape[1], "
            f"got {lon.shape[0]} and {pred_data.shape[1]}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(FIGWIDTH, FIGHEIGHT), squeeze=False, 
        subplot_kw={'projection': ccrs.PlateCarree()})

    # Target Field
    ax1=axes[0,0]
    cf1 = ax1.contourf(lon, lat, targ_data, levels=levels_list[0], 
        cmap=color_map_list[0], extend='both')
    # Plot geographic features
    ax1.coastlines(resolution='50m')  # '10m' for more detail
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    # Draw and label gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude') 
    ax1.set_title('Target Field')

    # Prediction Field
    ax1=axes[0,1]  
    cf2 = ax1.contourf(lon, lat, pred_data, levels=levels_list[0],
        cmap=color_map_list[0], extend='both')
    # Plot geographic features
    ax1.coastlines(resolution='50m')  # '10m' for more detail
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    # Draw and label gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Prediction Field')
    # Add colorbar
    cbar = plt.colorbar(cf2, ax=ax1, orientation='vertical', pad=0.02)
    cbar.set_label(cbartitle)

    # Target Objects
    ax1=axes[1,0]
    cf3 = ax1.contourf(lon, lat, targ_objs, levels=levels_list[1],
        cmap=color_map_list[1])
    # Plot geographic features
    ax1.coastlines(resolution='50m')  # '10m' for more detail
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    # Draw and label gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')  
    ax1.set_title('Target Objects')

    # Prediction Objects
    ax1=axes[1,1]
    cf4 = ax1.contourf(lon, lat, pred_objs, levels=levels_list[1],
        cmap=color_map_list[1])
    # Plot geographic features
    ax1.coastlines(resolution='50m')  # '10m' for more detail
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    # Draw and label gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')  
    ax1.set_title('Prediction Objects')
    # Add colorbar
    cbar2 = plt.colorbar(cf4, ax=ax1, orientation='vertical', pad=0.02)
    cbar2.set_label('Object No.') # , fontsize=11)
    
    plt.tight_layout()
    fig.suptitle(title)
    plt.savefig(filepath)
    plt.close()

# -------------------------------------------------------------------------------------
def plot_3d_sal_scatter(
    sal_vals, 
    filepath='figure.png', 
    ptitle='', 
    ScaleList=[None], 
    color_map='Spectral', 
    ptiles_col=None, 
    FIGWIDTH=8, 
    FIGHEIGHT=6):

    """
    Plot Amplitude (y) vs Structure (x) scatter plot with colours
    representing Location (z)
    As in Wernli et al., 2008
    journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml
    Added by R. Eade.

    Parameters:
    -----------
    sal_vals : xr.Dataset
        Dataset output from compute_sal_xr with s, a and l values vs time
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location
    filepath : str
        Scatterplot saved to filepath (including .png or equiv)
    ptitle : str
        Optional string of text for title
    color_map : matplotlib Colormap
        Define colormap for location score e.g. 'Spectral' 'Reds' 'RdBu_r'
    ptiles_col : str | None
        Option to display s vs v percentiles with the given color string:
        50th %tile lines and 25-75th %tile shaded box
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs a scatterplot saved to filepath.
    
    """

    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')

    x=sal_vals.sal_s
    y=sal_vals.sal_a
    z=sal_vals.sal_l
    xlab='Structure'
    ylab='Amplitude'
    zlab='Location'

    # Reject nonfinite values
    is_real=np.isfinite(x)*np.isfinite(y)*np.isfinite(z)
    x=x[is_real]
    y=y[is_real]
    z=z[is_real]
    
    # Plot 0 lines
    plt.axhline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7, zorder=3)
    plt.axvline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7, zorder=3)

    if ptiles_col:
        # plot shaded rectangle for 25-75th percentiles of x and y.
        plt.plot(np.percentile(x, [25, 75]), np.percentile(y, [50, 50]),
            color=ptiles_col, alpha = 0.8, zorder=3)
        plt.plot(np.percentile(x, [50, 50]), np.percentile(y, [25, 75]),
            color=ptiles_col, alpha = 0.8, zorder=3)

        # This rectangle doesnt show if completely covered by scatter so include zorder
        rect = matplotlib.patches.Rectangle((np.percentile(x,25), np.percentile(y,25)),
            (np.percentile(x,75)-np.percentile(x,25)),
            (np.percentile(y,75)-np.percentile(y,25)), linewidth=1,
            edgecolor=ptiles_col, facecolor=ptiles_col, alpha=0.4, zorder=2)
        ax.add_patch(rect)

        # plot lines for median (50th percentile) of x and y; 
        plt.axhline(np.percentile(y, 50), color ='black', linestyle=':', alpha = 1,
            zorder=3)
        plt.axvline(np.percentile(x, 50), color ='black', linestyle=':', alpha = 1,
            zorder=3)

    useSList=False
    if len(ScaleList)==6: useSList=True
    if useSList==True:
        if ScaleList[0]!=None: MinX=ScaleList[0]
        if ScaleList[1]!=None: MaxX=ScaleList[1]
        if ScaleList[2]!=None: MinY=ScaleList[2]
        if ScaleList[3]!=None: MaxY=ScaleList[3]
        if ScaleList[4]!=None: MinZ=ScaleList[4]
        if ScaleList[5]!=None: MaxZ=ScaleList[5]

    if useSList==True: plt.scatter(x,y,c=y,cmap=color_map,vmin=MinZ,vmax=MaxZ,zorder=1)
    if useSList==False: plt.scatter(x, y, c=y, cmap=color_map, zorder=1)
    if useSList==True: ax.set_xlim(left=MinX, right=MaxX)
    if useSList==True: ax.set_ylim(bottom=MinY, top=MaxY)

    plt.colorbar(label=zlab)
    plt.title(ptitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.savefig(filepath)
    plt.close('all')
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def plot_matrix_1d_sal_pdf(
    sal_list, 
    color_list, 
    label_list, 
    bins_list=[None, None, None, None], 
    ptitle='', 
    filepath='figure.png', 
    PlotRad=False, 
    log=False, 
    density=False, 
    InclKDE=False, 
    KDEapprox=None, 
    InclPerc=0, 
    InclMN=False, 
    InclSD=False, 
    InclWD=False, 
    InclCorr=False, 
    InclRMSE=False, 
    FIGWIDTH=8, 
    FIGHEIGHT=12):

    """
    Plot pdf histograms of sal score values for Structure, Amplitude and
    Location as a column of pdf plots over multiple timesteps for a list
    of different prediction models in different colours. Also option to
    include a pdf of the 3d radii compute from s, a and l.
    Added by R. Eade.

    Parameters:
    -----------
    sal_list : list of xr.Dataset
        Each dataset is output from compute_sal_xr with s, a and l vs time
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location 
    color_list : list of str
        Define linecolor for each dataset in sal_list
    label_list : list of str
        Define label text for each dataset in sal_list e.g. model name
    bins_list : list of numpy.ndarray | None
        Define the bins used for each histogram: s, a, l, radii
        e.g. for whole possible range of values, use:
        [np.arange(-2,2,0.05), np.arange(-2,2,0.05), 
            np.arange(0,2,0.02), np.arange(0,3,0.02)]
        [None, None, None, None] => Use default of hist function for all
    ptitle : str
        Optional string of text for overall title
    filepath : str
        Matrix of pdfs saved to filepath (including .png or equiv)
    PlotRad : boolean
        True: Include pdf plot of radii values (False: don't include)
    log : boolean
        True: plot y axis as log scale (False: don't)
    density : boolean
        True: plot as density rather than frequency
    InclKDE : boolean
        If True: plot kernal density estimate (KDE) of pdf too
    KDEapprox : int | None
        Option for different methods to approx KDE
        0 | None : Standard sp.stats.gaussian_kde() # Slow if large data
        1 : As above but use random subset of data # approx
        2 : gaussian_filter1d from scipy.ndimage # Much faster
    InclPerc : float | None
        > 0: include InclPerc %tile P of score in legend (l and r)
        < 0: include -InclPerc %tile AP of abs(score) in legend (s and a)
        0 or None: don't include
    InclMN : boolean
        True: include pdf mean in legend
    InclSD : boolean
        True: include pdf sd in legend
    InclWD : boolean
        True: include wasserstein distance wrt 1st list element in legend
    InclCorr : boolean
        True: include correlation wrt 1st list element in legend
    InclRMSE : boolean
        True: include RMSE wrt 1st list element in legend
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs a 3x1 or 4x1 column of pdf hist plots saved to filepath:
    S, A, L (and radii)
    
    """

    len_sal_list=len(sal_list)

    # Set up figure (nrows, ncols, ...)
    nrows=3
    if PlotRad is True: nrows=4
    fig, axes = plt.subplots(nrows, 1, figsize=(FIGWIDTH, FIGHEIGHT),
        layout='constrained', squeeze=False)
    if density is True: y_label='density'
    if density is False: y_label='frequency'

    ax = axes[0, 0] # Row, Col
    d_list=[]
    for lcount in range(len_sal_list): d_list.append(sal_list[lcount].sal_s.values)
    plot_1d_sal_pdf(d_list, ax, color_list, label_list, bins = bins_list[0],
        density=density, log=log, InclKDE=InclKDE, KDEapprox=KDEapprox, 
        InclPerc=-1*InclPerc, InclMN=InclMN, InclSD=InclSD, 
        InclWD=InclWD, InclCorr=InclCorr, InclRMSE=InclRMSE, 
        title='Structure', xlab='Structure', ylab=y_label)

    ax = axes[1, 0] # Row, Col
    d_list=[]
    for lcount in range(len_sal_list): d_list.append(sal_list[lcount].sal_a.values)
    plot_1d_sal_pdf(d_list, ax, color_list, label_list, bins = bins_list[1],
        density=density, log=log, InclKDE=InclKDE, KDEapprox=KDEapprox, 
        InclPerc=-1*InclPerc, InclMN=InclMN, InclSD=InclSD, 
        InclWD=InclWD, InclCorr=InclCorr, InclRMSE=InclRMSE, 
        title='Amplitude', xlab='Amplitude', ylab=y_label)
    
    ax = axes[2, 0] # Row, Col
    d_list=[]
    for lcount in range(len_sal_list): d_list.append(sal_list[lcount].sal_l.values)
    plot_1d_sal_pdf(d_list, ax, color_list, label_list, bins = bins_list[2],
        density=density, log=log, InclKDE=InclKDE, KDEapprox=KDEapprox, 
        InclPerc=InclPerc, InclMN=InclMN, InclSD=InclSD, 
        InclWD=InclWD, InclCorr=InclCorr, InclRMSE=InclRMSE, 
        title='Location', xlab='Location', ylab=y_label)


    if PlotRad is True: 
        ax = axes[3, 0] # Row, Col
        d_list=[]
        for lcount in range(len_sal_list): 
            d_list.append(calc_3dradius_sal(sal_list[lcount], percentile=None))
        plot_1d_sal_pdf(d_list, ax, color_list, label_list, bins = bins_list[3],
            density=density, log=log, InclKDE=InclKDE, KDEapprox=KDEapprox, 
            InclPerc=InclPerc, InclMN=InclMN, InclSD=InclSD, 
            InclWD=InclWD, InclCorr=InclCorr, InclRMSE=InclRMSE, 
            title='SAL Radius', xlab='SAL Radius', ylab=y_label)

    fig.suptitle(ptitle)
    plt.savefig(filepath)
    plt.close('all')

# -------------------------------------------------------------------------------------
def plot_1d_sal_pdf(
    data_list, 
    ax1, 
    color_list, 
    label_list, 
    bins = None, 
    log=False, 
    density=False, 
    InclKDE=False, 
    KDEapprox=None, 
    InclPerc=0, 
    InclMN=False, 
    InclSD=False,
    InclWD=False, 
    InclCorr=False, 
    InclRMSE=False, 
    title=None, 
    xlab=None, 
    ylab=None):


    """
    Plot pdf histogram of input data values for a list of different 
    prediction models in different colours.
    Added by R. Eade.

    Parameters:
    -----------
    data_list : list of float
        Data values to be plotted as pdf hist
    ax1 : matplotlib.axes.Axes
        Axes object on which to draw the histogram. Should be an element
        of a subplot grid created via ``plt.subplots()``, e.g.
        ``axes[0, 0]``.
    color_list : list of str
        Define linecolor for each dataset in data_list
    label_list : list of str
        Define label text for each dataset in data_list e.g. model name
    bins : numpy.ndarray | None
        Define the bins used for the histogram
        e.g. for whole possible range of values, use:
        np.arange(-2,2,0.05) or np.arange(0,2,0.02) or np.arange(0,3,0.02)
        None => Use default of hist function
    log : boolean
        True: plot y axis as log scale (False: don't)
    density : boolean
        True: plot as density rather than frequency (False: don't)
    InclKDE : boolean
        If True: plot kernal density estimate (KDE) of pdf too
    KDEapprox : int | None
        Option for different methods to approx KDE
        0 | None : Standard sp.stats.gaussian_kde() # Slow if large data
        1 : As above but use random subset of data # approx
        2 : gaussian_filter1d from scipy.ndimage # Much faster
    InclPerc : float | None
        > 0: include InclPerc %tile P of score in legend (l and r)
        < 0: include -InclPerc %tile AP of abs(score) in legend (s and a)
        0 or None: don't include
    InclMN : boolean
        True: include pdf mean in legend
    InclSD : boolean
        True: include pdf sd in legend
    InclWD : boolean
        True: include wasserstein distance wrt 1st list element in legend
    InclCorr : boolean
        True: include correlation wrt 1st list element in legend
    InclRMSE : boolean
        True: include RMSE wrt 1st list element in legend 
    title : str | None
        Option to define text for title
    xlab : str | None
        Option to define text for x-axis label
    ylab : str | None
        Option to define text for y-axis label

    Returns:
    --------
    Outputs pdf hist plot to axes object ax1 (element of a subplot grid)
    e.g. as called by plot_matrix_1d_sal_pdf()
    
    """
    
    len_data_list=len(data_list)    

    # If only 1 dataset, have nothing to compare to
    if len_data_list==1: InclWD=False
    if len_data_list==1: InclCorr=False
    if len_data_list==1: InclRMSE=False
    
    label_list_tmp=label_list.copy()
    
    addlabel=0 # count no. additions to legend labels (for adjusting text size later)
    
    # If datasets arent paired, can't compute paired diagnostics (Corr and RMSE)
    len_data_list_each=np.zeros(len_data_list)
    for dcount in range(len_data_list): 
        len_data_list_each[dcount]=len(data_list[dcount])
    unq_len=np.unique(len_data_list_each)
    if len(unq_len) > 1: 
        InclCorr=False
        InclRMSE=False
        print(f"Warning: data lengths different so can't compute paried diagnostics")

    if InclPerc>0:
        # Compute Percentile of values of data_list (e.g. for SAL Radius)
        RParray=np.zeros(len_data_list)
        for dcount in range(len_data_list):
            RParray[dcount]=np.nanpercentile(data_list[dcount], InclPerc)
        for dcount in range(len_data_list):
            label_list_tmp[dcount] += f",  {int(InclPerc)}P={RParray[dcount]:.3f}"

    if InclPerc<0:
        # Compute Percentile of abs values of data_list (e.g. for SAL Radius)
        RParray=np.zeros(len_data_list)
        for dcount in range(len_data_list):
            RParray[dcount]=np.nanpercentile(np.abs(data_list[dcount]), -1*InclPerc)
        for dcount in range(len_data_list):
            label_list_tmp[dcount] += f", {-1*int(InclPerc)}AP={RParray[dcount]:.3f}"
    
    if np.abs(InclPerc)>0: addlabel=addlabel+1


    if InclMN==True:
        # Compute Mean of data_list
        MNarray=np.zeros(len_data_list)
        for dcount in range(len_data_list):
            MNarray[dcount]=np.nanmean(data_list[dcount])
        for dcount in range(len_data_list): 
            label_list_tmp[dcount] += f", MN={MNarray[dcount]:.3f}"
        addlabel=addlabel+1

    if InclSD==True:
        # Compute SD of data_list
        SDarray=np.zeros(len_data_list)
        for dcount in range(len_data_list):
            SDarray[dcount]=np.nanstd(data_list[dcount])
        for dcount in range(len_data_list):
            label_list_tmp[dcount] += f", SD={SDarray[dcount]:.3f}"
        addlabel=addlabel+1
    
    if InclWD==True:
        # Compute WD wrt 1st element in data_list
        WDarray=np.zeros(len_data_list-1)
        ## Reject nonfinite values
        is_real=np.isfinite(data_list[0])
        data_tmp0=data_list[0][is_real]
        for dcount in range(len_data_list-1): 
            ## Reject nonfinite values
            is_real=np.isfinite(data_list[dcount+1])
            data_tmp=data_list[dcount+1][is_real]
            WDarray[dcount]=sp.stats.wasserstein_distance(data_tmp0,data_tmp)
            label_list_tmp[dcount+1] += f", WD={WDarray[dcount]:.3f}"
        addlabel=addlabel+1

    if InclCorr==True:
        # Compute Pearson Correlation wrt 1st element in data_list (assumes paired data and ignores non-finite values)
        CORRarray=np.zeros(len_data_list-1)
        for dcount in range(len_data_list-1): 
            CORRarray[dcount]=calculate_pearsoncorr_nparray(data_list[0],
                data_list[dcount+1], axis=None)
            label_list_tmp[dcount+1] += f", r={CORRarray[dcount]:.3f}"
        addlabel=addlabel+1

    if InclRMSE==True:
        # Compute RMSE wrt 1st element in data_list (assumes paired data and ignores non-finite values)
        RMSEarray=np.zeros(len_data_list-1)
        for dcount in range(len_data_list-1):
            RMSEarray[dcount]=calculate_error_nparray(data_list[0],
                data_list[dcount+1], axis=None, typeerror='rmse')
            label_list_tmp[dcount+1] += f", RM={RMSEarray[dcount]:.3f}"
        addlabel=addlabel+1
    
    if InclKDE is True:
        x_min_list=[]
        x_max_list=[]
        for dcount in range(len_data_list):
            x_min_list.append(np.nanmin(data_list[dcount]))
            x_max_list.append(np.nanmax(data_list[dcount]))
        x_min=np.nanmin(np.array(x_min_list))
        x_max=np.nanmax(np.array(x_max_list))
        x_vals = np.linspace(x_min, x_max, 1000)
        density=True # Normalises pdf such that area == 1

    NANexist=False
    for dcount in range(len_data_list):
        ## Reject nonfinite values before plotting
        is_real=np.isfinite(data_list[dcount])
        data_tmp=data_list[dcount][is_real]
        if len(is_real[~is_real])>0: NANexist=True

        ax1.hist(data_tmp, bins=bins, edgecolor=color_list[dcount], 
            color=color_list[dcount], log=log, density=density, 
            label=label_list_tmp[dcount], alpha=0.3)
        
        if InclKDE is True and KDEapprox is None:
            kde = sp.stats.gaussian_kde(data_tmp) # Very slow if large data
            pdf_vals = kde(x_vals)
        if InclKDE is True and KDEapprox == 1:
            # Use random subset of data instead of all of the data
            # < 1 sec for ~2e6 data points but tail too smooth
            if data_tmp <= 10000: 
                kde = sp.stats.gaussian_kde(data_tmp)
                pdf_vals = kde(x_vals)
            if data_tmp > 10000: 
                rng = np.random.default_rng(82)
                tmp_sample = rng.choice(data_tmp, size=10000, replace=False)
                kde = sp.stats.gaussian_kde(tmp_sample) 
                pdf_vals = kde(x_vals)
        if InclKDE is True and KDEapprox == 2:
            # Use gaussian_filter1d from scipy.ndimage instead of sp.stats.gaussian_kde
            # < 1 sec for ~2e6 data points and very similar to default method
            grid_sz = 1024
            hist_counts, bin_edges = np.histogram(data_tmp, bins=grid_sz, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_centres[1] - bin_centres[0]
            bw = 1.06 * data_tmp.std() * len(data_tmp) ** -0.2   # Scott's rule
            sigma_bins = bw / bin_width                  # bandwidth in grid units
            pdf_vals = sp.ndimage.gaussian_filter1d(hist_counts, sigma=sigma_bins)
            x_vals=bin_centres

        if InclKDE is True: 
            ax1.plot(x_vals, pdf_vals, color=color_list[dcount],
                linewidth=2, linestyle='-')

    if log is True: ax1.set_yscale("log")

    # Plot 0 lines
    ax1.axvline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7)

    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.set_title(title) 
    fontsizestr=''
    if addlabel>1: fontsizestr='small'
    if addlabel>3: fontsizestr='x-small'
    if addlabel==0: ax1.legend(loc="upper right")
    if addlabel>0: ax1.legend(loc="upper right", fontsize=fontsizestr)
   
    if NANexist:
        print(f"Warning: Nonfinite values removed from input data before plotting")


# -------------------------------------------------------------------------------------
def plot_matrix_2d_sal_pdf(
    sal1, 
    sal2=None, 
    ptitle='', 
    ScaleListList=[None,None,None], 
    color_map='Reds', 
    filepath='figure.png', 
    InclScat=False, 
    InclWD=True, 
    level_count=5,
    FIGWIDTH=8, 
    FIGHEIGHT=12):

    """
    Plot 2d pdfs of pairs of sal score values as a column of plots for a
    single prediction model or a pair of different prediction models 
    (shading vs contour lines). Plots are in the order:
    Amplitude vs Structure
    Amplitude vs Location
    Locations vs Structure
    Added by R. Eade.

    Parameters:
    -----------
    sal1 : xr.Dataset
        Dataset output from compute_sal_xr with s, a and l vs time
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location 
    sal2 : xr.Dataset | None
        Optional 2nd dataset to compare with first
    ptitle : str
        Optional string of text for overall title
    ScaleListList : list of lists of floats | list of None
        A list of ScaleList
        ScaleList : list of 6 floats | None
        Info for x and y axes range and colorbar levels
        [float MinX, float MaxX, float MinY, float MaxY, 
            float vmin, float vmax] (vmin and vmax for colorbar)
        Use None for values that don't want to specify
        e.g. ScaleListList=[[-2,2,-2,2,0,1],[0,2,-2,2,0,1],[-2,2,0,2,0,5]]
        e.g. ScaleListList=[[-2,2,-2,2,0,None],[-2,2,0,2,0,None],
            [-2,2,0,2,0,None]]
        e.g. ScaleListList=[None,None,None]
    color_map : matplotlib.colors.Colormap
        Colormap used for coloring the plot. Can be a named colormap
        string (e.g. ``'viridis'``, ``'plasma'``) passed via
        ``plt.get_cmap()``, or a ``Colormap`` instance directly.
    filepath : str
        Matrix of pdfs saved to filepath (including .png or equiv)
    InclScat : boolean
        True: include scatterplot too (overlaid on pdf contours)
    InclWD : boolean
        True: include wasserstein distance between 2 datasets in legend
    level_count : int
        Number of levels to include in colorbar (same for all plots)
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs a 3x1 column of 2d pdf plots saved to filepath:
    A vs S; A vs L; L vs S
    
    """
    
    if not isinstance(level_count, (int, float)): level_count=5
    if isinstance(level_count, float): level_count=int(level_count)
    if level_count <5: level_count=5

    # Set up figure (nrows, ncols, ...)
    fig, axes = plt.subplots(3, 1, figsize=(FIGWIDTH, FIGHEIGHT),
        layout='constrained', squeeze=False)
    
    TwoVar=True
    if sal2 is None: TwoVar=False

    if TwoVar is False:
        # 1 Model prediction
        # axes[Row, Col]
        ax = axes[0, 0]
        plot_2d_sal_pdf(sal1.sal_s.values, sal1.sal_a.values, None, None,
            ax, title='A vs S', xlab='structure', ylab='amplitude', 
            ScaleList=ScaleListList[0], InclScat=InclScat, InclWD=InclWD,
            level_count=level_count, color_map=color_map)
        
        ax = axes[1, 0]
        plot_2d_sal_pdf(sal1.sal_l.values, sal1.sal_a.values, None, None,
            ax, title='A vs L', xlab='location', ylab='amplitude',
            ScaleList=ScaleListList[1], InclScat=InclScat, InclWD=InclWD,
            level_count=level_count, color_map=color_map)
        
        ax = axes[2, 0]
        plot_2d_sal_pdf(sal1.sal_s.values, sal1.sal_l.values, None, None,
            ax, title='L vs S', xlab='structure', ylab='location',
            ScaleList=ScaleListList[2], InclScat=InclScat, InclWD=InclWD,
            level_count=level_count, color_map=color_map)

    if TwoVar is True:
        # 2 Model predictions
        # axes[Row, Col]
        ax = axes[0, 0]
        plot_2d_sal_pdf(sal1.sal_s.values, sal1.sal_a.values,
            sal2.sal_s.values, sal2.sal_a.values, 
            ax, title='A vs S', xlab='structure', ylab='amplitude', 
            ScaleList=ScaleListList[0], InclScat=InclScat, InclWD=InclWD,
            level_count=level_count, color_map=color_map)
        ax = axes[1, 0]
        plot_2d_sal_pdf(sal1.sal_l.values, sal1.sal_a.values,
            sal2.sal_l.values, sal2.sal_a.values,
            ax, title='A vs L', xlab='location', ylab='amplitude',
            ScaleList=ScaleListList[1], InclScat=InclScat, InclWD=InclWD,
            level_count=level_count, color_map=color_map)

        ax = axes[2, 0]
        plot_2d_sal_pdf(sal1.sal_s.values, sal1.sal_l.values,
            sal2.sal_s.values, sal2.sal_l.values, 
            ax, title='L vs S', xlab='structure', ylab='location', 
            ScaleList=ScaleListList[2], InclScat=InclScat, InclWD=InclWD, 
            level_count=level_count, color_map=color_map)

    fig.suptitle(ptitle)
    plt.savefig(filepath)
    plt.close('all')

# -------------------------------------------------------------------------------------
def plot_2d_sal_pdf(
    x, 
    y, 
    x2, 
    y2, 
    ax1, 
    ScaleList=[None],
    color_map='Reds',
    InclScat=False, 
    InclWD=True, 
    level_count=5, 
    title='',  
    xlab='', 
    ylab='',
    print_crange=False):
    """
    Plot 2d PDF of a pair of variables, comparing predictions to target
    Added by R. Eade.

    Parameters:
    -----------
    x : numpy.ndarray
        Time series data for variable x, dataset 1
        (plotted as shading)
    y : numpy.ndarray
        Time series data for variable y, dataset 1
    x2 : numpy.ndarray | None
        Optional 2nd time series data for variable x, dataset 2
        (plotted as contours)
    y2 : numpy.ndarray | None
        Optional 2nd time series data for variable y, dataset 2
    ax1 : matplotlib.axes.Axes
        Axes object on which to draw the histogram. Should be an element
        of a subplot grid created via ``plt.subplots()``, e.g.
        ``axes[0, 0]``.
    ScaleList : list of 6 floats
        Info for x and y axes range and colorbar levels
        [float MinX, float MaxX, float MinY, float MaxY,
            float vmin, float vmax] (vmin and vmax for colorbar)
        Use None for values that don't want to specify
    color_map : matplotlib.colors.Colormap
        Colormap used for coloring the plot. Can be a named colormap
        string (e.g. ``'viridis'``, ``'plasma'``) passed via
        ``plt.get_cmap()``, or a ``Colormap`` instance directly.
    InclScat : boolean
        True: include scatterplot too (overlaid on pdf contours)
    InclWD : boolean
        True: include wasserstein distance between 2 datasets in legend
    level_count : int
        Number of levels to include in colorbar
    title : str | None
        Option to define text for title
    xlab : str | None
        Option to define text for x-axis label
    ylab : str | None
        Option to define text for y-axis label
    print_crange : boolean
        True: Print range of kernal density (zmin zmax) to help refine
        choice in ScaleList

    Returns:
    --------
    Outputs 2d pdf plot to axes object ax1 (element of a subplot grid)
    e.g. as called by plot_matrix_2d_sal_pdf()

    See Also
    --------
    scipy.stats.gaussian_kde :
    Kernel density non-parametric estimation of PDF of a random variable.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    numpy.vstack : 
    Stack arrays in sequence vertically 
    https://numpy.org/doc/2.2/reference/generated/numpy.vstack.html

    Method removes NaN and Inf from input arrays before plotting

    """
    
    if not isinstance(level_count, (int, float)): level_count=5
    if isinstance(level_count, float): level_count=int(level_count)
    if level_count <5: level_count=5


    TwoVar=True
    if x2 is None: TwoVar=False
    if y2 is None: TwoVar=False
    if TwoVar==False: InclWD=False

    if InclWD==False:
        is_real=np.isfinite(x)*np.isfinite(y)
        x=x[is_real]
        y=y[is_real]
        if TwoVar==True:
            is_real2=np.isfinite(x2)*np.isfinite(y2)
            x2=x2[is_real2]
            y2=y2[is_real2]

    if InclWD==True:
        is_real=np.isfinite(x)*np.isfinite(y)*np.isfinite(x2)*np.isfinite(y2)
        x=x[is_real]
        y=y[is_real]
        if TwoVar==True:
            x2=x2[is_real]
            y2=y2[is_real]

    kde1 = sp.stats.gaussian_kde([x, y])
    if TwoVar is True: kde2 = sp.stats.gaussian_kde([x2, y2])
    
    # Set X and Y axis range
    # Default is to use data
    if TwoVar is True:
        MinX=np.array([x.min(),x2.min()]).min()
        MaxX=np.array([x.max(),x2.max()]).max()
        MinY=np.array([y.min(),y2.min()]).min()
        MaxY=np.array([y.max(),y2.max()]).max()
    if TwoVar is False:
        MinX=x.min()
        MaxX=x.max()
        MinY=y.min()
        MaxY=y.max()
    
    useSList=False
    if len(ScaleList)==6: useSList=True    
    
    if useSList==True:
        # ScaleList : [MinX, MaxX, MinY, MaxY, zmin, zmax]
        if ScaleList[0]!=None: MinX=ScaleList[0]
        if ScaleList[1]!=None: MaxX=ScaleList[1]
        if ScaleList[2]!=None: MinY=ScaleList[2]
        if ScaleList[3]!=None: MaxY=ScaleList[3]

    # Create meshgrid for evaluation
    xi = np.linspace(MinX, MaxX, 100)
    yi = np.linspace(MinY, MaxY, 100)

    Xi, Yi = np.meshgrid(xi, yi)
    zi1 = kde1(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    if TwoVar is True:
        zi2 = kde2(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    zmin=zi1.min()
    zmax=zi1.max()
    if zmax==0.0: zmax=0.001
    if print_crange is True: print(f"zmin={zmin}, zmax={zmax}")

    if useSList==True:
        # ScaleList : [MinX, MaxX, MinY, MaxY, zmin, zmax]
        if ScaleList[4]!=None: zmin=ScaleList[4]
        if ScaleList[5]!=None: zmax=ScaleList[5]

    # Determine shared contour levels based on the range of target KDE (shading)
    levels = np.linspace(zmin, zmax, level_count+1)  # 6 points = 5 levels

    # Plot with contourf and contour for labels
    clwd=1.0
    if InclScat==True: clwd=0.5
    ax = plt.gca()
    cf = ax1.contourf(Xi, Yi, zi1, levels=levels, cmap=color_map, extend='both')

    if InclScat==True:
        ax1.scatter(x, y, s=7, color="cyan", alpha=0.4)
        if TwoVar is True: ax1.scatter(x2, y2, s=5, color="darkblue", alpha=0.4)

    cs1 = ax1.contour(Xi, Yi, zi1, levels=levels, linewidths=clwd,
        colors='black', alpha=0.3)
    if TwoVar is True: 
        cs2 = ax1.contour(Xi, Yi, zi2, levels=levels, linewidths=clwd, colors='black')

    plt.colorbar(cf, ax=ax1, label='Density')
    ax1.clabel(cs1, inline=True, fontsize=8, fmt='%.3f')
    if TwoVar is True: ax1.clabel(cs2, inline=True, fontsize=8, fmt='%.3f')

    # Plot 0 lines
    ax1.axhline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7)
    ax1.axvline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7) 

    # Plot median lines
    ax1.axhline(np.percentile(y, 50), color ='cyan', linestyle=':',
        linewidth=0.5, alpha = 1)
    ax1.axvline(np.percentile(x, 50), color ='cyan', linestyle=':',
        linewidth=0.5, alpha = 1)
    if TwoVar is True:
        ax1.axhline(np.percentile(y2, 50), color ='darkblue', linestyle=':',
            linewidth=0.5, alpha = 1)
        ax1.axvline(np.percentile(x2, 50), color ='darkblue', linestyle=':',
            linewidth=0.5, alpha = 1)
    
    MinXY=np.array([MinX,MinY]).min()
    MaxXY=np.array([MaxX,MaxY]).max()

    if InclWD==True:
        WassDist=sp.stats.wasserstein_distance_nd(np.column_stack((x,y)).T,
            np.column_stack((x2,y2)).T)
        WassDistTxt="%7.3f" % WassDist
        plt.plot(np.array([MinX,MinX]), np.array([MaxX,MaxX]), '.',
            label='WD='+WassDistTxt, color='white')
        ax1.legend(loc="upper left")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.set_title(title) 
    
    ax1.set_xlim(left=MinX, right=MaxX)
    ax1.set_ylim(bottom=MinY, top=MaxY)
# -------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------
#https://matplotlib.org/stable/users/explain/colors/colormaps.html
# -------------------------------------------------------------------------------------