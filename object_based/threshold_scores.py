"""
Code to compute threshold based scores for pairs of gridded fields:
Prediction and Target (truth)

    compute_ets()
    - compute the Equitable Threat Score (ETS) for paired xarray DataArrays.
    - Option to compute the Threat Score instead

    compute_proximity_distance()
    - compute the proximity distance
    
    compute_sfss2D
    - compute the spatial fractional skill score


Copyright (c) 2026 Klima consulting
Author: Rosie Eade
 
"""

import numpy as np
import scipy as sp
import xarray as xr


# ---------------------------------------------------------------------------------------
# Computation Code
# ---------------------------------------------------------------------------------------
def make_mask(input, eThresh, relation='gt'):
    
    comparison_map = {
        'gt': lambda x, t: x > t,
        'lt': lambda x, t: x < t,
        'ge': lambda x, t: x >= t,
        'le': lambda x, t: x <= t,
        'eq': lambda x, t: x == t,
        'ne': lambda x, t: x != t,
    }
    
    if relation not in comparison_map:
        raise ValueError(f"relation must be one of {list(comparison_map.keys())}, got '{relation}'")
    
    return comparison_map[relation](input, eThresh) & np.isfinite(input)

# ---------------------------------------------------------------------------------------
def compute_ets(prediction, target, eThresh, relation='gt', axis=None, TSopt=False, print_ct=False):
    '''
    Equitable Threat Score (ETS)
    Threshold based binary fields [0, 1] = [none-event, event]
    Compute ETS (or Threat Score if TSopt=True) wrt given threshold.

    a=number of true positive, b=number of false positives, c=number of false negatives, d=number of true negatives

    TS = a/(a+b+c) = hits / (hits + false alarms + misses)
    
    ETS = (a-r)/(a+b+c-r) = (hits - expected_hits) / (hits + false alarms + misses - expected_hits)

    r = (a+b)(a+c)/(a+b+c+d) = predicted number of positives *  target number of positives / total number of points = expected number of hits by chance

    ETS = 0 for random and 1 for perfect model. 
    
    Based on https://cfs.ncep.noaa.gov/GFS_test/NewIce/www/precip/precip_body.htm
    
    Works with xarray or nparray as input. Outputs in same format as input.
    
    Parameters:
    -----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    eThresh : float
        Event Threshold to define binary event
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'
    axis : (tuple of) int or (tuple of) str
        Axis or axes to compute score over, default=None i.e. over all axes.
    TSopt : bool
        True implies compute Threat Score instead of ETS.
    print_ct : bool
        True implies print related contingency table values to terminal,
        only if axis=None (avoids printing out large arrays per table element)

    Return
    -----------
    numpy.ndarray | xr.DataArray
        Array of ETS (or Threat Score), same format as input target
    
    '''

    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")

    if axis is not None:
        print_ct = False

    # -------------------------------
    # Make sure axis is a tuple in correct form for np.array or xr.DataArray
    if isinstance(axis, type(None)):
        if isinstance(target, xr.DataArray): axis = target.dims
        if not isinstance(target, xr.DataArray): axis = tuple(int(x) for x in np.arange(target.ndim))
    
    if isinstance(axis, int) or isinstance(axis, str):
        axis = (axis,)
    
    axis_ok = 0
    if isinstance(axis, tuple) and all(isinstance(item, int) for item in axis):
        axis_ok = 1
    elif isinstance(axis, tuple) and all(isinstance(item, str) for item in axis):
        axis_ok = 2
    else:
        raise ValueError(f"axis expected (tuple of) str or int, got: {axis}")
    
    # If axis is a tuple of str and input is a np.array then can't identify axes
    if axis_ok == 2 and not isinstance(target, xr.DataArray):
        raise ValueError(f"can't use axis of strings with a np array")

    # If axis is a tuple of str and input is xr.DataArray, then identify axis numbers
    if axis_ok == 2 and isinstance(target, xr.DataArray):
        dim_positions = []
        for target_dim in axis:
            try:
                dim_positions.append(target.dims.index(target_dim))
            except ValueError:
                raise ValueError(f"Dimension '{target_dim}' not found in {target.dims}")
        axis = tuple(dim_positions)
    # -------------------------------

    # False/True fields
    prediction_masked = make_mask(prediction, eThresh, relation=relation)
    target_masked = make_mask(target, eThresh, relation=relation)
    
    # Compute Hit/Miss Contingency Table
    Hits=target_masked*prediction_masked         # a
    FalseAlarms=~target_masked*prediction_masked # b
    Misses=target_masked*~prediction_masked      # c
    TrueNegs=~target_masked*~prediction_masked   # d
    
    # Sum over all or sub-axes
    nHits=np.sum(Hits, axis=axis)
    nMisses=np.sum(Misses, axis=axis)
    nFalseAlarms=np.sum(FalseAlarms, axis=axis)
    nTrueNegs=np.sum(TrueNegs, axis=axis)
    
    if print_ct:
        # Print contingency table individual values
        # - Only if axis=None i.e. 1 value per table element
        print(f"Hits: {np.sum(nHits)}")
        print(f"Misses: {np.sum(nMisses)}")
        print(f"FalseAlarms: {np.sum(nFalseAlarms)}")
        print(f"TrueNegs: {np.sum(nTrueNegs)}")
        print(f"Sum-Total = {np.sum(nHits)+np.sum(nMisses)+np.sum(nFalseAlarms)+np.sum(nTrueNegs)-target_masked.size}") # Check
    
    exp_nHits = (nHits + nFalseAlarms) * (nHits + nMisses) / (nHits+nMisses+nFalseAlarms+nTrueNegs)
    
    # *1.0 to make sure arrays are floats not ints
    # What if denom = 0? 
    # -- Invalid so set as np.nan
    
    ets_val = np.nan

    if not TSopt:
        # ets_val = (nHits - exp_nHits) / (nHits+nFalseAlarms+nMisses-exp_nHits)
        numerator1 = (nHits - exp_nHits)*1.0
        denominator1 = (nHits+nFalseAlarms+nMisses-exp_nHits)*1.0
        ets_val = np.divide(numerator1, denominator1, out=np.full_like(numerator1, np.nan), where=denominator1!=0) # np.zeros_like(numerator1)
    
    if TSopt:
        # ts_val = nHits / (nHits+nFalseAlarms+nMisses)
        numerator2=nHits*1.0
        denominator2=(nHits+nFalseAlarms+nMisses)*1.0
        ets_val = np.divide(numerator2, denominator2, out=np.full_like(numerator2, np.nan), where=denominator2!=0)

    return ets_val


# ----------------------------------------------------------------------------
def compute_proximity_distance(prediction, target, eThresh, relation='gt', axis=('lat','lon'), sampling=None, f2inverse=False):
    '''
    Proximity distance
    Threshold based binary fields [0, 1] = [none-event, event]
    Transform to distance based field so each point = distance from nearest 1.
    - in grid box space (with fixed sampling option), 
    - caveat: does not account for unequal grid spacing 
      e.g. E-W distance in km can vary a lot from low to high lat
    
    Output field has same shape as input field, but distances are only
    calculated in the dimensions implied by axis.
    
    Resample based on just the cases where target has an event (or doesn't, if f2inverse True)
    => distribution of distance of prediction (forecast) events from target (target) events.
    
    Parameters
    ----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    eThresh : float
        Event Threshold to define binary event
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'
    axis : (tuple of) int or (tuple of) str
        Axes or dims to compute distances over (default: ('lat','lon'))
        If input as np.array, expects axis as tuple of int.
        If input as xr.DataArray, expects axis as tuple of str.
    sampling : tuple, optional
        Sampling for distance_transform_edt, tuple same size as axis
    f2inverse : bool, optional
        If True, instead resample for cases where target doesn't have event
    
    Return
    ------
    numpy.ndarray | xr.DataArray
        Array of proximity distance, same format as input target
    
    '''

    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")

    # -------------------------------
    # Make sure axis is a tuple in correct form for np.array or xr.DataArray
    if isinstance(axis, type(None)):
        if isinstance(target, xr.DataArray): axis = target.dims
        if not isinstance(target, xr.DataArray): axis = tuple(int(x) for x in np.arange(target.ndim))
    
    if isinstance(axis, int) or isinstance(axis, str):
        axis = (axis,)
    
    axis_ok = 0
    if isinstance(axis, tuple) and all(isinstance(item, int) for item in axis):
        axis_ok = 1
    elif isinstance(axis, tuple) and all(isinstance(item, str) for item in axis):
        axis_ok = 2
    else:
        raise ValueError(f"axis expected (tuple of) str or int, got: {axis}")
   
    # If axis is a tuple of str and input is a np.array then can't identify axes
    if axis_ok == 2 and not isinstance(target, xr.DataArray):
        raise ValueError(f"Cannot use axis of str with a np.array, need int values.")

    # If axis is a tuple of str and input is xr.DataArray, then check all dims exist
    if axis_ok == 2 and isinstance(target, xr.DataArray):
        dim_positions = []
        for target_dim in axis:
            try:
                dim_positions.append(target.dims.index(target_dim))
            except ValueError:
                raise ValueError(f"Dimension '{target_dim}' not found in {target.dims}.")

    # If axis is a tuple of int and input is xr.DataArray, convert to tuple of str
    if axis_ok==1 and isinstance(prediction, xr.DataArray):
        try:
            dimensions = tuple(prediction.dims[i] for i in axis)
        except IndexError:
            raise ValueError(f"Cannot match all values in axis {axis}. Target has only {len(target.dims)} dimensions.")
        axis = dimensions
    # -------------------------------

    # False/True fields
    prediction_masked = make_mask(prediction, eThresh, relation=relation)
    target_masked = make_mask(target, eThresh, relation=relation)

    prediction_bin=prediction.copy()
    prediction_mindistance=prediction*0.0    

    if isinstance(prediction, xr.DataArray):
        prediction_bin.values[prediction_masked]=1.0
        prediction_bin.values[~prediction_masked]=0.0
        prediction_dims = list(prediction.dims)
    if isinstance(prediction, np.ndarray):
        prediction_bin[prediction_masked]=1.0
        prediction_bin[~prediction_masked]=0.0
        prediction_dims = [i for i in range(prediction.ndim)]

    # Get all axes that are NOT in axis
    output_dims = [d for d in prediction_dims if d not in axis]

    # Create the transpose order: other_axes first, then axis
    ordered_dims = output_dims + list(axis)

    # Transpose both arrays
    prediction_mindistance_rs = prediction_mindistance.transpose(*ordered_dims)
    prediction_bin_rs = prediction_bin.transpose(*ordered_dims)

    # Calculate n_total as product of all dimensions except the last 2
    n_total = int(np.prod(prediction_mindistance_rs.shape[0:-2]))

    # Convert to np array if needed
    if isinstance(prediction, xr.DataArray):
        prediction_bin_rs_np=prediction_bin_rs.values
        prediction_mindistance_rs_np=prediction_mindistance_rs.values
    else:
        prediction_bin_rs_np=prediction_bin_rs.copy()
        prediction_mindistance_rs_np=prediction_mindistance_rs.copy()

    # flatten so has dimensions (concat output_dims, axis dims)
    prediction_bin_rs_flat = prediction_bin_rs_np.reshape(n_total, *prediction_bin_rs.shape[-2:])
    prediction_mindistance_rs_flat = prediction_mindistance_rs_np.reshape(n_total, *prediction_mindistance_rs.shape[-2:])
    for tcount in range(n_total):
        prediction_mindistance_rs_flat[tcount] = sp.ndimage.distance_transform_edt((prediction_bin_rs_flat[tcount]-1)*-1, sampling=sampling)
    
    # unflatten and reshape back to original prediction
    prediction_mindistance_rs_flat = prediction_mindistance_rs_flat.reshape(*prediction_mindistance_rs.shape)
    if isinstance(prediction, xr.DataArray):
        prediction_mindistance_rs.values = prediction_mindistance_rs_flat
    if isinstance(prediction, np.ndarray):
        prediction_mindistance_rs = prediction_mindistance_rs_flat
    
    prediction_mindistance = prediction_mindistance_rs.transpose(*prediction_dims)

    # Sub-sample to include just the points where target has event
    # or if f2inverse==True: resample for cases where target doesnt have an event
    if isinstance(prediction, xr.DataArray):
        if not f2inverse: prediction_mindistance = prediction_mindistance.where(target_masked)
        if f2inverse: prediction_mindistance = prediction_mindistance.where(~target_masked)
    if isinstance(prediction, np.ndarray):
        if not f2inverse: prediction_mindistance = np.where(~target_masked, np.nan, prediction_mindistance)
        if f2inverse: prediction_mindistance = np.where(target_masked, np.nan, prediction_mindistance)

    return prediction_mindistance

# ----------------------------------------------------------------------------
def compute_sfss2D(prediction, target, eThresh, relation='gt', window=1):

    '''
    Spatial Fractional Skill Score
    Threshold based binary fields [0, 1] = [none-event, event]
    Compute for each 2d image separately e.g. for each timestep
    ***Assumes SFSS computed on 2d fields defined by final 2 dimensions in array***
    e.g. [ time, lat, lon ] => compute for lat/lon fields at each timestep.
    
    Roberts, N. M., & Lean, H. W. (2008). Scale-Selective Verification of 
    Rainfall Accumulations from High-Resolution Forecasts of Convective 
    Events, https://doi.org/10.1175/2007MWR2123.1
    

    Parameters
    ----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
        ***Assumes spatial dimensions are final 2 in array***
        e.g. [time, member, lat, lon]
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    eThresh : float
        Event Threshold to define binary event
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'
    window : int
        Size of square window for allowed neighbouts, centred on point;
        window=1 reduces to a pointwise comparison.
    
    Return
    ------
    numpy.ndarray | xr.DataArray
        Array of SFSS, same format as input target
    
    '''
    
    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    
    # False/True fields
    prediction_masked = make_mask(prediction, eThresh, relation=relation)
    target_masked = make_mask(target, eThresh, relation=relation)

    prediction_bin=prediction.copy()
    target_bin=target.copy()

    if isinstance(prediction, xr.DataArray):
        prediction_bin.values[prediction_masked]=1.0
        prediction_bin.values[~prediction_masked]=0.0
        target_bin.values[target_masked]=1.0
        target_bin.values[~target_masked]=0.0
    if isinstance(prediction, np.ndarray):
        prediction_bin[prediction_masked]=1.0
        prediction_bin[~prediction_masked]=0.0
        target_bin[target_masked]=1.0
        target_bin[~target_masked]=0.0
    
    # Define filter_axes, used for assigning neighbours within window
    if len(prediction.shape) <= 2: filter_axes=None
    if len(prediction.shape) > 2: filter_axes=(-1,-2)
    
    # Generate fractions: for each point, compute fraction of surrounding points 
    # within given length scale (window) that have event (value = 1)    
    # e.g. use smoothing fn so Xij becomes the average of all values in surrounding square window
    # For values at edge, this fills empty part of window with zeros
    # - outputs np.ndarray regardless if input type

    prediction_frac = sp.ndimage.uniform_filter(prediction_bin.astype(float), size=window, axes=filter_axes, mode="constant", cval=0.0)
    target_frac = sp.ndimage.uniform_filter(target_bin.astype(float), size=window, axes=filter_axes, mode="constant", cval=0.0)
    
    if isinstance(prediction, xr.DataArray):
        tmp=prediction.copy()
        tmp.values=prediction_frac
        prediction_frac=tmp.copy()
        tmp=target.copy()
        tmp.values=target_frac
        target_frac=tmp.copy()
        tmp=0.0
    
    print(type(prediction_frac))

    # Compute MSE between 2 fields
    mse = ((prediction_frac - target_frac) ** 2).mean(axis=filter_axes)
    
    # Compute MSE reference <= worst (largest) possible MSE (assumes completely unaligned fields)
    mse_ref = (prediction_frac ** 2).mean(axis=filter_axes) + (target_frac ** 2).mean(axis=filter_axes)
    # Replace 0s with nan
    if isinstance(prediction, xr.DataArray):
        mse_ref = mse_ref.where(mse_ref != 0)
    if isinstance(prediction, np.ndarray):
        mse_ref = np.where(mse_ref == 0, np.nan, mse_ref)
    
    # Compute SFFS
    mse_ratio = mse / mse_ref
    sfss_out = 1.0 - mse_ratio

    return sfss_out
