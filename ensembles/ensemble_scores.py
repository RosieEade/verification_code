"""
Code to compute ensemble and probabilistic based scores for pairs of gridded fields:
Prediction and Target (truth) in the form of numpy.ndarray or xr.DataArray

    compute_crps()
    - compute the continous rank probability score (CRPS)

    compute_bs()
    - compute the Brier Score (BS)
    
    compute_mssr
    - compute the mean squared skill ratio (MSSR)


Copyright (c) 2026 Klima consulting
Author: Rosie Eade
 
"""

import numpy as np
import scipy as sp
import xarray as xr


# ---------------------------------------------------------------------------------------
# Computation Code
# ---------------------------------------------------------------------------------------
def make_mask(input, eThresh, relation):
    """
    Convert input to bool field based on occurence or non-occurence
    of events relative to the threshold using relation e.g. 'gt' eThresh

    Parameters
    ----------
    input : numpy.ndarray | xr.DataArray | xr.Dataset
        N-dimensional array Prediction (same shape as target)
    eThresh : float | int | numpy.ndarray | xr.DataArray | xr.Dataset
        Threshold used to define the binary event.
        Broadcasting rules need to be observed if eThresh has
        different type or dimensions to input, e.g. numpy broadcasting,
        can't be xr.Dataset if input is not.
    relation : str
        Relationship of event to threshold eThresh: 'gt', 'ge', 'lt', 'le'.

    Returns
    -------
    numpy.ndarray | xr.DataArray | xr.Dataset
        Array of mask, same format & shape as input target.

    Raises
    ------
    ValueError
        If there are mismatches between input and eThresh dimensions
        or if relation str no recognised.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    """

    # Special case conversion so output same format as input
    if isinstance(eThresh, xr.Dataset) and not isinstance(input, xr.Dataset):
            raise ValueError(f"Can't match eThresh xr.Dataset to input type {type(input)} for mask")

    if isinstance(eThresh, xr.DataArray):
        if isinstance(input, np.ndarray): eThresh = eThresh.values
    
    if isinstance(eThresh, xr.DataArray) or isinstance(eThresh, np.ndarray):
        if eThresh.ndim > input.ndim:
            raise ValueError(f"eThresh has more dims than input so can't match for mask")
    
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
def compute_crps(prediction, target, member_dim, axis=None, fair_est=True):

    """
    Compute the continous rank probability score for two xarray Datasets,
    using the form:
    CRPS(X, y) = E|X - y| - 1/2 * E|X - X'|


    Parameters
    ----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    member_dim : int or str
        prediction dimension that identifies the ensemble member dimension
        If input as np.array, expects int.
        If input as xr.DataArray, expects str (or can convert from int).
    axis : (tuple of) int or (tuple of) str
        target axis or axes to aggregate over over
        default=None => don't aggregate.
        If input as np.array, expects axis as tuple of int.
        If input as xr.DataArray, expects axis as tuple of str (or can convert from int).
    fair_est : bool
        True includes option to adjust to fair spread estimate,
        i.e. ignores pairs i=j on diagonal.

    Returns
    -------
    numpy.ndarray | xr.DataArray
        Array of crps, same format as input target.

    Raises
    ------
    ValueError
        If there are mismatches between prediction and target dimensions
        or also wrt member_dim or axis specifications.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    """

    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    if isinstance(axis, int): axis = (axis,)
    if isinstance(axis, str): axis = (axis,)
    if isinstance(axis, float): axis = (axis,)
    
    if isinstance(axis, tuple) and any(isinstance(item, float) for item in axis):
        orig_axis = axis
        axis = tuple(map(int, axis))
        print(f"Warning: converting axis of floats {orig_axis} to ints {axis}")

    # member_dim = 'member' or 0
    if isinstance(prediction, xr.DataArray):

        if isinstance(axis, tuple) and all(isinstance(item, int) for item in axis):
            try:
                dimensions = tuple(target.dims[i] for i in axis)
            except IndexError:
                raise ValueError(f"Cannot match all values in axis {axis}. Target has only {len(target.dims)} dimensions.")
            axis = dimensions

        if isinstance(member_dim, int): member_dim = prediction.dims[member_dim]

        set_pred_dims = set(prediction.dims)

        if not member_dim in set_pred_dims:
            raise ValueError(f"member_dim: {member_dim} not found in prediction dims {prediction.dims}")
        set_targ_dims = set(target.dims)
        set_pred_dims_nomem = set_pred_dims - set((member_dim,))

        if set_targ_dims != set_pred_dims_nomem:
            raise ValueError(f"Dimension mismatch of non-member dims: P {prediction.dims} vs T {target.dims}")

        pred_nomem_shape = tuple(prediction.sizes[dim] for dim in prediction.dims if dim != member_dim)
        if target.shape != pred_nomem_shape:
            raise ValueError(f"Shape mismatch of non-member dims: P {pred_nomem_shape} vs T {target.shape}")

        nmembers = prediction[member_dim].size

    # member_dim = 0
    if isinstance(prediction, np.ndarray):

        if isinstance(axis, tuple) and all(isinstance(item, str) for item in axis):
            raise ValueError(f"Cannot use axis of str with a np.array, need int values.")

        if isinstance(member_dim, str):
            raise ValueError(f"Can't use member_dim as str with a np.array")
	
        if member_dim >= prediction.ndim or member_dim < -1*prediction.ndim:
            raise ValueError(f"member_dim: {member_dim} outside prediction ndim {prediction.ndim}")

        if prediction.ndim - 1 != target.ndim:
            raise ValueError(f"Dimension mismatch of non-member dims: P {prediction.ndim-1} vs T {target.ndim}")

        pred_nomem_shape = prediction.shape[:member_dim] + prediction.shape[member_dim+1:]
        if pred_nomem_shape != target.shape:
            raise ValueError(f"Shape mismatch of non-member dims: P {prediction.shape} vs T {target.shape}")

        nmembers = prediction.shape[member_dim]

        if member_dim != 0:
            prediction = np.moveaxis(prediction, member_dim, 0)
            member_dim = 0

    if nmembers < 2:
        fair_est = False

    # Accuracy
    accuracy = np.abs(prediction - target).mean(member_dim) # check works for xr.DataArray and np.array

    # Spread
    if isinstance(prediction, xr.DataArray):
        pred_i = prediction
        pred_j = prediction.rename({member_dim: member_dim + "_2"})
        spread = np.abs(pred_i - pred_j).mean([member_dim, member_dim + "_2"])
    if isinstance(prediction, np.ndarray):
        pred_i = np.expand_dims(prediction, axis=member_dim + 1)
        pred_j = np.expand_dims(prediction, axis=member_dim)
        spread = np.abs(pred_i - pred_j).mean(axis=(member_dim, member_dim + 1))

    if fair_est:
        spread = spread * (nmembers / (nmembers - 1))

    crps_out = accuracy - 0.5 * spread

    if axis is not None:
        crps_out = crps_out.mean(axis)

    return crps_out


# ---------------------------------------------------------------------------------------
def compute_bs(prediction, target, member_dim, eThresh, relation, axis=None, fair_est=True):

    """
    Compute the Brier Skill for two xarray Datasets,
    relative to a given threhold. This reduces to the CRPS
    on binary version of fields (event vs non-event) with
    probability field p in [0, 1], using the form:
    E|X - y| - 1/2*E|X - X'| == (p - y)^2


    Parameters
    ----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    member_dim : int or str
        prediction dimension that identifies the ensemble member dimension
        If input as np.array, expects int.
        If input as xr.DataArray, expects str (or can convert from int).
    eThresh : float | int | numpy.ndarray | xr.DataArray
        Threshold used to define the binary event.
        If an array, should be same shape as target or follow        
        numpy broadcasting rules wrt target.
    relation : str
        Relationship of event to threshold eThresh: 'gt', 'ge', 'lt', 'le'.
    axis : (tuple of) int or (tuple of) str
        target axis or axes to aggregate over over
        default=None => don't aggregate.
        If input as np.array, expects axis as tuple of int.
        If input as xr.DataArray, expects axis as tuple of str (or can convert from int).
    fair_est : bool
        True includes option to adjust to fair spread estimate,
        i.e. ignores pairs i=j on diagonal.

    Returns
    -------
    numpy.ndarray | xr.DataArray
        Array of Brier skill, same format as input target.

    Raises
    ------
    ValueError
        If there are mismatches between prediction and target dimensions
        or also wrt member_dim or axis specifications.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    """

    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    # Convert both fields to binary (0/1) event indicators, preserving
    # all dims/coords (including the member dim on x1).
    targ_mask = make_mask(target, eThresh, relation)
    pred_mask = make_mask(prediction, eThresh, relation)
    targ_binary = (targ_mask).astype(int)
    pred_binary = (pred_mask).astype(int)

    # CRPS of binary fields == (fair) Brier Score.
    brier_skill = compute_crps(pred_binary, targ_binary, member_dim, axis=axis, fair_est=fair_est)

    return brier_skill


# ---------------------------------------------------------------------------------------
def compute_mssr(prediction, target, member_dim, axis=None):

    """
    Compute the mean spread skill ratio for two xarray Datasets:
    = SQRT(MEAN(VARIANCE)) / RMSE of Ens Mean

    Parameters
    ----------
    prediction : numpy.ndarray | xr.DataArray
        N-dimensional array Prediction (same shape as target)
    target : numpy.ndarray | xr.DataArray
        N-dimensional array Target (same shape as prediction)
    member_dim : int or str
        prediction dimension that identifies the ensemble member dimension
        If input as np.array, expects int.
        If input as xr.DataArray, expects str (or can convert from int).
    axis : (tuple of) int or (tuple of) str
        target axis or axes to aggregate over over
        default=None => aggregate over all dimensions
        If input as np.array, expects axis as tuple of int.
        If input as xr.DataArray, expects axis as tuple of str (or can convert from int).

    Returns
    -------
    numpy.ndarray | xr.DataArray
        Array of mean spread skill ratio, same format as input target.
        e.g. axis=['time'] -> output dims ['lat','lon']
        ~1 = well calibrated
        >1 => over-dispersive
        <1 => under-dispersive

    Raises
    ------
    ValueError
        If there are mismatches between prediction and target dimensions
        or also wrt member_dim or axis specifications.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    """

    # Check input array types
    if type(prediction) != type(target):
        raise ValueError(f"input type mismatch: {type(prediction)} vs {type(target)}")

    if not isinstance(prediction, xr.DataArray) and not isinstance(prediction, np.ndarray):
        raise ValueError(f"input type error, expected np.ndarray or xr.DataArray but got: {type(prediction)}")

    if isinstance(axis, int): axis = (axis,)
    if isinstance(axis, str): axis = (axis,)
    if isinstance(axis, float): axis = (axis,)
    
    if isinstance(axis, tuple) and any(isinstance(item, float) for item in axis):
        orig_axis = axis
        axis = tuple(map(int, axis))
        print(f"Warning: converting axis of floats {orig_axis} to ints {axis}")

    # member_dim = 'member' or 0
    if isinstance(prediction, xr.DataArray):

        if isinstance(axis, tuple) and all(isinstance(item, int) for item in axis):
            try:
                dimensions = tuple(target.dims[i] for i in axis)
            except IndexError:
                raise ValueError(f"Cannot match all values in axis {axis}. Target has only {len(target.dims)} dimensions.")
            axis = dimensions

        if isinstance(member_dim, int): member_dim = prediction.dims[member_dim]

        set_pred_dims = set(prediction.dims)

        if not member_dim in set_pred_dims:
            raise ValueError(f"member_dim: {member_dim} not found in prediction dims {prediction.dims}")
        set_targ_dims = set(target.dims)
        set_pred_dims_nomem = set_pred_dims - set((member_dim,))

        if set_targ_dims != set_pred_dims_nomem:
            raise ValueError(f"Dimension mismatch of non-member dims: P {prediction.dims} vs T {target.dims}")

        pred_nomem_shape = tuple(prediction.sizes[dim] for dim in prediction.dims if dim != member_dim)
        if target.shape != pred_nomem_shape:
            raise ValueError(f"Shape mismatch of non-member dims: P {pred_nomem_shape} vs T {target.shape}")

        nmembers = prediction[member_dim].size

    # member_dim = 0
    if isinstance(prediction, np.ndarray):

        if isinstance(axis, tuple) and all(isinstance(item, str) for item in axis):
            raise ValueError(f"Cannot use axis of str with a np.array, need int values.")

        if isinstance(member_dim, str):
            raise ValueError(f"Can't use member_dim as str with a np.array")
	
        if member_dim >= prediction.ndim or member_dim < -1*prediction.ndim:
            raise ValueError(f"member_dim: {member_dim} outside prediction ndim {prediction.ndim}")

        if prediction.ndim - 1 != target.ndim:
            raise ValueError(f"Dimension mismatch of non-member dims: P {prediction.ndim-1} vs T {target.ndim}")

        pred_nomem_shape = prediction.shape[:member_dim] + prediction.shape[member_dim+1:]
        if pred_nomem_shape != target.shape:
            raise ValueError(f"Shape mismatch of non-member dims: P {prediction.shape} vs T {target.shape}")

        nmembers = prediction.shape[member_dim]

        if member_dim != 0:
            prediction = np.moveaxis(prediction, member_dim, 0)
            member_dim = 0

    if nmembers < 2:
        fair_est = False

    # Ensemble mean and spread with correction for small samples
    ens_mean = prediction.mean(member_dim)
    ens_var = prediction.var(member_dim, ddof=0) * (nmembers + 1) / (nmembers - 1)

    # Error of ensemble mean vs target
    error = ens_mean - target
    sq_error = error**2

    rmserror = np.sqrt(sq_error.mean(axis))
    mean_spread = np.sqrt(ens_var.mean(axis))

    spread_skill_ratio = mean_spread / rmserror

    return spread_skill_ratio
