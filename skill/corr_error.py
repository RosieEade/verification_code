"""
Code to compute verification skill metrics
Prediction vs Target (truth)

    calculate_error_nparray()
    - compute different types of error metrics on numpy arrays
      over all or subset of axes

    calculate_pearsoncorr_nparray()
    - compute Pearson correlation metric on numpy arrays
      over all or subset of axes

Copyright (c) 2026 Klima consulting
Author: Rosie Eade
 
"""

import numpy as np

# -------------------------------------------------------------------------------------
# Computation Code
# -------------------------------------------------------------------------------------
def calculate_error_nparray(arr0, arr1, axis=0, typeerror='rmse'):
    """
    Calculate (rms) error between 2 N-dimensional numpy arrays
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

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
    
    # Mask positions where either array is NaN, so both are excluded pairwise
    nan_mask = np.isnan(arr0) | np.isnan(arr1)
    arr0 = np.where(nan_mask, np.nan, arr0)
    arr1 = np.where(nan_mask, np.nan, arr1)

    # Compute Error over axis/axes specified
    # - subtract first then take mean so consistent method for bias and rmse
    # - better level of accuracy than taking means then subtract (slight difference)
    if typeerror=='bias':
        output_error = np.nanmean(arr1-arr0, axis=axis)
    if typeerror=='abias':
        output_error = np.nanmean(np.abs(arr1-arr0), axis=axis)
    if typeerror=='rmse':
        output_error = np.sqrt(np.nanmean((arr1-arr0)**2, axis=axis))
    if typeerror=='ratio':
        output_error = np.nanmean(arr1, axis=axis)/np.nanmean(arr0, axis=axis)
    
    return output_error

# -------------------------------------------------------------------------------------
def calculate_pearsoncorr_nparray(arr0, arr1, axis=0):
    """
    Calculate Pearson correlation between 2 N-dimensional numpy arrays
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

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
    
    # Mask positions where either array is NaN, so both are excluded pairwise
    nan_mask = np.isnan(arr0) | np.isnan(arr1)
    arr0 = np.where(nan_mask, np.nan, arr0)
    arr1 = np.where(nan_mask, np.nan, arr1)

    # Center the data over axis/axes specified
    arr0_centered = arr0 - np.nanmean(arr0, axis=axis, keepdims=True)
    arr1_centered = arr1 - np.nanmean(arr1, axis=axis, keepdims=True)
    
    # Compute correlation over axis/axes specified
    numerator = np.nansum(arr0_centered * arr1_centered, axis=axis)
    denominator = np.nansum(arr0_centered**2, axis=axis) * np.nansum(arr1_centered**2, axis=axis)
    denominator = np.sqrt(denominator)

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator!=0)
    
    return correlations

# -------------------------------------------------------------------------------------