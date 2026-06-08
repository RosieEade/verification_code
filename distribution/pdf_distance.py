"""
Code to compare probablity distribution functions (pdf) of
lat/lon gridded fields using distance metrics and skill scores:

Prediction and Target (truth)

    calc_hist_metrics_xr()
    - compute set of distance metrics and skill scores between the histograms
     of 2 N-dimensional xarray DataArrays

    calc_pdf_metric_xr()
    - compute single distance metric or skill score between the pdfs
     of 2 N-dimensional xarray DataArrays (includes histogram based metrics)

Plotting code also provided for standard summary plots
distributions

Copyright (c) 2026 Klima consulting
Author: Rosie Eade
 
"""

import numpy as np
import scipy as sp
import xarray as xr

# -------------------------------------------------------------------------------------
# Computation Code
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def calc_hist_metrics_nparray(
    arr0, 
    arr1, 
    axis=0, 
    bins_arr=None, 
    nbins=100, 
    printBins=False):
    """
    Calculate distance metrics and skill scores between the histograms of
    2 N-dimensional numpy arrays
    - over all dimensions (axes) or
    - over 0th dimension (axis) (may need to reshape input arrays first)
    Computes bin probabilities, then uses
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      (good for very large data)
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : 0 | None
        None : compute over all axes
        0 : compute over 0th axis so output of form N-1 dimensions
    bins_arr : numpy.ndarray | None
        Explicitly define bins for histogram (default method, else use nbins)
        e.g. np.arange(0,1200,20)
        None => Use default of hist function with nbins
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
        
    Returns:
    --------
    numpy.ndarray | float
        Jenson Shannon Distance.
    numpy.ndarray | float
        Perkins skill score.
    numpy.ndarray | float
        Wasserstein Distance.

    if axis==None: Output is a float
    if axis==0: Output has N - 1 dimensions 
        (input shape with the 0th axis removed).

    See Also
    --------
    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>
    
    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml
    
    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    
    """

    axis_list=[None, 0]
    if not axis in axis_list:
        raise ValueError(f"axis not recognised as None or 0: {axis}")
    
    jsd_out=np.nan
    pss_out=np.nan
    wdh_out=np.nan
    
    if not isinstance(bins_arr, np.ndarray): bins_arr=None

    if isinstance(bins_arr, type(None)):
        if not isinstance(nbins, (int, float, np.int32, np.int64, np.float32, np.float64)):
            raise ValueError(f"nbins not recognised as integer: {nbins}")
        if nbins<1:
            raise ValueError(f"nbins not a positive integer: {nbins}")

        nbins=int(nbins)
        arr_min=np.array([arr0.min(),arr1.min()]).min()
        arr_max=np.array([arr0.max(),arr1.max()]).max()
        bins_arr = np.linspace(arr_min, arr_max, nbins + 1)
        # This means that the bins will be different for different arr1 model inputs
        # but provides flexibility to make sure the whole distributions are captured
        #These are the same methods but will have problems if arr1 extremes are very different
        #bins_arr = np.histogram_bin_edges(arr0, nbins)
        #bins_arr = np.linspace(arr0.min(), arr0.max(), nbins + 1)
    
    if printBins==True:
        print(f"edges: {bins_arr[0]}, {bins_arr[1]},... {bins_arr[-1]}")
        print(f"nbins: {len(bins_arr)-1}")
    
    if axis==0 and len(arr0.shape) < 2:  axis=None
    
    if axis is None:
        phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
        qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

        # Normalise so sums to 1
        phist = phist / phist.sum()
        qhist = qhist / qhist.sum()
    
        jsd_out=float(sp.spatial.distance.jensenshannon(phist, qhist))
    
        pss_out=float(np.sum(np.minimum(phist, qhist)))

        
        # Use bin midpoints as representative values
        bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2
        
        # scipy's wasserstein_distance accepts weighted samples:
        # pass bin centers as "samples" and normalized counts as weights
        wdh_out = sp.stats.wasserstein_distance(
            bin_centres, bin_centres,
            u_weights=phist,
            v_weights=qhist)

    if axis==0:
        phist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0)
        qhist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1)
    
        # Normalise so sums to 1 at each grid point?
        phist = phist / phist.sum(axis=0)
        qhist = qhist / qhist.sum(axis=0)
    
        jsd_out=sp.spatial.distance.jensenshannon(phist, qhist)
        jsd_out=jsd_out.astype(float)
    
        pss_out=np.sum(np.minimum(phist, qhist), axis=0).astype(float)
        
        # Use bin midpoints as representative values
        bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

        # scipy's wasserstein_distance accepts weighted samples:
        # pass bin centers as "samples" and normalized counts as weights
        wdh_out=np.zeros(np.prod(arr0.shape[1:]))
        phist_flat=phist.reshape([phist.shape[0], np.prod(phist.shape[1:])])
        qhist_flat=qhist.reshape([qhist.shape[0], np.prod(qhist.shape[1:])])
        
        for ii in range(np.prod(arr0.shape[1:])):
            wdh_out[ii]=sp.stats.wasserstein_distance(
                bin_centres, bin_centres,
                u_weights=phist_flat[:,ii],
                v_weights=qhist_flat[:,ii])

        wdh_out=wdh_out.reshape(arr0.shape[1:])
    
    return jsd_out, pss_out, wdh_out

# -------------------------------------------------------------------------------------
def calc_hist_metrics_xr(
    target_xr, 
    prediction_xr, 
    dim_over=None,
    member_name=None,
    emean=False,
    bins_arr=None,
    nbins=100,
    printBins=False):

    '''
    Calculate distance metrics and skill scores between the histograms of
    2 N-dimensional xarray DataArrays
    - over all dimensions: dim_over=None, or
    - over subset of dimensions: e.g. dim_over=('time',)
    Computes bin probabilities, then uses
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      (good for very large data)

    **Assumes data arrays have the same size, though prediction_xr may have
      an additional ensemble dimension: member_name must be specified to
    match the name of the member dimensions
    **If prediction_xr is an ensemble and emean is True, the array 
    mean will be taken over this member dimension before computing the 
    distance metric. If emean is False, the distance metric will be 
    computed for each member separately.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    target_xr : xr.DataArray
        Target field data, same shape as prediction_xr but without 
        ensemble dimension.
    prediction_xr : xr.DataArray
        Prediction field data. Must contain 'lat' and 'lon' dimensions,
        plus any number of leading dimensions (e.g. [time, lat, lon] or
        [time, member, lat, lon]).
    dim_over : tuple of str | None
        Define dimensions to compute distance over e.g.
        ('time,) or ('member', 'time')
        None : compute distance over all data flattened
    member_name : str | None
        Define name of member dimension (ensemble prediction) e.g. 
        'member' or 'realization'
    emean : booleon
        True : Option to compute ensemble mean of prediction over 
        member_name dim if available        
    bins_arr : numpy.ndarray | None
        Define the bins used for the histogram
        e.g. np.arange(0,1200,20)
        None => Use default of hist function based on arr0
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
        
    Returns:
    --------
    xr.Dataset
        'jsd' Jenson Shannon Distance.
        'pss' Perkins skill score.
        'wdh' Wasserstein Distance (over histogram).
        Floats if dim_over=None
        Else a subset of input dimension e.g. [lat, lon] or [member, lat, lon]

    if axis==None: Output is a float
    if axis==0: Output has N - 1 dimensions 
        (input shape with the 0th axis removed).

    See Also
    --------
    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>
    
    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml
    
    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

    '''

    if dim_over==None: dim_over=prediction_xr.dims

    # Detect if dim_over dimensions present in prediction
    pred_dims = list(prediction_xr.dims)
    for dname in dim_over:
        if dname not in pred_dims:
            raise ValueError(
                f"dim_over dimension '{dname}' not found in dims {pred_dims}. "
                f"Expected something like 'time', 'member', etc."
            )

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim = None
    if member_name in prediction_xr.dims and member_name not in target_xr.dims:
        extra_member_dim = member_name
        if emean==True:
            # Compute ensemble mean over member_name dim
            prediction_xr = prediction_xr.mean(dim=member_name)
            extra_member_dim = None
            pred_dims = list(prediction_xr.dims)

    if extra_member_dim is not None:
        target_xr = target_xr.expand_dims(
            {extra_member_dim: prediction_xr[extra_member_dim]}
        )
        target_xr = target_xr.broadcast_like(prediction_xr)

    if target_xr.values.shape != prediction_xr.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target_xr.values.shape}), "
            f"Prediction ({prediction_xr.values.shape})"
            f"Specify member_name if ensemble prediction)"
        )

    # Identify output dimensions i.e. those not in dim_over dimensions
    output_dims = [d for d in pred_dims if d not in dim_over]

    # --- Special case: output_dims empty, ---
    #  e.g. if prediction dims just [member, time]
    #  => compute jsd over all data
    if not output_dims:
        jsd_out, pss_out, wdh_out=calc_hist_metrics_nparray(target_xr.values, 
            prediction_xr.values, axis=None, bins_arr=bins_arr, nbins=nbins,
            printBins=printBins)
            
        dist_out_xr = xr.Dataset(
            {
                "jsd":        (output_dims, jsd_out),
                "pss":        (output_dims, pss_out),
                "wdh":        (output_dims, wdh_out),
            }
        )

        return dist_out_xr

    # Transpose to ensure dim_over dims come first, output_dims last e.g. spatial dims:
    # e.g. [member, time, lat, lon]
    ordered_dims = list(dim_over) + output_dims
    target_xr     = target_xr.transpose(*ordered_dims)
    prediction_xr = prediction_xr.transpose(*ordered_dims)

    # Flatten all dim_over dims into a single axis so can compute over 0th dim
    dimover_shape = target_xr.values.shape[0:len(dim_over)] # e.g. [member, time]
    output_shape = target_xr.values.shape[len(dim_over):]   # e.g. [lat, lon]
    n_total = int(np.prod(dimover_shape))

    target_flat     = target_xr.values.reshape(n_total, *output_shape)
    prediction_flat = prediction_xr.values.reshape(n_total, *output_shape)
    
    jsd_out, pss_out, wdh_out=calc_hist_metrics_nparray(target_flat, prediction_flat,
        axis=0, bins_arr=bins_arr, nbins=nbins, printBins=printBins)

    dist_out_xr = xr.Dataset(
        {
            "jsd":        (output_dims, jsd_out),
            "pss":        (output_dims, pss_out),
            "wdh":        (output_dims, wdh_out),
        }
    )

    # Restore coordinates for all output dimensions
    for dim in output_dims:
        if dim in prediction_xr.coords:
            dist_out_xr[dim] = prediction_xr[dim]

    return dist_out_xr

# -------------------------------------------------------------------------------------
def calc_pdf_metric_nparray(
    arr0, 
    arr1, 
    axis=0, 
    bins_arr=None, 
    nbins=100, 
    typedistance='jsd', 
    printBins=False):
    """
    Calculate Distance metrics between 2 N-dimensional numpy arrays
    - over all dimensions (axes) or
    - over 0th dimension (axis) (may need to reshape input arrays first)
    Computes bin probabilities, then uses typedistance string:
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      fast for very large data
    - wd: scipy.stats.wasserstein_distance on data (not histogram)
      slow for very large data
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : 0 | None
        None : compute over all axes
        0 : compute over 0th axis so output of form N-1 dimensions
    bins_arr : numpy.ndarray | None
        Explicitly define bins for histogram (default method, else use nbins)
        e.g. np.arange(0,1200,20)
        None => Use default of hist function with nbins
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
    typedistance : str
        Define type of distance or score to compute
        
    Returns:
    --------
    numpy.ndarray | float
        Jenson Shannon Distance. 
        if axis==None: Output is a float
        if axis==0: Output has N - 1 dimensions 
        (input shape with the 0th axis removed).

    See Also
    --------
    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>
    
    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml
    
    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    
    """

    distance_list=['jsd', 'pss', 'wdh', 'wd']
    if not typedistance in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    axis_list=[None, 0]
    if not axis in axis_list:
        raise ValueError(f"axis not recognised as None or 0: {axis}")
    
    dist_out=np.nan
    
    if not isinstance(bins_arr, np.ndarray): bins_arr=None
    
    if isinstance(bins_arr, type(None)):
        if not isinstance(nbins, (int, float, np.int32, np.int64, np.float32, np.float64)):
            raise ValueError(f"nbins not recognised as integer: {nbins}")
        if nbins<1:
            raise ValueError(f"nbins not a positive integer: {nbins}")
        nbins=int(nbins)
        arr_min=np.array([arr0.min(),arr1.min()]).min()
        arr_max=np.array([arr0.max(),arr1.max()]).max()
        bins_arr = np.linspace(arr_min, arr_max, nbins + 1)
        # This means that the bins will be different for different arr1 model inputs
        # but provides flexibility to make sure the whole distributions are captured
        #These are the same methods but will have problems if arr1 extremes are very different
        #bins_arr = np.histogram_bin_edges(arr0, nbins)
        #bins_arr = np.linspace(arr0.min(), arr0.max(), nbins + 1)
    
    if printBins==True:
        print(f"edges: {bins_arr[0]}, {bins_arr[1]},... {bins_arr[-1]}")
        print(f"nbins: {len(bins_arr)-1}")
    
    if axis==0 and len(arr0.shape) < 2:  axis=None
    
    if axis is None:
        if typedistance=='jsd':
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()
        
            dist_out=float(sp.spatial.distance.jensenshannon(phist, qhist))

        if typedistance=='pss':
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()
        
            dist_out=float(np.sum(np.minimum(phist, qhist)))

        if typedistance=='wdh':
            #dist_out=sp.stats.wasserstein_distance(arr0,arr1)
            
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)
            
            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2
            
            # scipy's wasserstein_distance accepts weighted samples:
            # pass bin centers as "samples" and normalized counts as weights
            dist_out = sp.stats.wasserstein_distance(
                bin_centres, bin_centres,
                u_weights=phist,
                v_weights=qhist
            )
        if typedistance=='wd':
            arr0_flat=arr0.reshape(np.prod(arr0.shape))
            arr1_flat=arr1.reshape(np.prod(arr1.shape))
            dist_out=sp.stats.wasserstein_distance(arr0_flat,arr1_flat)


    if axis==0:
        if typedistance=='jsd':
            phist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0)
            qhist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1)
        
            # Normalise so sums to 1 at each grid point?
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)
        
            dist_out=sp.spatial.distance.jensenshannon(phist, qhist)
            dist_out=dist_out.astype(float)

        if typedistance=='pss':
            phist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0)
            qhist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)
        
            dist_out=np.sum(np.minimum(phist, qhist), axis=0).astype(float)

        if typedistance=='wdh':
            phist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0)
            qhist=np.apply_along_axis(lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)
            
            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # scipy's wasserstein_distance accepts weighted samples:
            # pass bin centers as "samples" and normalized counts as weights
            dist_out=np.zeros(np.prod(arr0.shape[1:]))
            phist_flat=phist.reshape([phist.shape[0], np.prod(phist.shape[1:])])
            qhist_flat=qhist.reshape([qhist.shape[0], np.prod(qhist.shape[1:])])
            print(phist_flat.shape)
            print(qhist_flat.shape)
            
            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii]=sp.stats.wasserstein_distance(
                    bin_centres, bin_centres,
                    u_weights=phist_flat[:,ii],
                    v_weights=qhist_flat[:,ii])

            dist_out=dist_out.reshape(arr0.shape[1:])

        if typedistance=='wd':
            dist_out=np.zeros(np.prod(arr0.shape[1:]))
            print(dist_out.shape)
            arr0_flat=arr0.reshape(arr0.shape[0],np.prod(arr0.shape[1:]))
            arr1_flat=arr1.reshape(arr1.shape[0],np.prod(arr1.shape[1:]))
            print(arr0_flat.shape)
            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii]=sp.stats.wasserstein_distance(arr0_flat[:,ii],arr1_flat[:,ii])

            dist_out=dist_out.reshape(arr0.shape[1:])
    
    return dist_out


# -------------------------------------------------------------------------------------
def calc_pdf_metric_xr(
    target_xr, 
    prediction_xr, 
    dim_over=None,
    member_name=None,
    emean=False,
    bins_arr=None,
    nbins=100,
    printBins=False,
    typedistance='jsd'):

    '''
    Compute Distance metrics between 2 N-dimensional numpy arrays
    - over all dimensions (axes) or
    - over 0th dimension (axis) (may need to reshape input arrays first)
    Computes bin probabilities, then uses typedistance string:
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      good for very large data
    - wd: scipy.stats.wasserstein_distance on data (not histogram)
      slow for very large data

    **Assumes data arrays have the same size, though prediction_xr may have
      an additional ensemble dimension: member_name must be specified to
    match the name of the member dimensions
    **If prediction_xr is an ensemble and emean is True, the array 
    mean will be taken over this member dimension before computing the 
    distance metric. If emean is False, the distance metric will be 
    computed for each member separately.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    target_xr : xr.DataArray
        Target field data, same shape as prediction_xr but without 
        ensemble dimension.
    prediction_xr : xr.DataArray
        Prediction field data. Must contain 'lat' and 'lon' dimensions,
        plus any number of leading dimensions (e.g. [time, lat, lon] or
        [time, member, lat, lon]).
    dim_over : tuple of str | None
        Define dimensions to compute distance over e.g.
        ('time,) or ('member', 'time')
        None : compute distance over all data flattened
    member_name : str | None
        Define name of member dimension (ensemble prediction) e.g. 
        'member' or 'realization'
    emean : booleon
        True : Option to compute ensemble mean of prediction over 
        member_name dim if available        
    bins_arr : numpy.ndarray | None
        Define the bins used for the histogram
        e.g. np.arange(0,1200,20)
        None => Use default of hist function based on arr0
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
    typedistance : str
        Define type of distance or score to compute
    
    Returns:
    --------
    xr.Dataset
        Arrays of Jensen-Shannon distances for sets of input target and 
        prediction fields.
        Float if dim_over=None
        Else a subset of input dimension e.g. [lat, lon] or [member, lat, lon]

    See Also
    --------
    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>
    
    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml
    
    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

    '''

    distance_list=['jsd', 'pss', 'wdh', 'wd']
    if not typedistance in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    if dim_over==None: dim_over=prediction_xr.dims

    # Detect if dim_over dimensions present in prediction
    pred_dims = list(prediction_xr.dims)
    for dname in dim_over:
        if dname not in pred_dims:
            raise ValueError(
                f"dim_over dimension '{dname}' not found in dims {pred_dims}. "
                f"Expected something like 'time', 'member', etc."
            )

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim = None
    if member_name in prediction_xr.dims and member_name not in target_xr.dims:
        extra_member_dim = member_name
        if emean==True:
            # Compute ensemble mean over member_name dim
            prediction_xr = prediction_xr.mean(dim=member_name)
            extra_member_dim = None
            pred_dims = list(prediction_xr.dims)

    if extra_member_dim is not None:
        target_xr = target_xr.expand_dims(
            {extra_member_dim: prediction_xr[extra_member_dim]}
        )
        target_xr = target_xr.broadcast_like(prediction_xr)

    if target_xr.values.shape != prediction_xr.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target_xr.values.shape}), "
            f"Prediction ({prediction_xr.values.shape})"
            f"Specify member_name if ensemble prediction)"
        )

    # Identify output dimensions i.e. those not in dim_over dimensions
    output_dims = [d for d in pred_dims if d not in dim_over]

    # --- Special case: output_dims empty, ---
    #  e.g. if prediction dims just [member, time]
    #  => compute jsd over all data
    if not output_dims:
        dist_out=calc_pdf_metric_nparray(target_xr.values, prediction_xr.values,
            axis=None, bins_arr=bins_arr, nbins=nbins, typedistance=typedistance,
            printBins=printBins)
        return xr.Dataset({typedistance: dist_out})

    # Transpose to ensure dim_over dims come first, output_dims last e.g. spatial dims:
    # e.g. [member, time, lat, lon]
    ordered_dims = list(dim_over) + output_dims
    target_xr     = target_xr.transpose(*ordered_dims)
    prediction_xr = prediction_xr.transpose(*ordered_dims)

    # Flatten all dim_over dims into a single axis so can compute over 0th dim
    dimover_shape = target_xr.values.shape[0:len(dim_over)] # e.g. [member, time]
    output_shape = target_xr.values.shape[len(dim_over):]   # e.g. [lat, lon]
    n_total = int(np.prod(dimover_shape))

    target_flat     = target_xr.values.reshape(n_total, *output_shape)
    prediction_flat = prediction_xr.values.reshape(n_total, *output_shape)
    
    dist_out=calc_pdf_metric_nparray(target_flat, prediction_flat, axis=0,
        bins_arr=bins_arr, nbins=nbins, typedistance=typedistance,printBins=printBins)
    
    dist_out_xr = xr.Dataset({typedistance: (output_dims, dist_out)})

    # Restore coordinates for all output dimensions
    for dim in output_dims:
        if dim in prediction_xr.coords:
            dist_out_xr[dim] = prediction_xr[dim]

    return dist_out_xr

