"""
Code to compare probablity distribution functions (pdf) of
lat/lon gridded fields using histogram and qq plots:

Prediction and Target (truth)

    function()
    - description


Copyright (c) 2026 Klima consulting
Author: Rosie Eade
 
"""

from pathlib import Path

import os
import numpy as np
import scipy as sp
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# Plotting Code
# -------------------------------------------------------------------------------------
def plot_1dpdf(
    data_list, 
    ax1, 
    color_list, 
    label_list, 
    bins = None, 
    log=False, 
    density=False,
    InclHist=True, 
    InclKDE=False, 
    KDEapprox=2,
    title=None, 
    xlab=None, 
    ylab=None):

    """
    Plot pdf histogram of input data values for a list of different 
    prediction models in different colours.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    data_list : list of numpy.ndarray
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
    InclHist : boolean
        If True: plot histogram of pdf 
        raise error if InclHist and InclKDE both False
    InclKDE : boolean
        If True: plot kernal density estimate (KDE) of pdf
    KDEapprox : int | None
        Option for different methods to approx KDE
        0 | None : Standard sp.stats.gaussian_kde() # Slow if large data
        1 : As above but use random subset of data # approx
        2 : gaussian_filter1d from scipy.ndimage # Much faster (default)
            i.e. smoothing a histogram rather than computing KDE 
    title : str | None
        Option to define text for title.
    xlab : str | None
        Option to define text for x-axis label
    ylab : str | None
        Option to define text for y-axis label

    Returns:
    --------
    Outputs pdf hist plot to axes object ax1 (element of a subplot grid)
    e.g. as called by plotstats_matrix1_1d_pdf()
    
    """
    
    if InclHist==False and InclKDE==False:
        raise ValueError(f"InclHist and InclKDE false so nothing to plot")     

    LOG_MIN_POS=1E-9
    
    if KDEapprox is None: KDEapprox=0   

    KDEapprox_list=[0, 1, 2]
    if not KDEapprox in KDEapprox_list:
        print(f"KDEapprox not in [0, 1, 2]: {KDEapprox}")
        print(f"Set KDEapprox to be 2")
        KDEapprox=2
    
    len_data_list=len(data_list)

    # ----------------------------------------------------
    # Compute bins for PDF curve if plotted   
    if InclKDE is True:
        x_min_list=[]
        x_max_list=[]
        for dcount in range(len_data_list):
            x_min_list.append(np.nanmin(data_list[dcount]))
            x_max_list.append(np.nanmax(data_list[dcount]))
        x_min=np.nanmin(np.array(x_min_list))
        x_max=np.nanmax(np.array(x_max_list))
        x_vals = np.linspace(x_min, x_max, 1000)
        x_vals = np.linspace(x_min, x_max, 1000)
        #if log==True: x_vals = np.geomspace(np.maximum(x_min,1E-9), x_max, 1000)
        if log==True: x_vals = np.maximum(bins,1E-9)
        density=True # Normalises pdf such that area == 1

    NANexist=False
    for dcount in range(len_data_list):
        ## Reject nonfinite values before plotting
        is_real=np.isfinite(data_list[dcount])
        data_tmp=data_list[dcount][is_real]
        
        if len(is_real[~is_real])>0: NANexist=True
        #Flatten data before plotting
        data_tmp=data_tmp.reshape(np.prod(data_tmp.shape))
        
        # Plot histogram to represent PDF
        alpha_val=0.3
        hist_col=color_list[dcount]
        if InclHist==False: 
            #hist_col='white'
            alpha_val=0.0 # invisible
            hcounts, hbins, hpatches = ax1.hist(data_tmp, bins=bins, 
                edgecolor=hist_col, color=hist_col, log=log, density=density,
                alpha=alpha_val)
        if InclHist==True: 
            hcounts, hbins, hpatches = ax1.hist(data_tmp, bins=bins, 
                edgecolor=hist_col,color=hist_col, log=log, density=density,
                label=label_list[dcount], alpha=alpha_val)
        
        # ------------------------------------------------
        # Option to compute PDF curve using one of these methods
        if InclKDE is True and KDEapprox == 0:
            # Standard scipy method but Very slow if large data
            kde = sp.stats.gaussian_kde(data_tmp)
            pdf_vals = kde(x_vals)
        if InclKDE is True and KDEapprox == 1:
            # Use random subset of data instead of all of the data
            # < 1 sec for ~2e6 data points but tail too smooth
            if data_tmp.size <= 10000: 
                kde = sp.stats.gaussian_kde(data_tmp)
                pdf_vals = kde(x_vals)
            if data_tmp.size > 10000: 
                rng = np.random.default_rng(82)
                tmp_sample = rng.choice(data_tmp, size=10000, replace=False)
                kde = sp.stats.gaussian_kde(tmp_sample) 
                pdf_vals = kde(x_vals)
        if InclKDE is True and KDEapprox == 2:
            # Use gaussian_filter1d from scipy.ndimage instead of sp.stats.gaussian_kde
            # i.e. smoothing a histogram rather than computing KDE
            # < 1 sec for ~2e6 data points and very similar to default method
            bin_edges=hbins
            hist_counts=hcounts
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_centres[1] - bin_centres[0]
            bw = 1.06 * data_tmp.std() * len(data_tmp) ** -0.2   # Scott's rule
            sigma_bins = bw / bin_width                  # bandwidth in grid units
            pdf_vals = sp.ndimage.gaussian_filter1d(hist_counts, sigma=sigma_bins)
            x_vals=bin_centres
        # -----------------

        if InclKDE==True and InclHist==True: 
            print(log)
            ax1.plot(x_vals, pdf_vals, color=color_list[dcount],
                linewidth=2, linestyle='-')

        if InclKDE==True and InclHist==False: 
            print(log)
            ax1.plot(x_vals, pdf_vals, color=color_list[dcount],
                linewidth=2, linestyle='-', label=label_list[dcount])
        # ------------------------------------------------

    # Axis options
    if log==True: ax1.set_yscale("log")

    topDef=None
    botDef=0.0
    if log==True: botDef=LOG_MIN_POS
    ax1.set_ylim(bottom=botDef, top=topDef)
    ax1.set_xlim(left=np.min(bins), right=np.max(bins))

    # Plot 0 lines
    ax1.axvline(0, color ='gray', linestyle='-', linewidth=0.4, alpha = 0.7)

    # Title and legend options
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.legend()
      
    if NANexist:
        print(f"Warning: Nonfinite values removed from input data before plotting")

# -------------------------------------------------------------------------------------
def plot_qq(
    target,
    prediction, 
    ax1, 
    n_quantiles=200,
    title=None, 
    xlab=None, 
    ylab=None,
    axmin=None,
    axmax=None):

    """
    Plot qq plot of input target and prediction data values.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    target : numpy.ndarray
        Data values to be plotted as target (x-axis)
    prediction : numpy.ndarray
        Data values to be plotted as prediction (y-axis)
    ax1 : matplotlib.axes.Axes
        Axes object on which to draw the histogram. Should be an element
        of a subplot grid created via ``plt.subplots()``, e.g.
        ``axes[0, 0]``.
    n_quantiles : int
        Define number of quantiles for plot
    title : str | None
        Option to define text for title.
        e.g. "Q-Q Plot"
    xlab : str | None
        Option to define text for x-axis label
        e.g. "Quantiles of Target"
    ylab : str | None
        Option to define text for y-axis label
        e.g. "Quantiles of Prediction"
    axmin : float | None
        Assign min value for x and y axes or if None use plot default
    axmax : float | None
        Assign max value for x and y axes or if None use plot default
    
    Returns:
    --------
    Outputs qq plot to axes object ax1 (element of a subplot grid)
    e.g. as called by plot_matrix1_qq()
    
    """
    
    nparr_quantiles = np.linspace(0, 100, n_quantiles)

    qtarget = np.nanpercentile(target, nparr_quantiles)
    qprediction = np.nanpercentile(prediction, nparr_quantiles)

    ax1.scatter(qtarget, qprediction, s=12, color='blue', alpha=0.7, zorder=3, label="Quantiles")

    lo = min(qtarget.min(), qprediction.min())
    hi = max(qtarget.max(), qprediction.max())
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="1:1 line")

    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.legend()

    # How best set axes?
    # e.g. include extreme tails or concentrate on centre of distribution?
    ax1.set_ylim(bottom=axmin, top=axmax)
    ax1.set_xlim(left=axmin, right=axmax)

# -------------------------------------------------------------------------------------
def plot_matrix1_1dpdf(
    data_list, 
    color_list, 
    label_list, 
    bins = None, 
    log=False, 
    density=False,
    InclHist=True,
    InclKDE=False,
    KDEapprox=2,
    filepath='figure.png',
    title=None, 
    xlab=None, 
    ylab=None, 
    FIGWIDTH=8, 
    FIGHEIGHT=12):

    """

    Plot pdf histogram plot of input target and prediction data values.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    data_list : list of numpy.ndarray
        Data values to be plotted as pdf hist
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
    InclHist : boolean
        If True: plot histogram of pdf 
        raise error if InclHist and InclKDE both False
    InclKDE : boolean
        If True: plot kernal density estimate (KDE) of pdf
    KDEapprox : int | None
        Option for different methods to approx KDE
        0 | None : Standard sp.stats.gaussian_kde() # Slow if large data
        1 : As above but use random subset of data # approx
        2 : gaussian_filter1d from scipy.ndimage # Much faster (default)
            i.e. smoothing a histogram rather than computing KDE 
    filepath : str
        Matrix of pdfs saved to filepath (including .png or equiv)
    title : str | None
        Option to define text for title.
    xlab : str | None
        Option to define text for x-axis label
    ylab : str | None
        Option to define text for y-axis label
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs pdf histogram plot saved to filepath
    
    """

    # Set up figure (nrows, ncols, ...)
    nrows=1
    fig, axes = plt.subplots(nrows, 1, figsize=(FIGWIDTH, FIGHEIGHT),
        layout='constrained', squeeze=False)

    plot_1dpdf(
        data_list, 
        axes[0,0], 
        color_list, 
        label_list, 
        bins=bins, 
        log=log, 
        density=density,
        InclHist=InclHist, 
        InclKDE=InclKDE, 
        KDEapprox=KDEapprox,
        title=title, 
        xlab=xlab, 
        ylab=ylab)

    plt.savefig(filepath)
    plt.close('all')

# -------------------------------------------------------------------------------------
def plot_matrix1_qq(
    target, 
    prediction, 
    n_quantiles=200,
    filepath='figure.png',
    title=None, 
    xlab=None, 
    ylab=None, 
    axmin=None,
    axmax=None,
    FIGWIDTH=8, 
    FIGHEIGHT=12):

    """

    Plot qq plot of input target and prediction data values.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    target : numpy.ndarray
        Data values to be plotted as target (x-axis)
    prediction : numpy.ndarray
        Data values to be plotted as prediction (y-axis)
    n_quantiles : int
        Define number of quantiles for plot
    filepath : str
        Matrix of pdfs saved to filepath (including .png or equiv)
    title : str | None
        Option to define text for title.
        e.g. "Q-Q Plot"
    xlab : str | None
        Option to define text for x-axis label
        e.g. "Quantiles of Target"
    ylab : str | None
        Option to define text for y-axis label
        e.g. "Quantiles of Prediction"
    axmin : float | None
        Assign min value for x and y axes or if None use plot default
    axmax : float | None
        Assign max value for x and y axes or if None use plot default
    FIGWIDTH : int | float
        Define width of output figure (in inches)
    FIGHEIGHT : int | float
        Define height of output figure (in inches)

    Returns:
    --------
    Outputs qq plot saved to filepath
    
    """

    # Set up figure (nrows, ncols, ...)
    nrows=1
    fig, axes = plt.subplots(nrows, 1, figsize=(FIGWIDTH, FIGHEIGHT),
        layout='constrained', squeeze=False)

    plot_qq(
        target,
        prediction, 
        axes[0,0], 
        n_quantiles=n_quantiles,
        title=title, 
        xlab=xlab, 
        ylab=ylab,
        axmin=axmin,
        axmax=axmax)

    plt.savefig(filepath)
    plt.close('all')