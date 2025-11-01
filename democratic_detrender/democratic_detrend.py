"""
Democratic detrending module for stellar time-series photometry.

This module provides functions for ensemble-based detrending of stellar 
photometry data using multiple detrending methods and rejection criteria.
"""

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import os
import warnings

from democratic_detrender.detrend import detrend_all 
from democratic_detrender.method_reject import method_reject



def democratic_detrend(input_id, mission, flux_type='both', input_planet_number=1, input_dir='./',
    input_depth=0.01, input_period=None, input_t0=None, input_duration=None, input_mask_width=1.1, 
    input_show_plots=False, input_dont_bin=False, input_use_sap_problem_times=False, 
    input_no_pdc_problem_times=True, input_user_light_curve=None,
    input_polyAM=True, input_CoFiAM=True, input_GP=True, input_local=True):
    """
    Apply democratic detrending with method rejection to stellar photometry data.
    
    This is the main function for performing ensemble-based detrending with 
    automated method rejection based on white noise tests. It combines multiple
    detrending methods and rejects poorly performing methods based on statistical
    criteria.
    
    Parameters
    ----------
    input_id : str
        Target identifier (e.g., TIC ID, KIC ID, or TOI designation).
    mission : str
        Space mission name, either 'TESS' or 'Kepler'.
    flux_type : str, optional
        Type of flux to process. Options are 'pdc', 'sap', or 'both'. Default is 'both'.
    input_planet_number : int, optional
        Planet number in the system for multi-planet systems. Default is 1.
    input_dir : str, optional
        Directory to save output files. Default is './'.
    input_depth : float, optional
        Expected transit depth for plotting purposes. Default is 0.01.
    input_period : float, optional
        User-provided orbital period in days. Default is None.
    input_t0 : float, optional
        User-provided transit center time. Default is None.
    input_duration : float, optional
        User-provided transit duration in hours. Default is None.
    input_mask_width : float, optional
        Multiplier for transit mask width. Default is 1.1.
    input_show_plots : bool, optional
        Whether to display diagnostic plots. Default is False.
    input_dont_bin : bool, optional
        Whether to skip binning operations. Default is False.
    input_use_sap_problem_times : bool, optional
        Whether to use SAP problem times for PDC flux. Default is False.
    input_no_pdc_problem_times : bool, optional
        Whether to ignore PDC problem times. Default is True.
    input_user_light_curve : array-like, optional
        User-provided light curve data. Default is None.
    input_polyAM : bool, optional
        Whether to use polynomial AM detrending method. Default is True.
    input_CoFiAM : bool, optional
        Whether to use CoFi AM detrending method. Default is True.
    input_GP : bool, optional
        Whether to use Gaussian Process detrending method. Default is True.
    input_local : bool, optional
        Whether to use local polynomial detrending method. Default is True.
        
    Returns
    -------
    pandas.DataFrame
        Detrended light curve data after method rejection, containing time,
        flux values from different methods, uncertainties, and ensemble result.
    """
    
    
    # determine the path to directory to load and save files
    if input_dir == "./":
        today = date.today()
        current_day = today.strftime("%B_%d_%Y")

        foldername = (
            input_id
            + "/"
            + input_id
            + ".0"
            + str(input_planet_number)
            + "/"
            + "detrending"
            + "/"
            + current_day
        )
        path = os.path.join(input_dir, foldername)

        os.makedirs(path, exist_ok=True)

    else:
        path = input_dir



    df, t0s, period, duration = detrend_all(
    input_id,
    mission,
    flux_type=flux_type,
    input_planet_number=input_planet_number,
    input_dir=input_dir,
    input_depth=input_depth,
    input_period=input_period,
    input_t0=input_t0,
    input_duration=input_duration,
    input_mask_width=input_mask_width,
    input_show_plots=input_show_plots,
    input_dont_bin=input_dont_bin,
    input_use_sap_problem_times=input_use_sap_problem_times,
    input_no_pdc_problem_times=input_no_pdc_problem_times,
    input_user_light_curve=input_user_light_curve,
    input_polyAM=input_polyAM,
    input_CoFiAM=input_CoFiAM,
    input_GP=input_GP,
    input_local=input_local
)




    detrend_df_post_rej = method_reject(path, input_depth, input_period, input_duration, input_mask_width)


    return detrend_df_post_rej



