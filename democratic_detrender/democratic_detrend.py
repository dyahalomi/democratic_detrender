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



