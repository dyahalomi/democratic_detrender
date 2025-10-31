import os
import warnings
import numpy as np
import pandas as pd

from democratic_detrender.plot import plot_detrended_lc
from democratic_detrender.method_rejection_functions_dw import reject_via_DW, dw_rejection_plots
from democratic_detrender.method_rejection_functions_binning import reject_via_binning, binning_rejection_plots
from democratic_detrender.method_rejection_functions_general import ensemble_step, merge_epochs, reject_epochs_by_white_noise_tests




def method_reject(path, detrended_lc_file, orbital_data_file, t0s_file,
    input_depth=0.01, input_period=None, input_t0=None, input_duration=None, input_mask_width=1.1, 
    input_show_plots=False, input_dont_bin=False, input_use_sap_problem_times=False, 
    input_no_pdc_problem_times=True, input_user_light_curve=None,
    input_polyAM=True, input_CoFiAM=True, input_GP=True, input_local=True):
    
    

    # Load the file into a Pandas DataFrame, skipping the first column
    df = pd.read_csv(path+detrended_lc_file)


    #df, t0s, period, duration  = detrend_all(input_id, mission, flux_type, input_planet_number, input_dir,
    #input_depth, input_period, input_t0, input_duration, input_mask_width, 
    #input_show_plots, input_dont_bin, input_use_sap_problem_times, 
    #input_no_pdc_problem_times, input_user_light_curve,
    #input_polyAM, input_CoFiAM, input_GP, input_local)

    orbital_data = pd.read_csv(path+orbital_data_file)
    t0s = pd.read_csv(path+t0s_file)

    period = orbital_data['period']
    duration = orbital_data['duration']

    t0s = list(pd.read_csv(path+t0s_file)['t0s_in_data'])

    # Fixed columns
    fixed_cols = ['time', 'yerr', 'mask', 'method marginalized']

    # Variable columns (everything else)
    detrending_methods = [col for col in df.columns if col not in fixed_cols]

    print("detrending methods used:", detrending_methods)


    # Initialize sublists
    time_epochs = []
    y_epochs = []
    yerr_epochs = []

    # Temporary variables to hold data for current sublist
    time_temp = []
    y_temp = []
    yerr_temp = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if len(time_temp) == 0:  # If it's the first data point
            # Check if all values in the specified columns are not NaN for the current row
            #if row[['local SAP', 'local PDCSAP', 
            #        'polyAM SAP', 'polyAM PDCSAP', 
            #        'GP SAP', 'GP PDCSAP', 
            #        'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                
            time_temp.append(row['time'])
            y_temp.append(row[detrending_methods])
            yerr_temp.append(row['yerr'])
        else:
            time_diff = row['time'] - time_temp[-1]
            if time_diff > 5:  # If there is a gap greater than 5 in time
                # Check if all values in the specified columns are not NaN for the current row
                #if row[['local SAP', 'local PDCSAP', 
                #        'polyAM SAP', 'polyAM PDCSAP', 
                #        'GP SAP', 'GP PDCSAP', 
                #        'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                # Append current sublist to the main list
                time_epochs.append(time_temp)
                y_epochs.append(pd.DataFrame(y_temp))
                yerr_epochs.append(yerr_temp)
                # Reset temporary variables for the new sublist
                time_temp = [row['time']]
                y_temp = [row[detrending_methods]]
                yerr_temp = [row['yerr']]
            else:
                # Check if all values in the specified columns are not NaN for the current row
                #if row[['local SAP', 'local PDCSAP', 
                #        'polyAM SAP', 'polyAM PDCSAP', 
                #        'GP SAP', 'GP PDCSAP', 
                #        'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                time_temp.append(row['time'])
                y_temp.append(row[detrending_methods])
                yerr_temp.append(row['yerr'])

    # Append the last sublist
    time_epochs.append(time_temp)
    y_epochs.append(pd.DataFrame(y_temp))
    yerr_epochs.append(yerr_temp)

    period = period[0]
    duration=input_mask_width*duration[0]/24.


    # START OF METHOD REJECTION TESTS!!!!
    method_reject_figpath = path + "/" + "method_rejection_figures/"
    os.makedirs(method_reject_figpath, exist_ok=True)

    # DW method rejection test
    dw_sigma_test, DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs = reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000)
    dw_rejection_plots(DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs, detrending_methods, method_reject_figpath)


    # binning vs. RMS method rejection test
    binning_sigma_test, beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs = reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000)
    binning_rejection_plots(beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs, detrending_methods, method_reject_figpath)


    # method rejection step
    y_epochs_post_rej = reject_epochs_by_white_noise_tests(y_epochs, dw_sigma_test, binning_sigma_test, detrending_methods)
    times_all_post_rej, y_all_post_rej, yerr_all_post_rej = merge_epochs(time_epochs, y_epochs_post_rej, yerr_epochs)
    detrend_df_post_rej = ensemble_step(times_all_post_rej, y_all_post_rej, yerr_all_post_rej, detrending_methods, df['mask'])


    ## now to plot and save data!!
    green2, green1 = '#355E3B', '#18A558'
    blue2, blue1 = '#000080', '#4682B4'
    purple2, purple1 = '#2E0854','#9370DB'
    red2, red1 = '#770737', '#EC8B80'


    colors = [red1, red2,
              blue1, blue2,
              green1, green2,
              purple1, purple2]

        
    # plot all detrended data
    plot_detrended_lc(times_all_post_rej, y_all_post_rej, detrending_methods,
                      t0s, float(6*duration)/period/input_mask_width, period,
                      colors, duration*24., depth=0.01, mask_width=1,
                      figname = path+'/individual_detrended_post_rejection.pdf')

    # plot method marginalized detrended data
    plot_detrended_lc(
        times_all_post_rej,
        [detrend_df_post_rej["method marginalized"]],
        ["method marg"],
        t0s,
        float(6*duration)/period/input_mask_width, 
        period,
        ["k"], 
        duration*24., depth=0.01, mask_width=1,
        figname=path + "/" + "method_marg_detrended_post_rejection.pdf"
            )


    #save post method rejection as csv
    detrend_df_post_rej.to_csv(path + "/" + "detrended_post_method_rejection.csv", index=False)

    return detrend_df_post_rej


