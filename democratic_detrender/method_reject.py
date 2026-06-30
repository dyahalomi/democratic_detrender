import os
import warnings
import numpy as np
import pandas as pd

from democratic_detrender.plot import plot_detrended_lc
from democratic_detrender.method_rejection_functions_dw import reject_via_DW, dw_rejection_plots
from democratic_detrender.method_rejection_functions_binning import reject_via_binning, binning_rejection_plots
from democratic_detrender.method_rejection_functions_general import ensemble_step, merge_epochs, reject_epochs_by_white_noise_tests




def method_reject(path, input_depth=0.01, input_period=None, input_duration=None, input_mask_width=1.1):
    
    

    # Load the detrending output produced before method rejection.
    pre_rejection_path = path + '/detrended_pre_rejection.csv'
    df = pd.read_csv(pre_rejection_path)

    orbital_data = pd.read_csv(path+'/orbital_data.csv')

    if input_period == None:
        period = orbital_data['period']
    
    if input_duration == None:
        duration = orbital_data['duration']

    t0s = list(pd.read_csv(path+'/t0s.csv')['t0s_in_data'])

    # Fixed columns
    fixed_cols = ['time', 'yerr', 'mask', 'method marginalized']

    # Variable columns (everything else)
    detrending_methods = [col for col in df.columns if col not in fixed_cols]

    print("detrending methods used:", detrending_methods)


    # Previous: >-5-day time gaps, splitted chunks but sometimes short period planets can have many transits per chunk
    # transit per data chunk when transits are >5 days apart (long-period planets)
    # cutting chunk based on midpoint of transit midtimes

    t0s_arr = np.sort(np.array(t0s, dtype=float))
    times_all = df['time'].to_numpy(dtype=float)
    edges = (t0s_arr[:-1] + t0s_arr[1:]) / 2.0       # midpoints = chunk boundaries
    nearest = np.searchsorted(edges, times_all)      # nearest-transit index  (no N x M array for memory saving)

    time_epochs = []
    y_epochs = []
    yerr_epochs = []
    t0s_used = []                                    # transit midtimes kept aligned to the epochs
    for k in range(len(t0s_arr)):
        sel = np.where(nearest == k)[0]
        if len(sel) == 0:                           
            continue
        time_epochs.append(list(times_all[sel]))
        y_epochs.append(df[detrending_methods].iloc[sel].reset_index(drop=True))
        yerr_epochs.append(list(df['yerr'].to_numpy()[sel]))
        t0s_used.append(t0s_arr[k])
    pd.DataFrame({'t0s_used': t0s_used}).to_csv(path + '/final_t0s.csv', index=False)

    period = period[0]
    duration=input_mask_width*duration[0]/24.


    # START OF METHOD REJECTION TESTS!!!!
    method_reject_figpath = path + "/" + "method_rejection_figures/"
    os.makedirs(method_reject_figpath, exist_ok=True)

    # DW method rejection test
    dw_sigma_test, DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs = reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s_used, period, duration, niter=100000)
    dw_rejection_plots(DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs, detrending_methods, method_reject_figpath)


    # binning vs. RMS method rejection test
    binning_sigma_test, beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs = reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s_used, period, duration, niter=100000)
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
    # mask width already inflated in method rejection step so mask_width=1
    plot_detrended_lc(times_all_post_rej, y_all_post_rej, detrending_methods,
                      t0s_used, float(6*duration)/period/input_mask_width, period,
                      colors, duration*24., depth=input_depth, mask_width=1,
                      figname = path+'/individual_detrended_post_rejection.pdf')

    # plot method marginalized detrended data
    # mask width already inflated in method rejection step so mask_width=1
    plot_detrended_lc(
        times_all_post_rej,
        [detrend_df_post_rej["method marginalized"]],
        ["method marg"],
        t0s_used,
        float(6*duration)/period/input_mask_width, 
        period,
        ["k"], 
        duration*24., depth=input_depth, mask_width=1,
        figname=path + "/" + "method_marg_detrended_post_rejection.pdf"
            )


    #save post method rejection as csv
    detrend_df_post_rej.to_csv(
        path + "/" + "detrended_post_rejection.csv", index=False
    )

    return detrend_df_post_rej
