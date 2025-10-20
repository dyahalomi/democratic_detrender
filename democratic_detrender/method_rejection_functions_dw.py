from democratic_detrender.helper_functions import durbin_watson
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import erf

def DW_monte_carlo(x, yerr, t0, duration, in_transit_index, out_transit_index, niter=10000):
    
    if in_transit_index == 0:
        no_pre_transit = True
    else:
        no_pre_transit = False
        
    
    if out_transit_index == len(x):
        no_post_transit = True
    else:
        no_post_transit = False
        
        
    ###create 10,000 white noise profiles and determine DW for epoch
    ###then count the number of these that have greater DW than detrended DW
    DWMC = []
    for jj in range(0, niter):


        if no_pre_transit:
            DWstat_fake_pre_transit = 2.
        else:
            fake_residuals_pre_transit = np.random.normal(loc=0.0, 
                                                          scale=yerr[0:in_transit_index], 
                                                          size=np.shape(yerr[0:in_transit_index]))
            DWstat_fake_pre_transit = durbin_watson(fake_residuals_pre_transit)

        
        if no_post_transit:
            DWstat_fake_post_transit = 2.
        else:
            fake_residuals_post_transit = np.random.normal(loc=0.0, 
                                                           scale=yerr[out_transit_index:len(yerr)], 
                                                           size=np.shape(yerr[out_transit_index:len(yerr)]))
            DWstat_fake_post_transit = durbin_watson(fake_residuals_post_transit)

        
        #determine DW for white noise data
        DWfake = np.sqrt((DWstat_fake_pre_transit-2.)**2. + (DWstat_fake_post_transit-2.)**2.)
        
        DWMC.append(DWfake)
        


    return DWMC
        
    
def determine_transit_indices(x, t0, duration):        
    # Calculate the start and end times of the transit window
    transit_start = t0 - duration / 2.0
    transit_end = t0 + duration / 2.0
    
    # Create a boolean mask for the transit region
    mask = (x >= transit_start) & (x <= transit_end)    
        
    # determine native DW for a single epoch and a single detrended LC
    # first, determine the index where transits start and stop
    in_transit = False
    in_transit_index = 0
    out_transit = True
    out_transit_index = len(x)

    
    for index in range(0, len(mask)):
        mask_val = mask[index]

        if out_transit:
            if mask_val:
                in_transit_index = index

                in_transit = True
                out_transit = False

        if in_transit:
            if not mask_val:
                out_transit_index = index

                in_transit = False
                out_transit = True
                
                
    return in_transit_index, out_transit_index
                
                
def DW_outlier_detection(x, y, yerr, t0, in_transit_index, out_transit_index, DWMC, niter=10000, max_sigma=3):
    """
    test how much the DW statistic for a detrended LC is an outlier (in terms of sigma) 
    for a single epoch LC in time series data.

    Parameters:
    - x (array): Time values.
    - y (array): Detrended flux values assuming centered on zero.
    - yerr (array): Flux error values.
    - t0 (float): Midtransit time.

    Returns:
    - sigma (float): what sigma value is the DW for a given epoch vs. white noise
    """
   
    if in_transit_index == 0:
        no_pre_transit = True
    else:
        no_pre_transit = False
        
    
    if out_transit_index == len(x):
        no_post_transit = True
    else:
        no_post_transit = False

    # if the index where transit starts is first index, then there is no pre-transit data
    # therefore pre-transit DW stat equals 2
    if no_pre_transit:
        DWstat_pre_transit = 2.
    
    
    # else determine the DW statistic of the pre-transit data
    else:
        residuals_pre_transit = y[0:in_transit_index]
        if np.all(np.isnan(residuals_pre_transit)):
            DWstat_pre_transit = 2.
        else:
            DWstat_pre_transit = durbin_watson(residuals_pre_transit)
            
    
            
            
    
    # if the index where transit ends is last index, then there is no post-transit data
    # therefore post-transit DW stat equals 2
    if no_post_transit:
        DWstat_post_transit = 2.
    
    # else determine the DW statistic of the post-transit data
    else:
        residuals_post_transit = y[out_transit_index:len(y)]
        if np.all(np.isnan(residuals_post_transit)):
            DWstat_post_transit = 2.
        else:
            DWstat_post_transit = durbin_watson(residuals_post_transit)
            
            




    # combine pre and post transit data to determine the DW statistic of detrended data
    DWdetrend = np.sqrt((DWstat_pre_transit-2.)**2. + (DWstat_post_transit-2.)**2.)

    # Calculate the error function term
    # determine how far from 50th percentile we allow for based on max sigma
    erf_term = 0.5 * erf(max_sigma / np.sqrt(2))

    # Calculate the quantile bounds
    #lower_bound = np.quantile(DWMC, 0.5 - erf_term)
    upper_bound = np.quantile(DWMC, 0.5 + erf_term)

    # Test whether DWdetrend is within the bounds
    is_within_bounds = DWdetrend <= upper_bound


        
    
    return is_within_bounds, DWdetrend, upper_bound
            
                
                



def reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=10000):
    
    print('')
    print("starting individual model rejection via DW metric:")
    print("--------------------------------------------------")
    start_time = time.time()

    DWMC_epochs = []
    in_transit_index_epochs = []
    out_transit_index_epochs = []
    for ii, file in enumerate(time_epochs):
        in_transit_index, out_transit_index = determine_transit_indices(np.array(time_epochs[ii]), t0s[ii], duration*1.1)       
        in_transit_index_epochs.append(in_transit_index)
        out_transit_index_epochs.append(out_transit_index)

        DWMC = DW_monte_carlo(np.array(time_epochs[ii]), np.array(yerr_epochs[ii]), t0s[ii], duration, in_transit_index, out_transit_index, niter=10000)
        DWMC_epochs.append(DWMC)




    is_within_bounds_epochs = [] 
    DWdetrend_epochs = []
    #lower_bound_epochs = []
    upper_bound_epochs = []
    for ii, file in enumerate(time_epochs):
        is_within_bounds_methods = [] 
        DWdetrend_methods = []
        #lower_bound_methods = []
        upper_bound_methods = []
        for jj, col in enumerate(y_epochs[ii].columns):
            DW_outlier_output = DW_outlier_detection(np.array(time_epochs[ii]), 
                                                      np.array(y_epochs[ii][col]),
                                                      np.array(yerr_epochs[ii]), 
                                                      t0s[ii],
                                                      in_transit_index_epochs[ii], out_transit_index_epochs[ii],
                                                      DWMC_epochs[ii], 
                                                      niter=niter,
                                                      max_sigma=3
                                                     )

            is_within_bounds_methods.append(DW_outlier_output[0]) 
            DWdetrend_methods.append(DW_outlier_output[1]) 
            upper_bound_methods = DW_outlier_output[2] #this is the same for each method, only varies per epoch

        is_within_bounds_epochs.append(is_within_bounds_methods) 
        DWdetrend_epochs.append(DWdetrend_methods) 
        upper_bound_epochs.append(upper_bound_methods) 



    end_time = time.time()

    # Calculate the elapsed time
    execution_time = end_time - start_time
    print("DW rejection completed in :", execution_time, "seconds")
    print('')


    columns = y_epochs[ii].columns
    sigma_test= pd.DataFrame(is_within_bounds_epochs, columns=columns)
    
    return sigma_test, DWMC_epochs, DWdetrend_epochs, upper_bound_epochs




def dw_rejection_plots(DWMC_epochs, DWdetrend_epochs, upper_bound_epochs, columns, figpath):
        
    green2, green1 = '#355E3B', '#18A558'
    blue2, blue1 = '#000080', '#4682B4'
    purple2, purple1 = '#2E0854','#9370DB'
    red2, red1 = '#770737', '#EC8B80'

    colors = [red1, red2,
              blue1, blue2,
              green1, green2,
              purple1, purple2]


    for ii in range(0, len(DWMC_epochs)):
        plt.figure(figsize=[9,7])
        plt.hist(DWMC_epochs[ii], bins=100, density=True, histtype='step', color='k', linewidth=1, alpha=.7, label = r'White Noise Monte Carlo $\overline{\mathrm{DW}}$')

        for jj in range(0, len(DWdetrend_epochs[ii])):
            #if sigma_test[columns[jj]][ii]:
            plt.axvline(DWdetrend_epochs[ii][jj], color=colors[jj], lw=3, label=columns[jj]+ r' $\overline{\mathrm{DW}}$ = ' + str(np.round(DWdetrend_epochs[ii][jj], 2)))
            #else:
            #    plt.axvline(DWdetrend_epochs[ii][jj], color=colors[jj], lw=3, label=columns[jj]+' DW = ' + str(np.round(DWdetrend_epochs[ii][jj], 2)), ls='dotted')

        # Plot vertical lines at median, median ± 1 std dev, and median ± 2 std dev
        # plt.axvspan(0., upper_bound_epochs[ii], color='g', alpha=0.3)
        plt.axvspan(upper_bound_epochs[ii], 1.5*np.max(DWMC_epochs[ii]), color='r', alpha=0.1)
        #plt.axvline(upper_bound_epochs[ii], color='r', lw=3, ls='--', label=r'2$\sigma$')
        plt.text(1.5*np.max(DWMC_epochs[ii])-.1, 0.5, r'$\overline{\mathrm{DW}}$ $>$ 3$\sigma$ outlier', 
                 verticalalignment='bottom', horizontalalignment='right', fontsize=27, color='r')

        plt.title(r'Bootstrap MC $\overline{\mathrm{DW}}$ Distribution vs. Detrended $\overline{\mathrm{DW}}$s, epoch \#'+str(ii+1), fontsize=23)
        plt.xlabel(r'$\overline{\mathrm{DW}}$ Statistic of Light Curve', fontsize=23)
        plt.ylabel('Density', fontsize=23)
        plt.legend(fontsize=13, loc=1)

        plt.xlim(np.min(DWMC_epochs[ii]), 1.5*np.max(DWMC_epochs[ii]))


        plt.tight_layout()

        plt.savefig(figpath + 'reject_via_dw_epoch'+str(ii+1)+'.pdf')

    return None

