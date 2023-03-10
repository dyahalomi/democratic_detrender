import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button


def test_DW_per_epoch(x_detrended, y_detrended, yerr_detrended, mask_detrended, mask_fitted_planet_detrended, period, t0s, duration):
    from scipy.special import erfcinv
    
    #break up the epochs...
    x_split, y_split, yerr_split, mask_split, mask_fitted_planet_split = \
    split_around_transits(x_detrended, y_detrended, yerr_detrended, 
                          mask_detrended, mask_fitted_planet_detrended,
                          t0s, float(6*duration/(24.))/period, period)
        
    #if len(x_split) != len(DWs):
    #    print('error mismatching number of epochs with DW array length')
        
    sigmas = []
    for ii in range(0, len(x_split)):
        x = x_split[ii]
        y = y_split[ii]
        yerr = yerr_split[ii]
        mask = mask_split[ii]
        mask_fitted_planet = mask_fitted_planet_split[ii]
        #DWdetrend = DWs[ii]
        
        
        
        ###determine native DW for a single epoch and a single detrended LC
        in_transit = False
        out_transit = True
        
        no_pre_transit = False
        no_post_transit = False
        for index in range(0, len(mask_fitted_planet)):
            mask_val = mask_fitted_planet[index]

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



        if in_transit_index == 0:
            no_pre_transit = True

        if out_transit_index == len(x):
            no_post_transit = True


            
        
        
        
        
        if no_pre_transit:
            DWstat_pre_transit = 2.
            
        else:
            residuals_pre_transit = y[0:in_transit_index]
            if np.all(np.isnan(residuals_pre_transit)):
                DWstat_pre_transit = 2.
            else:
                DWstat_pre_transit = DurbinWatson(residuals_pre_transit)


        if no_post_transit:
            DWstat_post_transit = 2.
        else:
            residuals_post_transit = y[out_transit_index:len(y)]
            if np.all(np.isnan(residuals_post_transit)):
                DWstat_post_transit = 2.
            else:
                DWstat_post_transit = DurbinWatson(residuals_post_transit)
            
            
        DWdetrend = np.sqrt((DWstat_pre_transit-2.)**2. + (DWstat_post_transit-2.)**2.)
        
        
        
        ###create 10,000 white noise profiles and determine DW for epoch
        ###then count the number of these that have greater DW than detrended DW
        n_DWfake_greater_than_DWdetrend = 0
        for jj in range(0, 10000):
                
                
            if no_pre_transit:
                DWstat_fake_pre_transit = 2.
            else:
                fake_residuals_pre_transit = np.random.normal(loc=0.0, 
                                                              scale=yerr[0:in_transit_index], 
                                                              size=np.shape(yerr[0:in_transit_index]))
                DWstat_fake_pre_transit = DurbinWatson(fake_residuals_pre_transit)
                
            
            if no_post_transit:
                DWstat_fake_post_transit = 2.
            else:
                fake_residuals_post_transit = np.random.normal(loc=0.0, 
                                                               scale=yerr[out_transit_index:len(yerr)], 
                                                               size=np.shape(yerr[out_transit_index:len(yerr)]))
                DWstat_fake_post_transit = DurbinWatson(fake_residuals_post_transit)
                
        
            #determine DW for white noise data
            DWfake = np.sqrt((DWstat_fake_pre_transit-2.)**2. + (DWstat_fake_post_transit-2.)**2.)
            
            
            if DWfake > DWdetrend:
                n_DWfake_greater_than_DWdetrend += 1
                
        
        
        #calculate and save sigma value for each epoch
        sigma = np.sqrt(2.) * erfcinv(n_DWfake_greater_than_DWdetrend/10000.)
        sigmas.append(sigma)
        
        
    
    return sigmas
                
                
                





def reject_epochs_by_DW(x_detrended, y_detrended, yerr_detrended, mask_detrended, mask_fitted_planet_detrended, sigma_values, t0s):
    #break up the epochs...
    x_split, y_split, yerr_split, mask_split, mask_fitted_planet_split = \
    split_around_transits(x_detrended, y_detrended, yerr_detrended, 
                          mask_detrended, mask_fitted_planet_detrended,
                          t0s, float(6*duration/(24.))/period, period)
    
    
    if len(x_split) == len(sigma_values):
        epochs = len(sigma_values)
    else:
        print("mistmatched array lengths!")
        print(len(x_split))
        print(len(sigma_values))
        return(None)
    
    y_detrended_out = []
    for ii in range(0, epochs):
        sigma = sigma_values[ii]
        if sigma > 2.:
            nan_array = np.empty(np.shape(y_split[ii]))
            y_split[ii][:] = np.nan
            
        y_detrended_out.extend(y_split[ii])
    
    return(np.array(y_detrended_out))
            
        

