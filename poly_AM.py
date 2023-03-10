
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button
from helper_functions import *
from manipulate_data import split_around_transits

def DurbinWatson(residuals):
    residual_terms = []
    for nres, res in enumerate(residuals):
        try:
            residual_terms.append(residuals[nres+1] - residuals[nres])
        except:
            pass
    residual_terms = np.array(residual_terms)
    numerator = np.nansum(residual_terms**2)
    denominator = np.nansum(residuals**2)
    
    assert denominator != 0.
    return numerator / denominator






def polyAM_function(times, fluxes, degree):
    #print(times)
    #print(type(times))
    poly_coeffs = np.polyfit(times, fluxes, degree)
    model = np.polyval(poly_coeffs, times)
    return model





def polyAM_iterative(times, fluxes, mask, mask_fitted_planet, 
                     local_start_x, local_end_x, 
                     max_degree=20, min_degree=1):
    ### this function utilizes polyAM_function above, iterates it up to max_degree.
    no_pre_transit = False
    no_post_transit = False
    
    vals_to_minimize = []
    degs_to_try = np.arange(min_degree, max_degree+1,1)
    DWstats = []
    models = []
    
    in_transit = False
    out_transit = True
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


    #this was added in to solve the case of only an ingress      
    try:
        out_transit_index
    except NameError:
        out_transit_index = len(times)



    if in_transit_index == 0:
        no_pre_transit = True
    
    if out_transit_index == len(times):
        no_post_transit = True
    
    for deg in degs_to_try:
        model = polyAM_function(times[~mask], fluxes[~mask], deg)
        if no_pre_transit:
            DWstat_pre_transit = 2.
            local_start_index = 0 #just for plotting
        else:
            local_start_index = np.where(times == local_start_x)[0][0]
            residuals_pre_transit = (((fluxes[local_start_index:in_transit_index] + 1) / \
                                    (model[local_start_index:in_transit_index] + 1)) - 1)
            DWstat_pre_transit = DurbinWatson(residuals_pre_transit)

        
        if no_post_transit:
            DWstat_post_transit = 2.
            local_end_index = len(times)-1 #just for plotting
        else:
            local_end_index = np.where(times == local_end_x)[0][0]
            npoints_missing_from_model = out_transit_index - in_transit_index
            residuals_post_transit = (((fluxes[out_transit_index:local_end_index] + 1) / \
                                     (model[out_transit_index-npoints_missing_from_model:local_end_index-npoints_missing_from_model] + 1)) - 1)
            
            DWstat_post_transit = DurbinWatson(residuals_post_transit)
        
        val_to_minimize = np.sqrt((DWstat_pre_transit-2.)**2. + (DWstat_post_transit-2.)**2.)
        vals_to_minimize.append(val_to_minimize)
        
        models.append(model)
        
        


    best_degree = degs_to_try[np.argmin(np.array(vals_to_minimize))]
    best_DW_val = vals_to_minimize[np.argmin(np.array(vals_to_minimize))]
    best_model = models[np.argmin(np.array(vals_to_minimize))]
    

    
    
    return best_model, best_degree, best_DW_val, max_degree 












def polynomial_method(x, y, yerr, mask, mask_fitted_planet, t0s, duration, period, local_x):

    
        
    poly_mod = []
    poly_mod_all = []
    
    x_all = []
    y_all = []
    yerr_all = []
    mask_all = []
    mask_fitted_planet_all = []
    DWs = []
    
    for ii in range(0, len(x)):
        x_ii = x[ii]
        y_ii = y[ii]
        yerr_ii = yerr[ii]
        mask_ii = mask[ii]
        mask_fitted_planet_ii = mask_fitted_planet[ii]
        
        local_start_x_ii = local_x[ii][0]
        local_end_x_ii = local_x[ii][len(local_x[ii])-1]
        
        

        try:
            poly = polyAM_iterative(x_ii, y_ii, mask_ii, mask_fitted_planet_ii,
                                    local_start_x_ii, local_end_x_ii, max_degree=30)

            
            poly_interp = interp1d(x_ii[~mask_ii], poly[0], bounds_error=False, fill_value='extrapolate')
            best_model = poly_interp(x_ii)
            DWs.append(poly[2])
            
            
            poly_mod.append(best_model)
            poly_mod_all.extend(best_model)
            


        except:
            print('polyAM failed for the ' + str(ii) + 'th epoch')
            #gp failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            poly_mod.append(nan_array)
            poly_mod_all.extend(nan_array)




        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_ii)
            
            
    
    #zoom into local window
    x_out, y_out, yerr_out, \
    mask_out, mask_fitted_planet_out, model_out = split_around_transits(np.array(x_all), 
                                                                        np.array(y_all), 
                                                                        np.array(yerr_all), 
                                                                        np.array(mask_all),
                                                                        np.array(mask_fitted_planet_all),
                                                                        t0s, float(6*duration/(24.))/period, 
                                                                        period, model=np.array(poly_mod_all))
    

    #add a linear polynomial fit at the end
    model_linear = []
    y_out_detrended = []
    x_out_detrended = []
    for ii in range(0, len(model_out)):
        x_ii = np.array(x_out[ii], dtype=float)
        y_ii = np.array(y_out[ii], dtype=float)
        mask_ii = np.array(mask_out[ii], dtype=bool)
        model_ii = np.array(model_out[ii], dtype=float)
        
        
        y_ii_detrended = get_detrended_lc(y_ii, model_ii)
        


        try:
            linear_ii = polyAM_function(x_ii[~mask_ii], y_ii_detrended[~mask_ii], 1)
            poly_interp = interp1d(x_ii[~mask_ii], linear_ii, bounds_error=False, fill_value='extrapolate')
            model_ii_linear = poly_interp(x_ii)
            
            model_linear.append(model_ii_linear)
            
            y_ii_linear_detrended = get_detrended_lc(y_ii_detrended, model_ii_linear)
            y_out_detrended.append(y_ii_linear_detrended)
            x_out_detrended.append(x_ii)

        except:
            print('polyAM failed for the ' + str(ii) + 'th epoch')
            #polyAM failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)
        

        
        
        
    
    detrended_lc = np.concatenate(y_out_detrended, axis=0)
    
    return detrended_lc, DWs