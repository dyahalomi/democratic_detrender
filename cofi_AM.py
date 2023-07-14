
### Special thanks to Alex Teachey --> adapted from MoonPy package
### GitHub: https://github.com/alexteachey/MoonPy

"""
we need to solve the problem
AX = B
where A is a vector of coefficients for our linear problem
X is a matrix of terms, multiplying those values by the coefficients in A will give us
B the function values.
NOTE THAT THE COFIAM ALGORITHM FITS TERMS AS FOLLOWS
offset + (amp_s1 * (sin(2pi * time * 1) / (2 * baseline)) + amp_c1 * (cos(2*pi*time * 1) / 2*baseline) + ... up to the degree in question.
NOW FOR THE MATRIX REPRESENTATION, YOU NEED TO DO THIS FOR EVERY TIMESTEP! The matrix rows are for each time in your array!
"""
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button
from manipulate_data import split_around_transits
from helper_functions import *
from plot import *
from poly_AM import *


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



def cofiam_matrix_gen(times, degree):
    baseline = np.nanmax(times) - np.nanmin(times)
    assert baseline > 0
    rows = len(times)
    cols = 2 * (degree+1)
    X_matrix = np.ones(shape=(rows,cols))
    for x in range(rows):
        for y in range(1, int(cols/2)):
            sinarg = (2*np.pi*times[x] * y) / baseline
            X_matrix[x,y*2] = np.sin(sinarg)
            X_matrix[x,y*2 + 1] = np.cos(sinarg)
        X_matrix[x,1] = times[x]
    return X_matrix 


def cofiam_matrix_coeffs(times, fluxes, degree):
    assert len(times) > 0
    Xmat = cofiam_matrix_gen(times, degree)
    beta_coefs = np.linalg.lstsq(Xmat, fluxes, rcond=None)[0]
    return Xmat, beta_coefs



### this function spits out the best fit line!
def cofiam_function(times, fluxes, degree):
    input_times = times.astype('f8')
    input_fluxes = fluxes.astype('f8')
    cofiam_matrix, cofiam_coefficients = cofiam_matrix_coeffs(input_times, input_fluxes, degree)
    output = np.matmul(cofiam_matrix, cofiam_coefficients)
    return output 


def cofiam_iterative(times, fluxes, mask, mask_fitted_planet,
                     local_start_x, local_end_x, 
                     max_degree=30, min_degree=1):
    
    ### this function utilizes cofiam_function above, iterates it up to max_degree.
    no_pre_transit = False
    no_post_transit = False
    
    vals_to_minimize = []
    models = []
    degs_to_try = np.arange(min_degree, max_degree+1, 1)
    DWstats = []
    
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
        model = cofiam_function(times[~mask], fluxes[~mask], deg)
        if no_pre_transit:
            DWstat_pre_transit = 2.
        else:
            local_start_index = np.where(times == local_start_x)[0][0]
            residuals_pre_transit = (((fluxes[local_start_index:in_transit_index] + 1) / \
                                    (model[local_start_index:in_transit_index] + 1)) - 1)
            DWstat_pre_transit = DurbinWatson(residuals_pre_transit)

        if no_post_transit:
            DWstat_post_transit = 2.
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










def cofiam_method(x, y, yerr, mask, mask_fitted_planet, t0s, duration, period, local_x):
    
    
    
    from scipy.stats import median_absolute_deviation
    
    cofiam_mod = []
    cofiam_mod_all = []
    
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
            cofiam = cofiam_iterative(x_ii, y_ii, mask_ii, mask_fitted_planet_ii,
                                      local_start_x_ii, local_end_x_ii, max_degree=30)

            
            cofiam_interp = interp1d(x_ii[~mask_ii], cofiam[0], bounds_error=False, fill_value='extrapolate')
            best_model = cofiam_interp(x_ii)
            DWs.append(cofiam[2])
            

            
            cofiam_mod.append(best_model)
            cofiam_mod_all.extend(best_model)
            
            
        
        except:
            print('CoFiAM failed for the ' + str(ii) + 'th epoch')
            #gp failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            cofiam_mod.append(nan_array)
            cofiam_mod_all.extend(nan_array)



        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_ii)
            

    
    #zoom into local region of each transit
    x_out, y_out, yerr_out, \
    mask_out, mask_fitted_planet_out, model_out = split_around_transits(np.array(x_all), 
                                                                        np.array(y_all), 
                                                                        np.array(yerr_all), 
                                                                        np.array(mask_all),
                                                                        np.array(mask_fitted_planet_all),
                                                                        t0s, float(6*duration/(24.))/period, 
                                                                        period, model=np.array(cofiam_mod_all))
    
    
    
    

    #add a linear polynomial fit at the end
    model_linear = []
    y_out_detrended = []
    for ii in range(0, len(model_out)):
        x_ii = np.array(x_out[ii], dtype=float)
        y_ii = np.array(y_out[ii], dtype=float)
        mask_ii = np.array(mask_out[ii], dtype=bool)
        model_ii = np.array(model_out[ii], dtype=float)
        
        
        try:
            y_ii_detrended = get_detrended_lc(y_ii, model_ii)
            
            linear_ii = polyAM_function(x_ii[~mask_ii], y_ii_detrended[~mask_ii], 1)
            poly_interp = interp1d(x_ii[~mask_ii], linear_ii, bounds_error=False, fill_value='extrapolate')
            model_ii_linear = poly_interp(x_ii)
            
            model_linear.append(model_ii_linear)
            
            y_ii_linear_detrended = get_detrended_lc(y_ii_detrended, model_ii_linear)
            y_out_detrended.append(y_ii_linear_detrended)

        except:
            print('CofiAM failed for the ' + str(ii) + 'th epoch')
            #CofiAM failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)
        
    
    
    detrended_lc = np.concatenate(y_out_detrended, axis=0)
    
    return detrended_lc, DWs
