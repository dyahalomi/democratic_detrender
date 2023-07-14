import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button


def get_detrended_lc(y, detrending_model):
    '''
    input:
    y = light curve
    detrending model = stellar detrending model evaluated at same time as fluxes
    
    returns:
    detrended_lc = detrended light curve evaluated at same time as fluxes
    
    '''
    detrended_lc = (((y + 1) / (detrending_model + 1)) - 1)
    
    return np.array(detrended_lc)





def determine_cadence(times):
    time_gaps = {}
    for ii in range(1, len(times)):
        time_gap = np.round(times[ii]-times[ii-1], 4)
        if time_gap in time_gaps.keys():
            time_gaps[time_gap] += 1
        else:
            time_gaps[time_gap] = 1
            
    #find the key that corresponds to the most data gaps, this is the cadence
    cadence = max(time_gaps, key=time_gaps.get)
    return cadence
            




def find_nearest(array, value):
    #returns the value in an array closest to another input value
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]




def bin_data(xs, ys, window):
    
    import warnings


    xmin = np.min(xs)
    xmax = np.max(xs)
    
    x_bin = np.arange(xmin-window, xmax+window, window)
    
    y_bin = []
    for ii in range(0, len(x_bin)):
        y_bin.append([])
        
    for jj in range(1, len(x_bin)):
        x_bin_jj = x_bin[jj]
        x_bin_jj_minus_1 = x_bin[jj-1]
        
            
            
    for ii in range(0, len(xs)):
        found_bin = False
        x_ii = xs[ii]
        y_ii = ys[ii]
        
        for jj in range(1, len(x_bin)):
            x_bin_jj = x_bin[jj]
            x_bin_jj_minus_1 = x_bin[jj-1]
            
            if not found_bin:
                if x_bin_jj_minus_1 <= x_ii <= x_bin_jj:
                    found_bin = True
                    y_bin[jj].append(y_ii)
                
    
        if not found_bin:
            print("careful, the time " + str(x_ii) + " didn't find a bin!")
            
    
    y_bin_mean = []
    for ii in range(0, len(y_bin)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_bin_mean.append(np.nanmean(np.array(y_bin[ii])))
        
    
    
    return x_bin, y_bin_mean




