import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button


#print(f"exoplanet.__version__ = '{xo.__version__}'")

from get_lc import *
from helper_functions import *
from outlier_rejection import *
from manipulate_data import *
from plot import *



def find_flux_jumps(star_id, flux_type, save_to_directory, show_plots, TESS = False, Kepler = False, 
                    user_periods = None, user_t0s = None, user_durations = None,
                    planet_number = 1, mask_width = 1.3, no_pdc_problem_times = True, dont_bin = False, 
                    data_name = None, problem_times_default = None):



    #pulls in light curve
    time, lc, lc_err, mask, mask_fitted_planet, \
    t0s, period, duration, quarters, crowding, flux_fraction  = \
    get_light_curve(star_id, flux_type, TESS, Kepler, 
                    user_periods, user_t0s, user_durations,
                    planet_number, mask_width)



    #determine cadence of observation
    cadence = determine_cadence(time)

    #find end time of quarters
    quarters_end = [el[1] for el in quarters]


    time_out, flux_out, flux_err_out, mask_out, mask_fitted_planet_out, moving_median = \
    reject_outliers_out_of_transit(time, lc, lc_err, mask, mask_fitted_planet, 30*cadence, 4)

    plot_outliers(time, lc, time_out, flux_out, 
                  moving_median, quarters_end, save_to_directory+flux_type+'_'+'outliers.pdf', star_id)
    if show_plots: plt.show()

    x_quarters, y_quarters, yerr_quarters, mask_quarters, mask_fitted_planet_quarters = \
    split_around_problems(time_out, flux_out, flux_err_out, 
                          mask_out, mask_fitted_planet_out, quarters_end)


    plot_split_data(x_quarters, y_quarters, t0s, save_to_directory+flux_type+'_'+'quarters_split.pdf', star_id)
    if show_plots: plt.show()

    x_quarters_w_transits, y_quarters_w_transits, yerr_quarters_w_transits, \
    mask_quarters_w_transits, mask_fitted_planet_quarters_w_transits = \
    find_quarters_with_transits(x_quarters, y_quarters, yerr_quarters, 
                                mask_quarters, mask_fitted_planet_quarters, t0s)




    x_quarters_w_transits = np.concatenate(x_quarters_w_transits, axis=0, dtype=object)
    y_quarters_w_transits = np.concatenate(y_quarters_w_transits, axis=0, dtype=object)
    yerr_quarters_w_transits = np.concatenate(yerr_quarters_w_transits, axis=0, dtype=object)
    mask_quarters_w_transits = np.concatenate(mask_quarters_w_transits, axis=0, dtype=object)
    mask_fitted_planet_quarters_w_transits = np.concatenate(mask_fitted_planet_quarters_w_transits, axis=0, dtype=object)






    mask_quarters_w_transits = np.array(mask_quarters_w_transits, dtype=bool)
    mask_fitted_planet_quarters_w_transits = np.array(mask_fitted_planet_quarters_w_transits, dtype=bool)








    x_transits, y_transits, yerr_transits, mask_transits, mask_fitted_planet_transits = split_around_transits(x_quarters_w_transits, 
                                                                                                              y_quarters_w_transits, 
                                                                                                              yerr_quarters_w_transits, 
                                                                                                              mask_quarters_w_transits, 
                                                                                                              mask_fitted_planet_quarters_w_transits, 
                                                                                                              t0s, 1./2., period)
    



    if len(mask_transits)==1:
      mask_transits = np.array(mask_transits, dtype=bool)
      mask_fitted_planet_transits = np.array(mask_fitted_planet_transits, dtype=bool)


    x_epochs = np.concatenate(x_transits, axis=0, dtype=object)
    y_epochs = np.concatenate(y_transits, axis=0, dtype=object)
    yerr_epochs = np.concatenate(yerr_transits, axis=0, dtype=object)
    mask_epochs = np.concatenate(mask_transits, axis=0, dtype=object)
    mask_fitted_planet_epochs = np.concatenate(mask_fitted_planet_transits, axis=0, dtype=object)


    # check if problem times already exist
    if problem_times_default == 'use_sap':
      print('using sap problem times for pdc also')
      problem_path = save_to_directory + 'sap_flux_problem_times.txt'
    else:
      problem_path = save_to_directory + flux_type + '_problem_times.txt'

    if os.path.exists(problem_path):
      print(flux_type+' '+'problem times for '+star_id+' planet number '+str(planet_number)+' found')
      with open(problem_path, 'r') as problem_file:
        problem_times = json.load(problem_file)


    elif no_pdc_problem_times:
      if flux_type == 'pdcsap_flux':
        print('assuming no pdc problem times')
        problem_times = []

      # if not, mark out problem times manually
      else: 
        _, _, problem_times = plot_transits(x_transits, y_transits, mask_transits, t0s, period, cadence*5, star_id, dont_bin = dont_bin, data_name=data_name)
        # save problem times
        with open(problem_path, 'w') as problem_file:
          json.dump(problem_times, problem_file)
        print(flux_type+' problem times saved as ' + problem_path)

    # if not, mark out problem times manually for both pdc and sap
    else: 
      _, _, problem_times = plot_transits(x_transits, y_transits, mask_transits, t0s, period, cadence*5, star_id, dont_bin = dont_bin, data_name=data_name)
      # save problem times
      with open(problem_path, 'w') as problem_file:
        json.dump(problem_times, problem_file)
      print(flux_type+' problem times saved as ' + problem_path)

    return x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, problem_times, t0s, period, duration, cadence



def find_sap_and_pdc_flux_jumps(star_id, save_to_directory, show_plots, TESS = False, Kepler = False, 
                                user_periods = None, user_t0s = None, user_durations = None,
                                planet_number = 1, mask_width = 1.3, dont_bin = False, 
                                data_name = None, problem_times_default=None, no_pdc_problem_times=True):


    sap_vals = find_flux_jumps(star_id, 'sap_flux', save_to_directory, show_plots, TESS = TESS, Kepler = Kepler, 
                               user_periods = user_periods, user_t0s = user_t0s, user_durations = user_durations,
                               planet_number = planet_number, mask_width = mask_width, dont_bin = dont_bin, 
                               data_name=data_name, problem_times_default=problem_times_default, no_pdc_problem_times=no_pdc_problem_times)    

    pdc_vals = find_flux_jumps(star_id, 'pdcsap_flux', save_to_directory, show_plots, TESS = TESS, Kepler = Kepler, 
                               user_periods = user_periods, user_t0s = user_t0s, user_durations = user_durations,
                               planet_number = planet_number, mask_width = mask_width, dont_bin = dont_bin, 
                               data_name=data_name, problem_times_default=problem_times_default, no_pdc_problem_times=no_pdc_problem_times)    



    


    return sap_vals, pdc_vals






# #pulls in light curve
# [[sap_x_epochs, sap_y_epochs, sap_yerr_epochs, sap_mask_epochs, \
# sap_mask_fitted_planet_epochs, sap_problem_times, sap_t0s, sap_period, \
# sap_duration, sap_cadence], \
# [pdc_x_epochs, pdc_y_epochs, pdc_yerr_epochs, pdc_mask_epochs, \
# pdc_mask_fitted_planet_epochs, pdc_problem_times, pdc_t0s, pdc_period, \
# pdc_duration, pdc_cadence]]  = \
# find_sap_and_pdc_flux_jumps('toi-1130', TESS=True, planet_number = 2, mask_width = 1.8)






