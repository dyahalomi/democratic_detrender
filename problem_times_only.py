from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from matplotlib.widgets import Slider, Button
import sys, argparse
import os
import warnings
import ast

warnings.simplefilter('ignore', np.RankWarning)

parser = argparse.ArgumentParser(description="Looks up light curves for TESS and Kepler objects and enables labeling of jump times.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('object', type=str, help='TESS or Kepler identifier. ex: "toi-2088"')
parser.add_argument('--flux_type', type=str, default='sap', help='Flux type as a string. Options: pdc, sap, both, or qlp.')
parser.add_argument('--mission', type=str, default='TESS', help='Mission data select. ex: "TESS" or "Kepler"')
parser.add_argument('--planet_num', type=int, default=1, help='Which planet to look at in system. ex: 1')
parser.add_argument('--save_to_dir', type=str, default='./', help='Directory path to save csvs and figures to.')
parser.add_argument('-d', '--depth', default=0.02, help='Sets depth of detrended plots. Default is 0.02.')
parser.add_argument('-p', '--period', default=None, help='Optionally input period. Otherwise defaults to what \
    Exoplanet Archive can find.')
parser.add_argument('-t', '--t0', default=None, help='Optionally input t0. Otherwise defaults to what Exoplanet Archive can find.')
parser.add_argument('-du', '--duration', default=None, help='Optionally input duration. Otherwise defaults to what \
    Exoplanet Archive can find.')
parser.add_argument('-mw', '--mask_width', default=1.3, help='Sets mask width. Default is 1.3.')
parser.add_argument('-s', '--show_plots', default='True', help='Set whether to show non-problem-time plots.')
parser.add_argument('--dont_bin', default='False', help='if True, then jump time plots data wont be binned.')
parser.add_argument('--save_transit_data', default='False', help='if True, then will save data on each transit')


args = vars(parser.parse_args())

# user input parameters

input_id = args['object']
flux_type = args['flux_type']
mission_select = args['mission']
input_planet_number = args['planet_num']
input_dir = args['save_to_dir']
input_depth = args['depth']
input_period = args['period']
input_t0 = args['t0']
input_duration = args['duration']
input_mask_width = float(args['mask_width'])
input_show_plots = ast.literal_eval(args['show_plots'])
input_dont_bin = ast.literal_eval(args['dont_bin'])
input_save_transit_data = ast.literal_eval(args['save_transit_data'])



# # # ---------------------------------------------- now the fun begins ! ! ! ------------------------------------------------ # # #





from find_flux_jumps import *
from get_lc import *
from helper_functions import *
from outlier_rejection import *
from manipulate_data import *
from plot import plot_detrended_lc
from detrend import *


# checks user arguments
if mission_select == 'TESS': 
    tess_bool = True
    kepler_bool = False
else:
    tess_bool = False # assuming Kepler is selected
    kepler_bool = True


# determining figname
# determining today's date
today = date.today()
current_day = today.strftime('%B_%d_%Y')

foldername = input_id + '/' + input_id + '.0' + str(input_planet_number)+ '/' + 'detrending' + '/' + current_day
path = os.path.join(input_dir, foldername)

os.makedirs(path, exist_ok=True)


if input_save_transit_data:
    transit_data_name = path+'/'+flux_type+'_'+'transit_'
else:
    transit_data_name = None


# check if we should run both pdc and sap flux
if flux_type == 'both':

    #pulls in light curve
    [[sap_x_epochs, sap_y_epochs, sap_yerr_epochs, sap_mask_epochs, \
    sap_mask_fitted_planet_epochs, sap_problem_times, sap_t0s, sap_period, \
    sap_duration, sap_cadence], \
    [pdc_x_epochs, pdc_y_epochs, pdc_yerr_epochs, pdc_mask_epochs, \
    pdc_mask_fitted_planet_epochs, pdc_problem_times, pdc_t0s, pdc_period, \
    pdc_duration, pdc_cadence]]  = \
    find_sap_and_pdc_flux_jumps(input_id, path + '/', input_show_plots, TESS = tess_bool, Kepler = kepler_bool, planet_number = input_planet_number,
    user_periods = input_period, user_t0s = input_t0, user_durations = input_duration, mask_width=input_mask_width, dont_bin=input_dont_bin, data_name = transit_data_name) 


    







# check if we should run just pdc
elif flux_type == 'pdc':


    #pulls in light curve
    [pdc_x_epochs, pdc_y_epochs, pdc_yerr_epochs, pdc_mask_epochs, \
    pdc_mask_fitted_planet_epochs, pdc_problem_times, pdc_t0s, pdc_period, \
    pdc_duration, pdc_cadence]  = \
    find_flux_jumps(input_id, 'pdcsap_flux', path + '/', input_show_plots,
        TESS = tess_bool, Kepler = kepler_bool, 
        planet_number = input_planet_number,user_periods = input_period, 
        user_t0s = input_t0, user_durations = input_duration, 
        mask_width=input_mask_width, dont_bin=input_dont_bin, data_name = transit_data_name) 

    




# check if we should run just sap
elif flux_type == 'sap':

    #pulls in light curve
    [sap_x_epochs, sap_y_epochs, sap_yerr_epochs, sap_mask_epochs, \
    sap_mask_fitted_planet_epochs, sap_problem_times, sap_t0s, sap_period, \
    sap_duration, sap_cadence]  = \
    find_flux_jumps(input_id, 'sap_flux', path + '/', input_show_plots,
        TESS = tess_bool, Kepler = kepler_bool, 
        planet_number = input_planet_number,user_periods = input_period, 
        user_t0s = input_t0, user_durations = input_duration, 
        mask_width=input_mask_width, dont_bin=input_dont_bin, data_name = transit_data_name) 





# check if we should run just qlp
elif flux_type == 'qlp':


    #pulls in light curve
    [sap_x_epochs, sap_y_epochs, sap_yerr_epochs, sap_mask_epochs, \
    sap_mask_fitted_planet_epochs, sap_problem_times, sap_t0s, sap_period, \
    sap_duration, sap_cadence]  = \
    find_flux_jumps(input_id, 'qlp', path + '/', input_show_plots,
        TESS = tess_bool, Kepler = kepler_bool, 
        planet_number = input_planet_number,user_periods = input_period, 
        user_t0s = input_t0, user_durations = input_duration, 
        mask_width=input_mask_width, dont_bin=input_dont_bin, data_name = transit_data_name) 

    


else:
    print('ERROR!')
    print('invalid flux_type value entered...options are: pdc, sap, both, or qlp.')




