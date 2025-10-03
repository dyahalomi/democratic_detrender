#this works for 1 planet systems only for now!

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import argparse
import os
import warnings
import json

def detrend_only():
    print("working on turning this .py into a function")
    return None

# warnings.simplefilter('ignore', np.RankWarning)

# parser = argparse.ArgumentParser(description="Detrends a light curve file saved as a csv files [time, flux, flux_err, mask] saved at input_filename.",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('object', type=str, help='TESS or Kepler identifier. ex: "toi-2088"')
# parser.add_argument('input_lc_filename', type=str, help='Name of the input lightcurve filename')
# parser.add_argument('input_t0s_filename', type=str, help='Name of the input t0s filename')
# parser.add_argument('period', type=float, help='input period.')
# parser.add_argument('duration', type=float, help='input duration.')
# parser.add_argument('--save_to_dir', default='./', help='Directory path to save csvs and figures to.')
# parser.add_argument('-d', '--depth', default=0.02, help='Sets depth of detrended plots. Default is 0.02.')
# parser.add_argument('--problem_times', type=str, default=None, help='Location where problem times file is saved.')
# parser.add_argument('--output_figname', type=str, default='lc', help='Output figure name extension to be added on at the end.')





# args = vars(parser.parse_args())

# # user input parameters

# input_id = args['object']
# input_dir = args['save_to_dir']
# input_filename = args['input_lc_filename']
# input_t0s_filename = args['input_t0s_filename']
# input_period = float(args['period'])
# input_duration = float(args['duration'])
# input_depth = args['depth']
# input_dir = args['save_to_dir']
# input_problem_times = args['problem_times']
# output_figname = args['output_figname']





# # # # ---------------------------------------------- now the fun begins ! ! ! ------------------------------------------------ # # #

# from helper_functions import determine_cadence
# from plot import plot_detrended_lc
# from detrend import detrend_one_lc



# if input_dir == './':

#     # determining figname
#     # determining today's date
#     today = date.today()
#     current_day = today.strftime('%B_%d_%Y')

#     foldername = input_id + '/' + input_id + '.0' + str(1)+ '/' + 'detrending' + '/' + current_day
#     path = os.path.join(input_dir, foldername)

#     os.makedirs(path, exist_ok=True)

# else:

#     path = input_dir


# # load in the LC
# input_data = pd.read_csv(foldername+'/'+input_filename)
# input_time = input_data['time'].values
# input_flux = input_data['flux'].values
# input_flux_err = input_data['flux_err'].values
# input_mask = input_data['mask'].values

# input_time = np.array(input_time, dtype=float)
# input_flux = np.array(input_flux, dtype=float)
# input_flux_err = np.array(input_flux_err, dtype=float)
# input_mask = np.array(input_mask, dtype=bool)

# # load in the t0s
# t0s = np.genfromtxt(foldername+'/'+input_t0s_filename, delimiter=',')
# if t0s.ndim == 0:
#     t0s = np.array([t0s])


# # load in the problem times
# if input_problem_times != None:
#     with open(foldername+'/'+input_problem_times, 'r') as problem_file:
#         problem_times = json.load(problem_file)
# else:
#     problem_times = []

# # now for detrending!
# print('Detrending now.')
    


# cadence = determine_cadence(input_time)
# [local_x, local_y, local_yerr, local_mask, local_mask_fitted_planet, \
# local_detrended, local_x_no_outliers, local_detrended_no_outliers, \
# poly_detrended, poly_DWs, poly_x_no_outliers, poly_detrended_no_outliers, \
# gp_detrended, gp_x_no_outliers, gp_detrended_no_outliers, \
# cofiam_detrended, cofiam_DWs, cofiam_x_no_outliers, cofiam_detrended_no_outliers] = \
# detrend_one_lc(lc_values = [input_time, input_flux, input_flux_err, input_mask, input_mask, 
#                             problem_times, t0s, input_period, input_duration, cadence], 
#                 save_dir = input_dir, pop_out_plots = True)


# x_detrended = local_x

# # # now to plot!

# green2, green1 = '#355E3B', '#18A558'
# blue2, blue1 = '#000080', '#4682B4'
# purple2, purple1 = '#2E0854','#9370DB'
# red2, red1 = '#770737', '#EC8B80'


# colors = [red2,
#           blue2,
#           green2,
#           purple2]

# y_detrended = [local_detrended,
#                poly_detrended,
#                gp_detrended,
#                cofiam_detrended]

# yerr_detrended = local_yerr
# mask_detrended = local_mask

# detrend_label = ['local',
#                  'polyAM',
#                  'GP', 
#                  'CoFiAM']



# plot_detrended_lc(x_detrended, y_detrended, detrend_label,
#                   t0s, float(6*input_duration/(24.))/input_period, input_period,
#                   colors, input_duration, depth=input_depth,
#                   figname = path + '/' + 'individual_detrended_'+output_figname+'.pdf', mask_width=1.1)

# # plotting method_marginalized_detrended data
# y_detrended = np.array(y_detrended)
# y_detrended_transpose = y_detrended.T

# method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
# MAD = median_abs_deviation(y_detrended_transpose, axis=1, scale=1/1.4826, nan_policy = 'omit')

# yerr_detrended = np.sqrt(yerr_detrended.astype(float)**2 + MAD**2)


# plot_detrended_lc(x_detrended, [method_marg_detrended], ['method marg'],
#                   t0s, float(6*input_duration/(24.))/input_period, input_period,
#                   ['k'], input_duration, depth = input_depth,
#                   figname = path + '/' + 'method_marg_detrended_'+output_figname+'.pdf', mask_width=1.1)


# # saving detrend data as csv

# detrend_dict = {}

# detrend_dict["time"] = x_detrended
# detrend_dict["yerr"] = yerr_detrended
# detrend_dict["mask"] = mask_detrended
# detrend_dict["method marginalized"] = method_marg_detrended



# for ii in range(0, len(y_detrended)):
#     detrend = y_detrended[ii]
#     label = detrend_label[ii]
#     detrend_dict[label] = detrend
    
    
# detrend_df = pd.DataFrame(detrend_dict)

# detrend_df.to_csv(path + '/' + 'detrended_'+output_figname+'.csv')








