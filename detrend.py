import numpy as np
import time

from cofi_AM import *
from poly_AM import *
from poly_local import * 
from gp import *
from plot import plot_individual_outliers
from manipulate_data import *
from outlier_rejection import *

def trim_jump_times(x, y, yerr, mask, mask_fitted_planet, t0s, period, jump_times):
	#x = time 
	#y = flux 
	#yerr = flux error
	#mask = mask
	#t0s = midtransits
	#period = planet period to define plotting limit
	#jump_times = jump times to trim structured noise
	
   
	if jump_times != []:
		x_epochs = []
		y_epochs = []
		yerr_epochs = []
		mask_epochs = []
		mask_fitted_planet_epochs = []
	
		# making this so that minimum of two jump times per epoch isn't needed


		x_transits, y_transits, yerr_transits, mask_transits, mask_fitted_planet_transits = split_around_transits(x, y, yerr, 
																									mask, mask_fitted_planet,
																									t0s, 1./2., period)
		if len(mask_transits)==1:
			mask_transits = np.array(mask_transits, dtype=bool)
			mask_fitted_planet_transits = np.array(mask_fitted_planet_transits, dtype=bool)

		for ii in range(0, len(t0s)):
			t0 = t0s[ii]
			xs = x_transits[ii]
			ys = y_transits[ii]
			yerrs = yerr_transits[ii]
			masks = mask_transits[ii]
			mask_fitted_planets = mask_fitted_planet_transits[ii]
			
			epoch_jump_times = []
			for j in range(len(jump_times)):
				if jump_times[j] >= xs[0] and jump_times[j] <= xs[-1]: # if jump time falls within time range of epoch
					epoch_jump_times.append(jump_times[j])


			# should be in time order anyway
			# there should be a max of two per epoch
			if len(epoch_jump_times) == 2 : # if we have two jump times in this epoch
				jump_start = find_nearest(xs, epoch_jump_times[0]) # selecting all data in between jump times
				jump_end = find_nearest(xs, epoch_jump_times[1])

			elif len(epoch_jump_times) == 1:
				# first make sure t0 is intact

				if t0 < epoch_jump_times[0]:
					jump_start = xs[0]
					jump_end = find_nearest(xs,epoch_jump_times[0]) # selecting data to left of jump time
				elif t0 > epoch_jump_times[0]: # t0 is intact
					jump_start = find_nearest(xs, epoch_jump_times[0])
					jump_end = xs[-1] # selecting all data to right of jump time
				else: # t0 = jump time?? weird
					raise Exception('Missing about half the transit. Go back and relabel?')
					return None
				# not sure how to check whether transit is cut off without duration; write in later

			elif len(epoch_jump_times) == 0: # no jump times for this epoch
				jump_start = xs[0] # selecting all data
				jump_end = xs[-1]

			else: # ????? too many problem times
				raise Exception('Too many problem times for epoch '+ str(ii+1) + '. Go back and relabel?')
				return None

			# # assuming all went well
			epoch_split = [jump_start, jump_end]
			start_index = int(np.where(xs == epoch_split[0])[0])
			end_index = int(np.where(x == epoch_split[1])[0])

			# now to append current epoch's trimmed (if applicable) data to epoch arrays
			x_epochs.append(xs[start_index:end_index])
			y_epochs.append(ys[start_index:end_index])
			yerr_epochs.append(yerrs[start_index:end_index])
			mask_epochs.append(masks[start_index:end_index])
			mask_fitted_planet_epochs.append(mask_fitted_planets[start_index:end_index])

				
		# wrapping it all up!
		x_epochs = np.array(x_epochs, dtype=object)
		y_epochs = np.array(y_epochs, dtype=object)
		yerr_epochs = np.array(yerr_epochs, dtype=object)
		mask_epochs = np.array(mask_epochs, dtype=object)
		mask_fitted_planet_epochs = np.array(mask_fitted_planet_epochs, dtype=object)

	else:
		x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs = \
		split_around_transits(x, y, yerr, mask, mask_fitted_planet, t0s, 1./2., period)
	
	
	return x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs

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

def remove_trim_times(x, y, yerr, trim_times):
	
	for trim in trim_times:
		trim_min = find_nearest(x, trim[0]) 
		trim_max = find_nearest(x, trim[1]) 
		
		index_min = int(np.where(x == trim_min)[0])
		index_max = int(np.where(x == trim_max)[0])
		
		indices = np.arange(index_min, index_max)

		
		x = np.delete(x, indices)
		y = np.delete(y, indices)
		yerr = np.delete(yerr, indices)
		
	return x, y, yerr






'''
def detrend_all_methods(x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, problem_times,
	t0s, period, duration, cadence, save_to_directory, show_plots):

	print(len(x_epochs))
	
	x_trimmed, y_trimmed, yerr_trimmed, mask_trimmed, mask_fitted_planet_trimmed = \
	trim_jump_times(x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, t0s, period, problem_times)

	#### polyam, gp, cofiam friendly mask arrays ####

	friendly_mask_trimmed = []
	for boolean in range(len(mask_trimmed)):
		friendly_boolean = mask_trimmed[boolean].astype(bool)
		friendly_mask_trimmed.append(friendly_boolean)
	
	friendly_mask_fitted_planet_trimmed = []
	for boolean in range(len(mask_fitted_planet_trimmed)):
		friendly_boolean = mask_fitted_planet_trimmed[boolean].astype(bool)
		friendly_mask_fitted_planet_trimmed.append(friendly_boolean)

	friendly_x_trimmed = []
	for time_array in range(len(x_trimmed)):
		friendly_time_array = x_trimmed[time_array].astype(float)
		friendly_x_trimmed.append(friendly_time_array)

	friendly_y_trimmed = []
	for flux_array in range(len(y_trimmed)):
		friendly_flux_array = y_trimmed[flux_array].astype(float)
		friendly_y_trimmed.append(friendly_flux_array)

	friendly_yerr_trimmed = []
	for flux_err_array in range(len(yerr_trimmed)):
		friendly_flux_err_array = yerr_trimmed[flux_err_array].astype(float)
		friendly_yerr_trimmed.append(friendly_flux_err_array)

	#################################################

	# determine local window values for later use
	# zoom in around local window
	local_x_epochs, local_y_epochs, local_yerr_epochs, \
	local_mask_epochs, local_mask_fitted_planet_epochs = \
	split_around_transits(np.concatenate(x_trimmed, axis=0, dtype=object), 
						  np.concatenate(y_trimmed, axis=0, dtype=object),
						  np.concatenate(yerr_trimmed, axis=0, dtype=object),
						  np.concatenate(mask_trimmed, axis=0, dtype=object),
						  np.concatenate(mask_fitted_planet_trimmed, axis=0, dtype=object),
						  t0s, float(6*duration/(24.))/period, period)

	local_x = np.concatenate(local_x_epochs, axis=0, dtype=object)
	local_y = np.concatenate(local_y_epochs, axis=0, dtype=object)
	local_yerr = np.concatenate(local_yerr_epochs, axis=0, dtype=object)
	local_mask = np.concatenate(local_mask_epochs, axis=0, dtype=object)
	local_mask_fitted_planet = np.concatenate(local_mask_fitted_planet_epochs, axis=0, dtype=object)

	####################
	####################
	####################
	# local detrending
	start = time.time()
	print('')
	print('detrending via the local method')
	local_detrended = \
	local_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
				 friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
				 t0s, duration, period)


	# remove outliers in unmasked local detrended lc
	local_x_no_outliers, local_detrended_no_outliers = \
	reject_outliers_everywhere(local_x, local_detrended, local_yerr, 5*cadence, 5, 10)

	plot_individual_outliers(local_x, local_detrended, local_x_no_outliers, local_detrended_no_outliers,
							 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

	end = time.time()
	print('local detrending took ' + str(np.round(end - start, 2)) + ' seconds')
	




	####################
	####################
	####################
	# polyAM detrending
	start = time.time()
	print('')
	print('detrending via the polyAM method')
	poly_detrended, poly_DWs = \
	polynomial_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
					  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
					  t0s, duration, period, local_x_epochs)


	# remove outliers in unmasked poly detrended lc
	poly_x_no_outliers, poly_detrended_no_outliers = \
	reject_outliers_everywhere(local_x, poly_detrended, local_yerr, 5*cadence, 5, 10)

	plot_individual_outliers(local_x, poly_detrended, poly_x_no_outliers, poly_detrended_no_outliers,
							 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

	end = time.time()
	print('polyAM detrending took ' + str(np.round(end - start, 2)) + ' seconds')






	####################
	####################
	####################
	# gp detrending
	start = time.time()
	print('')
	print('detrending via the GP method')
	gp_detrended = \
	gp_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
			  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
			  t0s, duration, period)

	# remove outliers in unmasked gp detrended lc
	gp_x_no_outliers, gp_detrended_no_outliers = \
	reject_outliers_everywhere(local_x, gp_detrended, local_yerr, 5*cadence, 5, 10)

	plot_individual_outliers(local_x, gp_detrended, gp_x_no_outliers, gp_detrended_no_outliers,
							 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

	end = time.time()
	print('GP detrending took ' + str(np.round(end - start, 2)) + ' seconds')





	####################
	####################
	####################
	# CoFiAM detrending
	start = time.time()
	print('')
	print('detrending via the CoFiAM method')
	cofiam_detrended, cofiam_DWs = \
	cofiam_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
				  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
				  t0s, duration, period, local_x_epochs)

	# remove outliers in unmasked CoFiAM detrended lc
	cofiam_x_no_outliers, cofiam_detrended_no_outliers = \
	reject_outliers_everywhere(local_x, cofiam_detrended, local_yerr, 5*cadence, 5, 10)

	plot_individual_outliers(local_x, cofiam_detrended, cofiam_x_no_outliers, cofiam_detrended_no_outliers,
							 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

	end = time.time()
	print('CoFiAM detrending took ' + str(np.round(end - start, 2)) + ' seconds')







	return local_x, local_y, local_yerr, local_mask, local_mask_fitted_planet, \
	local_detrended, local_x_no_outliers, local_detrended_no_outliers, \
	poly_detrended, poly_DWs, poly_x_no_outliers, poly_detrended_no_outliers, \
	gp_detrended, gp_x_no_outliers, gp_detrended_no_outliers, \
	cofiam_detrended, cofiam_DWs, cofiam_x_no_outliers, cofiam_detrended_no_outliers


'''




def detrend_variable_methods(x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, problem_times,
	t0s, period, duration, cadence, save_to_directory, show_plots, detrend_methods):

	
	x_trimmed, y_trimmed, yerr_trimmed, mask_trimmed, mask_fitted_planet_trimmed = \
	trim_jump_times(x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, t0s, period, problem_times)

	#### polyam, gp, cofiam friendly mask arrays ####

	friendly_mask_trimmed = []
	for boolean in range(len(mask_trimmed)):
		friendly_boolean = mask_trimmed[boolean].astype(bool)
		friendly_mask_trimmed.append(friendly_boolean)
	
	friendly_mask_fitted_planet_trimmed = []
	for boolean in range(len(mask_fitted_planet_trimmed)):
		friendly_boolean = mask_fitted_planet_trimmed[boolean].astype(bool)
		friendly_mask_fitted_planet_trimmed.append(friendly_boolean)

	friendly_x_trimmed = []
	for time_array in range(len(x_trimmed)):
		friendly_time_array = x_trimmed[time_array].astype(float)
		friendly_x_trimmed.append(friendly_time_array)

	friendly_y_trimmed = []
	for flux_array in range(len(y_trimmed)):
		friendly_flux_array = y_trimmed[flux_array].astype(float)
		friendly_y_trimmed.append(friendly_flux_array)

	friendly_yerr_trimmed = []
	for flux_err_array in range(len(yerr_trimmed)):
		friendly_flux_err_array = yerr_trimmed[flux_err_array].astype(float)
		friendly_yerr_trimmed.append(friendly_flux_err_array)

	#################################################

	# determine local window values for later use
	# zoom in around local window
	local_x_epochs, local_y_epochs, local_yerr_epochs, \
	local_mask_epochs, local_mask_fitted_planet_epochs = \
	split_around_transits(np.concatenate(x_trimmed, axis=0, dtype=object), 
						  np.concatenate(y_trimmed, axis=0, dtype=object),
						  np.concatenate(yerr_trimmed, axis=0, dtype=object),
						  np.concatenate(mask_trimmed, axis=0, dtype=object),
						  np.concatenate(mask_fitted_planet_trimmed, axis=0, dtype=object),
						  t0s, float(6*duration/(24.))/period, period)

	local_x = np.concatenate(local_x_epochs, axis=0, dtype=object)
	local_y = np.concatenate(local_y_epochs, axis=0, dtype=object)
	local_yerr = np.concatenate(local_yerr_epochs, axis=0, dtype=object)
	local_mask = np.concatenate(local_mask_epochs, axis=0, dtype=object)
	local_mask_fitted_planet = np.concatenate(local_mask_fitted_planet_epochs, axis=0, dtype=object)

	####################
	####################
	####################
	detrend_methods_success = []
	# local detrending
	if 'local' in detrend_methods:
		start = time.time()
		#try:
		print('')
		print('detrending via the local method')
		local_detrended = \
		local_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
					 friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
					 t0s, duration, period)


		# remove outliers in unmasked local detrended lc
		local_x_no_outliers, local_detrended_no_outliers = \
		reject_outliers_everywhere(local_x, local_detrended, local_yerr, 5*cadence, 5, 10)

		plot_individual_outliers(local_x, local_detrended, local_x_no_outliers, local_detrended_no_outliers,
								 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

		end = time.time()
		print('local detrending completed in ' + str(np.round(end - start, 2)) + ' seconds')
		#detrend_methods_success.append('local')


		#except:
		#	end = time.time()
		#	print('local detrending failed in ' + str(np.round(end - start, 2)) + ' seconds')
		




	####################
	####################
	####################
	# polyAM detrending
	if 'polyAM' in detrend_methods:
		start = time.time()
		#try:
		print('')
		print('detrending via the polyAM method')
		poly_detrended, poly_DWs = \
		polynomial_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
						  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
						  t0s, duration, period, local_x_epochs)


		# remove outliers in unmasked poly detrended lc
		poly_x_no_outliers, poly_detrended_no_outliers = \
		reject_outliers_everywhere(local_x, poly_detrended, local_yerr, 5*cadence, 5, 10)

		plot_individual_outliers(local_x, poly_detrended, poly_x_no_outliers, poly_detrended_no_outliers,
								 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

		end = time.time()
		print('polyAM detrending completed in ' + str(np.round(end - start, 2)) + ' seconds')
		#detrend_methods_success.append('polyAM')

		#except:
		#	end = time.time()
		#	print('polyAM detrending failed in ' + str(np.round(end - start, 2)) + ' seconds')







	####################
	####################
	####################
	# gp detrending
	if 'GP' in detrend_methods:
		start = time.time()
		#try:
		print('')
		print('detrending via the GP method')
		gp_detrended = \
		gp_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
				  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
				  t0s, duration, period)

		# remove outliers in unmasked gp detrended lc
		gp_x_no_outliers, gp_detrended_no_outliers = \
		reject_outliers_everywhere(local_x, gp_detrended, local_yerr, 5*cadence, 5, 10)

		plot_individual_outliers(local_x, gp_detrended, gp_x_no_outliers, gp_detrended_no_outliers,
								 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

		end = time.time()
		print('GP detrending completed in ' + str(np.round(end - start, 2)) + ' seconds')
		#detrend_methods_success.append('GP')

		#except:
		#	end = time.time()
		#	print('GP detrending failed in ' + str(np.round(end - start, 2)) + ' seconds')



	####################
	####################
	####################
	# CoFiAM detrending
	if 'CoFiAM' in detrend_methods:
		start = time.time()
		#try:
		print('')
		print('detrending via the CoFiAM method')
		cofiam_detrended, cofiam_DWs = \
		cofiam_method(friendly_x_trimmed, friendly_y_trimmed, friendly_yerr_trimmed, 
					  friendly_mask_trimmed, friendly_mask_fitted_planet_trimmed,
					  t0s, duration, period, local_x_epochs)

		# remove outliers in unmasked CoFiAM detrended lc
		cofiam_x_no_outliers, cofiam_detrended_no_outliers = \
		reject_outliers_everywhere(local_x, cofiam_detrended, local_yerr, 5*cadence, 5, 10)

		plot_individual_outliers(local_x, cofiam_detrended, cofiam_x_no_outliers, cofiam_detrended_no_outliers,
								 t0s, period, float(6*duration/(24.))/period, 0.009, save_to_directory)

		end = time.time()
		print('CoFiAM completed in ' + str(np.round(end - start, 2)) + ' seconds')
		#detrend_methods_success.append('CoFiAM')

		#except:
		#	end = time.time()
		#	print('CoFiAM detrending failed in ' + str(np.round(end - start, 2)) + ' seconds')




	output = [local_x, local_y, local_yerr, local_mask, local_mask_fitted_planet]
	nan_array = np.empty(np.shape(local_x))
	nan_array[:] = np.nan
	detrend_methods_out = []
	if 'local' in detrend_methods:
		detrend_methods_out.append('local')
		output.append(local_detrended)
		output.append(local_x_no_outliers)
		output.append(local_detrended_no_outliers)
	else:
		output.append(nan_array)
		output.append(nan_array)
		output.append(nan_array)

	if 'polyAM' in detrend_methods:
		detrend_methods_out.append('polyAM')
		output.append(poly_detrended)
		output.append(poly_x_no_outliers)
		output.append(poly_detrended_no_outliers)
	else:
		output.append(nan_array)
		output.append(nan_array)
		output.append(nan_array)

	if 'GP' in detrend_methods:
		detrend_methods_out.append('GP')
		output.append(gp_detrended)
		output.append(gp_x_no_outliers)
		output.append(gp_detrended_no_outliers)
	else:
		output.append(nan_array)
		output.append(nan_array)
		output.append(nan_array)

	if 'CoFiAM' in detrend_methods:
		detrend_methods_out.append('CoFiAM')
		output.append(cofiam_detrended)
		output.append(cofiam_x_no_outliers)
		output.append(cofiam_detrended_no_outliers)
	else:
		output.append(nan_array)
		output.append(nan_array)
		output.append(nan_array)

	return detrend_methods_out, output
















def detrend_sap_and_pdc(sap_values, pdc_values, save_dir, pop_out_plots, detrend_methods):

	# assumes order of sap, pdc arrays are as follows:
	# [pdc_x_epochs, pdc_y_epochs, pdc_yerr_epochs, pdc_mask_epochs, \
	# pdc_mask_fitted_planet_epochs, pdc_problem_times, pdc_t0s, pdc_period, \
	# pdc_duration, pdc_cadence]

	sap_detrend_methods_out, detrended_sap_vals = \
	detrend_variable_methods(x_epochs = sap_values[0], y_epochs = sap_values[1], yerr_epochs = sap_values[2],
		mask_epochs = sap_values[3], mask_fitted_planet_epochs = sap_values[4], problem_times = sap_values[5],
		t0s = sap_values[6], period = sap_values[7], duration = sap_values[8], cadence = sap_values[9],
		save_to_directory = save_dir, show_plots = pop_out_plots, detrend_methods = detrend_methods)

	pdc_detrend_methods_out, detrended_pdc_vals = \
	detrend_variable_methods(x_epochs = pdc_values[0], y_epochs = pdc_values[1], yerr_epochs = pdc_values[2],
		mask_epochs = pdc_values[3], mask_fitted_planet_epochs = pdc_values[4], problem_times = pdc_values[5],
		t0s = pdc_values[6], period = pdc_values[7], duration = pdc_values[8], cadence = pdc_values[9],
		save_to_directory = save_dir, show_plots = pop_out_plots, detrend_methods = detrend_methods)

	return sap_detrend_methods_out, pdc_detrend_methods_out, detrended_sap_vals, detrended_pdc_vals




def detrend_one_lc(lc_values, save_dir, pop_out_plots, detrend_methods):

	# assumes order of sap, pdc arrays are as follows:
	# [pdc_x_epochs, pdc_y_epochs, pdc_yerr_epochs, pdc_mask_epochs, \
	# pdc_mask_fitted_planet_epochs, pdc_problem_times, pdc_t0s, pdc_period, \
	# pdc_duration, pdc_cadence]


	detrend_methods_out, detrended_lc_vals = \
	detrend_variable_methods(x_epochs = lc_values[0], y_epochs = lc_values[1], yerr_epochs = lc_values[2],
		mask_epochs = lc_values[3], mask_fitted_planet_epochs = lc_values[4], problem_times = lc_values[5],
		t0s = lc_values[6], period = lc_values[7], duration = lc_values[8], cadence = lc_values[9],
		save_to_directory = save_dir, show_plots = pop_out_plots, detrend_methods = detrend_methods)


	return detrend_methods_out, detrended_lc_vals




