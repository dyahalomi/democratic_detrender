### Special thanks to Alex Teachey --> adapted from MoonPy package
### GitHub: https://github.com/alexteachey/MoonPy

""" This module contains functions to implement the Polynomial Local (polyLOC) detrending method. """

import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial

from democratic_detrender.manipulate_data import split_around_transits
from democratic_detrender.helper_functions import get_detrended_lc
from democratic_detrender.poly_AM import polyAM_function


def BIC(model, data, errors, nparams):
    chi2 = np.nansum(((model - data) / errors) ** 2)
    BICval = nparams * np.log(len(data)) + chi2
    return BICval


### this function spits out the best fit line!
def polyLOC_function(times, fluxes, degree):

    times = np.array(times, dtype=float)
    fluxes = np.array(fluxes, dtype=float)

    #poly_coeffs = np.polyfit(times, fluxes, degree)
    #model = np.polyval(poly_coeffs, times)
    
    # Fit a polynomial in a numerically stable way (x normalized to [-1,1])
    p = Polynomial.fit(times, fluxes, deg=degree)
    # Evaluate the fitted model at the original x points
    model = p(times)

    return model


def polyLOC_iterative(times, fluxes, errors, mask, max_degree=30, min_degree=1):
    ### this function utilizes polyLOC_function above, iterates it up to max_degree.
    ### max degree may be calculated using max_order function

    vals_to_min = []
    degs_to_try = np.arange(min_degree, max_degree + 1, 1)
    BICstats = []

    mask = np.array(mask, dtype=bool)
    for deg in degs_to_try:
        output_function = polyLOC_function(
            times[~mask], fluxes[~mask], deg
        )  ### this is the model
        residuals = fluxes[~mask] - output_function
        BICstat = BIC(output_function, fluxes[~mask], errors[~mask], deg + 1)
        BICstats.append(BICstat)

    BICstats = np.array(BICstats)

    best_degree = degs_to_try[np.argmin(BICstats)]
    best_BIC = BICstats[np.argmin(np.array(BICstats))]

    ### re-generate the function with the best degree

    best_model = polyLOC_function(times[~mask], fluxes[~mask], best_degree)

    return best_model, best_degree, best_BIC, max_degree


def local_method(
    x_epochs,
    y_epochs,
    yerr_epochs,
    mask_epochs,
    mask_fitted_planet_epochs,
    t0s,
    duration,
    period,
):

    x = np.concatenate(x_epochs, axis=0)
    y = np.concatenate(y_epochs, axis=0)
    yerr = np.concatenate(yerr_epochs, axis=0)
    mask = np.concatenate(mask_epochs, axis=0, dtype=bool)
    mask_fitted_planet = np.concatenate(mask_fitted_planet_epochs, axis=0, dtype=bool)

    (
        x_local,
        y_local,
        yerr_local,
        mask_local,
        mask_fitted_planet_local,
    ) = split_around_transits(
        x,
        y,
        yerr,
        mask,
        mask_fitted_planet,
        t0s,
        float(6 * duration / (24.0)) / period,
        period,
    )

    local_mod = []

    x_all = []
    y_all = []
    yerr_all = []
    mask_all = []
    mask_fitted_planet_all = []

    for ii in range(0, len(x_local)):
        x_ii = np.array(x_local[ii], dtype=float)
        y_ii = np.array(y_local[ii], dtype=float)
        yerr_ii = np.array(yerr_local[ii], dtype=float)
        mask_ii = np.array(mask_local[ii], dtype=bool)
        mask_fitted_planet_all_ii = np.array(mask_fitted_planet_local[ii], dtype=bool)

        try:
            local = polyLOC_iterative(x_ii, y_ii, yerr_ii, mask_ii)

            polyLOC_interp = interp1d(
                x_ii[~mask_ii], local[0], bounds_error=False, fill_value="extrapolate"
            )
            best_model = polyLOC_interp(x_ii)

            local_mod.append(best_model)


        except Exception as e:
            print(f"local failed for the {ii+1}th epoch: {e}")
            # local failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(x_ii))
            nan_array[:] = np.nan

            local_mod.append(nan_array)

        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_all_ii)

    # add a linear polynomial fit at the end
    model_linear = []
    y_out_detrended = []
    for ii in range(0, len(local_mod)):
        x_ii = np.array(x_local[ii], dtype=float)
        y_ii = np.array(y_local[ii], dtype=float)
        mask_ii = np.array(mask_local[ii], dtype=bool)
        model_ii = np.array(local_mod[ii], dtype=float)

        y_ii_detrended = get_detrended_lc(y_ii, model_ii)

        try:
            linear_ii = polyAM_function(x_ii[~mask_ii], y_ii_detrended[~mask_ii], 1)
            poly_interp = interp1d(
                x_ii[~mask_ii], linear_ii, bounds_error=False, fill_value="extrapolate"
            )
            model_ii_linear = poly_interp(x_ii)

            model_linear.append(model_ii_linear)

            y_ii_linear_detrended = get_detrended_lc(y_ii_detrended, model_ii_linear)
            y_out_detrended.append(y_ii_linear_detrended)

        except Exception as e:
            print(f"local failed for the {ii+1}th epoch at linear step: {e}")
            # local failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(x_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)

    detrended_lc = np.concatenate(y_out_detrended, axis=0)
    # detrended_x = np.concatenate(x_local, axis=0)

    return detrended_lc
