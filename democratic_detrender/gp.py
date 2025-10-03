""" This module contains functions to implement the Gaussian Process (GP) detrending method. """

import numpy as np
from scipy.interpolate import interp1d

import pymc3 as pm
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
# TODO: only using theano import for log and warning statements; could remove
import theano 
import logging

from democratic_detrender.manipulate_data import split_around_transits
from democratic_detrender.helper_functions import get_detrended_lc
from democratic_detrender.poly_AM import polyAM_function

def gp_new(time_star, lc_star, lc_err_star, time_model):

    # ignore theano warnings unless it's an error
    logger = logging.getLogger("theano.tensor.opt")
    logger.setLevel(logging.ERROR)

    theano.config.compute_test_value = "warn"

    with pm.Model() as model:

        rho_gp = pm.InverseGamma(
            "rho_gp", testval=2.0, **pmx.estimate_inverse_gamma_parameters(1.0, 5.0),
        )

        with pm.Model() as model:
            # The flux zero point
            mean = pm.Normal("mean", mu=0.0, sigma=10.0)

            # Noise parameters
            med_yerr = np.median(lc_err_star)
            std_y = np.std(lc_star)

            sigma_gp = pm.InverseGamma(
                "sigma_gp",
                testval=0.5 * std_y,
                **pmx.estimate_inverse_gamma_parameters(med_yerr, std_y),
            )

        # The Gaussian Process noise model
        kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / 3)
        gp = GaussianProcess(kernel, t=time_star, diag=lc_err_star ** 2, mean=mean)
        gp.marginal("gp", observed=lc_star)

        # Compute the GP model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict(lc_star, t=time_model))

        # Optimize the model
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln)

        return map_soln


def gp_method(x, y, yerr, mask, mask_fitted_planet, t0s, duration, period):

    theano.config.compute_test_value = "warn"

    gp_mod = []
    gp_mod_all = []

    x_all = []
    y_all = []
    yerr_all = []
    mask_all = []
    mask_fitted_planet_all = []
    for ii in range(0, len(x)):
        x_ii = x[ii]
        y_ii = y[ii]
        yerr_ii = yerr[ii]
        mask_ii = mask[ii]
        mask_fitted_planet_ii = mask_fitted_planet[ii]

        try:
            gp_model = gp_new(x_ii[~mask_ii], y_ii[~mask_ii], yerr_ii[~mask_ii], x_ii)

            gp_mod.append(gp_model["pred"])
            gp_mod_all.extend(gp_model["pred"])

        except:
            print("GP failed for the " + str(ii) + "th epoch")
            # gp failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            gp_mod.append(nan_array)
            gp_mod_all.extend(nan_array)

        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_ii)

    # zoom into local window
    (
        x_out,
        y_out,
        yerr_out,
        mask_out,
        mask_fitted_planet_out,
        model_out,
    ) = split_around_transits(
        np.array(x_all),
        np.array(y_all),
        np.array(yerr_all),
        np.array(mask_all),
        np.array(mask_fitted_planet_all),
        t0s,
        float(6 * duration / (24.0)) / period,
        period,
        model=np.array(gp_mod_all),
    )

    # add a linear polynomial fit at the end
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
            poly_interp = interp1d(
                x_ii[~mask_ii], linear_ii, bounds_error=False, fill_value="extrapolate"
            )
            model_ii_linear = poly_interp(x_ii)

            model_linear.append(model_ii_linear)

            y_ii_linear_detrended = get_detrended_lc(y_ii_detrended, model_ii_linear)
            y_out_detrended.append(y_ii_linear_detrended)

        except:
            print("GP failed for the " + str(ii) + "th epoch")
            # GP failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)

    detrended_lc = np.concatenate(y_out_detrended, axis=0)

    return detrended_lc
