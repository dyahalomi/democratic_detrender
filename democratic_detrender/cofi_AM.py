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
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from scipy.stats import median_absolute_deviation
from scipy import optimize

from democratic_detrender.manipulate_data import *
from democratic_detrender.helper_functions import *
from democratic_detrender.poly_AM import *

def cofiam_matrix_gen(times, degree):
    """
    Generate the design matrix for the CoFiAM method.

    Parameters:
        times: array-like, time values
        degree: int, degree of the CoFiAM fit

    Returns:
        array: design matrix for CoFiAM
    """

    baseline = np.nanmax(times) - np.nanmin(times)
    assert baseline > 0
    rows = len(times)
    cols = 2 * (degree + 1)
    X_matrix = np.ones(shape=(rows, cols))

    for x in range(rows):
        for y in range(1, int(cols / 2)):
            sinarg = (2 * np.pi * times[x] * y) / baseline
            X_matrix[x, y * 2] = np.sin(sinarg)
            X_matrix[x, y * 2 + 1] = np.cos(sinarg)
        X_matrix[x, 1] = times[x]

    return X_matrix


def cofiam_matrix_coeffs(times, fluxes, degree):
    """
    Calculate the coefficients for the CoFiAM method.

    Parameters:
        times: array-like, time values
        fluxes: array-like, flux values
        degree: int, degree of the CoFiAM fit

    Returns:
        tuple: design matrix for CoFiAM, coefficients
    """

    assert len(times) > 0
    Xmat = cofiam_matrix_gen(times, degree)
    beta_coefs = np.linalg.lstsq(Xmat, fluxes, rcond=None)[0]
    return Xmat, beta_coefs


def cofiam_function(times, fluxes, degree):
    """
    Calculate the CoFiAM function.

    Parameters:
        times: array-like, time values
        fluxes: array-like, flux values
        degree: int, degree of the CoFiAM fit

    Returns:
        array: CoFiAM function values
    """

    input_times = times.astype("f8")
    input_fluxes = fluxes.astype("f8")
    cofiam_matrix, cofiam_coefficients = cofiam_matrix_coeffs(
        input_times, input_fluxes, degree
    )
    output = np.matmul(cofiam_matrix, cofiam_coefficients)
    return output


def cofiam_iterative(
    times,
    fluxes,
    mask,
    mask_fitted_planet,
    local_start_x,
    local_end_x,
    max_degree=30,
    min_degree=1,
):
    """
    Iteratively apply CoFiAM method for multiple CoFiAM degrees and select the best fit.

    Parameters:
        times: array-like, time values
        fluxes: array-like, flux values
        mask: array-like, boolean mask for data points
        mask_fitted_planet: array-like, boolean mask for fitted planet
        local_start_x: float, start time for local region
        local_end_x: float, end time for local region
        max_degree: int, maximum CoFiAM degree to try
        min_degree: int, minimum CoFiAM degree to try

    Returns:
        tuple: best-fit model, best degree, best Durbin-Watson statistic, maximum degree
    """

    no_pre_transit = False
    no_post_transit = False

    vals_to_minimize = []
    models = []
    degs_to_try = np.arange(min_degree, max_degree + 1, 1)
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
            DWstat_pre_transit = 2.0
        else:
            local_start_index = np.where(times == local_start_x)[0][0]
            residuals_pre_transit = (
                (fluxes[local_start_index:in_transit_index] + 1)
                / (model[local_start_index:in_transit_index] + 1)
            ) - 1
            DWstat_pre_transit = DurbinWatson(residuals_pre_transit)

        if no_post_transit:
            DWstat_post_transit = 2.0
        else:
            local_end_index = np.where(times == local_end_x)[0][0]
            npoints_missing_from_model = out_transit_index - in_transit_index
            residuals_post_transit = (
                (fluxes[out_transit_index:local_end_index] + 1)
                / (
                    model[
                        out_transit_index
                        - npoints_missing_from_model : local_end_index
                        - npoints_missing_from_model
                    ]
                    + 1
                )
            ) - 1
            DWstat_post_transit = DurbinWatson(residuals_post_transit)
        val_to_minimize = np.sqrt(
            (DWstat_pre_transit - 2.0) ** 2.0 + (DWstat_post_transit - 2.0) ** 2.0
        )
        vals_to_minimize.append(val_to_minimize)
        models.append(model)

    best_degree = degs_to_try[np.argmin(np.array(vals_to_minimize))]
    best_DW_val = vals_to_minimize[np.argmin(np.array(vals_to_minimize))]
    best_model = models[np.argmin(np.array(vals_to_minimize))]

    return best_model, best_degree, best_DW_val, max_degree


def cofiam_method(x, y, yerr, mask, mask_fitted_planet, t0s, duration, period, local_x):
    """
    Apply the CoFiAM method for detrending a light curve.

    Parameters:
        x: array-like, time values
        y: array-like, flux values
        yerr: array-like, flux error values
        mask: array-like, boolean mask for data points
        mask_fitted_planet: array-like, boolean mask for fitted planet
        t0s: array-like, transit center times
        duration: float, transit duration
        period: float, transit period
        local_x: array-like, local region times for each epoch

    Returns:
        tuple: detrended light curve, Durbin-Watson statistics
    """

    from scipy.interpolate import interp1d

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
        local_end_x_ii = local_x[ii][len(local_x[ii]) - 1]

        try:

            cofiam = cofiam_iterative(
                x_ii,
                y_ii,
                mask_ii,
                mask_fitted_planet_ii,
                local_start_x_ii,
                local_end_x_ii,
                max_degree=30,
            )

            cofiam_interp = interp1d(
                x_ii[~mask_ii], cofiam[0], bounds_error=False, fill_value="extrapolate"
            )
            best_model = cofiam_interp(x_ii)

            DWs.append(cofiam[2])

            cofiam_mod.append(best_model)
            cofiam_mod_all.extend(best_model)

        except:
            print("CoFiAM failed for the " + str(ii) + "th epoch")
            # CoFiAM failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            cofiam_mod.append(nan_array)
            cofiam_mod_all.extend(nan_array)

        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_ii)

    # zoom into the local region of each transit
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
        model=np.array(cofiam_mod_all),
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
            print("CofiAM failed for the " + str(ii) + "th epoch")
            # CoFiAM failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(y_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)

    detrended_lc = np.concatenate(y_out_detrended, axis=0)

    return detrended_lc, DWs
