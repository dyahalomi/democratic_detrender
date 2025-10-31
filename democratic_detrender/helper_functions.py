""" This module contains helper functions for statistical calculations and data processing. """

import numpy as np


def durbin_watson(residuals):
    """
    Calculate the Durbin-Watson statistic for a given set of residuals.

    Parameters:
        residuals: array-like, residuals from a model

    Returns:
        float: Durbin-Watson statistic
    """

    residual_terms = np.diff(residuals)
    numerator = np.nansum(residual_terms ** 2)
    denominator = np.nansum(residuals ** 2)
    assert denominator != 0.0
    return numerator / denominator


def get_detrended_lc(y, detrending_model):
    """
    Get detrended light curve (LC).

    Parameters:
        y (array): Light curve (LC).
        detrending_model (array): Stellar detrending model evaluated at the same time as LC.

    Returns:
        array: Detrended light curve evaluated at the same time as input LC.
    """
    detrended_lc = ((y + 1) / (detrending_model + 1)) - 1

    return np.array(detrended_lc)


def determine_cadence(times):
    time_gaps = {}
    for ii in range(1, len(times)):
        time_gap = np.round(times[ii] - times[ii - 1], 4)
        if time_gap in time_gaps.keys():
            time_gaps[time_gap] += 1
        else:
            time_gaps[time_gap] = 1

    # find the key that corresponds to the most data gaps, this is the cadence
    cadence = max(time_gaps, key=time_gaps.get)
    return cadence


def find_nearest(array, value):
    # returns the value in an array closest to another input value

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def bin_data(xs, ys, window):

    import warnings

    xmin = np.min(xs)
    xmax = np.max(xs)

    x_bin = np.arange(xmin - window, xmax + window, window)

    y_bin = []
    for ii in range(0, len(x_bin)):
        y_bin.append([])

    for jj in range(1, len(x_bin)):
        x_bin_jj = x_bin[jj]
        x_bin_jj_minus_1 = x_bin[jj - 1]

    for ii in range(0, len(xs)):
        found_bin = False
        x_ii = xs[ii]
        y_ii = ys[ii]

        for jj in range(1, len(x_bin)):
            x_bin_jj = x_bin[jj]
            x_bin_jj_minus_1 = x_bin[jj - 1]

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


import numpy as np
from scipy.stats import median_abs_deviation

def ensemble_step(
    y_detrended,
    yerr_detrended=None,
    method="median",
):
    """
    Combine multiple detrending methods into a single flux time series
    (method-marginalized light curve) and propagate uncertainties.

    Parameters
    ----------
    y_detrended : array-like, shape (n_times, n_methods)
        Detrended flux values. Each column is the SAME target light curve
        after a different detrending method. Rows are timestamps.
        Can include NaNs.

    yerr_detrended : array-like, shape (n_times,)
        The per-point uncertainty for the target light curve BEFORE
        method-marginalization. If None, it's treated as zeros.
        (This matches how your snippet uses `yerr_detrended` as a 1D array.)

    method : {"median", "mean"}
        How to combine across methods at each timestamp:
        - "median": use np.nanmedian across methods,
                    and use MAD as the method-scatter term.
        - "mean":   use np.nanmean across methods,
                    and use standard deviation as the method-scatter term.

    Returns
    -------
    flux_mm : ndarray, shape (n_times,)
        Method-marginalized flux (the final combined light curve).

    flux_err_mm : ndarray, shape (n_times,)
        Updated per-point uncertainty after marginalizing over method.
        This is:
            sqrt( yerr_detrended^2 + scatter_term^2 )
        where scatter_term is MAD (for median) or std (for mean),
        both computed across methods at that timestamp.

    Notes
    -----
    - We transpose internally only to mirror your original code, where you
      did y_detrended.T then axis=1. After transpose we have
      shape (n_methods, n_times), so axis=0 would also work, but we'll
      stick to axis=0/axis=1 choices that reproduce your math exactly.
    - MAD uses scale=1/1.4826 to match your exact call, i.e. the *raw*
      unscaled MAD, not the Gaussian-scaled MAD.
    """

    # ensure ndarray, shape (n_times, n_methods)
    y_detrended = np.array(y_detrended, dtype=float)
    if y_detrended.ndim != 2:
        raise ValueError("y_detrended must be 2D: shape (n_times, n_methods)")

    n_times, n_methods = y_detrended.shape

    # handle input per-point errors
    if yerr_detrended is None:
        yerr_detrended_arr = np.zeros(n_times, dtype=float)
    else:
        yerr_detrended_arr = np.array(yerr_detrended, dtype=float)
        if yerr_detrended_arr.shape != (n_times,):
            raise ValueError(
                "yerr_detrended must be 1D with shape (n_times,)"
            )

    # transpose to match your axis usage (methods, times)
    y_T = y_detrended.T  # shape (n_methods, n_times)

    if method.lower() == "median":
        # method-marginalized flux = nanmedian across methods
        flux_mm = np.nanmedian(y_T, axis=0)  # (n_times,)

        # scatter between methods at each timestamp via MAD
        scatter_term = median_abs_deviation(
            y_T,
            axis=0,
            scale=1 / 1.4826,
            nan_policy="omit",
        )  # (n_times,)

    elif method.lower() == "mean":
        # method-marginalized flux = nanmean across methods
        flux_mm = np.nanmean(y_T, axis=0)  # (n_times,)

        # scatter between methods via std
        scatter_term = np.nanstd(y_T, axis=0)  # (n_times,)

    else:
        raise ValueError("method must be 'median' or 'mean'")

    # final marginalized errors:
    # sqrt( original_error^2 + method-scatter^2 )
    flux_err_mm = np.sqrt(yerr_detrended_arr.astype(float) ** 2 + scatter_term ** 2)

    return flux_mm, flux_err_mm

