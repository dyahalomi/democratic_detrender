import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button


def reject_outliers_out_of_transit(
    time, flux, flux_err, mask, mask_fitted_planet, time_window, sigma_window
):
    """
    rejects outliers via moving median and sigma clipping outside of transit mask
    
    input:
    time = array of time values
    flux = array of flux values
    flux_err = array of flux error values
    mask = array of all mask values
    mask = array of mask values only for the planet we are fitting 
    time_window = int, how much time around which to determine median on
    sigma_window = int, how many sigmas to clip
    
    
    returns:
    flux_out = array of flux values with outliers value changed to np.nan 
    flux_err_out = array of flux error values with outliers value changed to np.nan  
    
    """

    if len(time) != len(flux):
        print("error, mismatched time and flux data length")

    # find the time right before and after data gaps, this is used to only do outlier rejection after 1 full time_window
    # data gap is considered anything greater than 1 time window wide
    time_gap_adjacent = [time[0]]
    for ii in range(1, len(time)):
        if time[ii] - time[ii - 1] > time_window:
            time_gap_adjacent.append(time[ii])
            time_gap_adjacent.append(time[ii - 1])

    # also add in right before and after transits to time_gap_adjacent
    for ii in range(1, len(time)):
        if mask[ii] and not mask[ii - 1]:
            time_gap_adjacent.append(time[ii - 1])
        if not mask[ii] and mask[ii - 1]:
            time_gap_adjacent.append(time[ii])

    time_out = []
    flux_out = []
    flux_err_out = []
    mask_out = []
    mask_fitted_planet_out = []
    moving_median = []
    for ii in range(0, len(time)):
        current_time = time[ii]
        current_flux = flux[ii]
        current_flux_err = flux_err[ii]
        current_mask = mask[ii]
        current_mask_fitted_planet = mask_fitted_planet[ii]

        # find indices that we want to include in the moving median
        # include if greater than minimum window value
        # and smaller than maximum window value
        # and not during a transit
        indices = np.where(
            np.logical_and(
                np.logical_and(
                    time >= current_time - time_window / 2.0,
                    time <= current_time + time_window / 2.0,
                ),
                ~mask,
            )
        )
        current_flux_median = np.median(flux[indices])

        # only do outlier rejection outside of transits
        if not current_mask:

            # only do outlier rejection if not within 1 time window of a data gap or transit
            near_time_gaps = False
            for time_gap in time_gap_adjacent:
                if np.abs(current_time - time_gap) < time_window:
                    near_time_gaps = True

            if not near_time_gaps:
                if (
                    current_flux + sigma_window * current_flux_err
                    >= current_flux_median
                    and current_flux - sigma_window * current_flux_err
                    <= current_flux_median
                ):
                    time_out.append(current_time)
                    flux_out.append(current_flux)
                    flux_err_out.append(current_flux_err)
                    mask_out.append(current_mask)
                    mask_fitted_planet_out.append(current_mask_fitted_planet)
                    moving_median.append(current_flux_median)

                else:
                    moving_median.append(current_flux_median)

            else:
                time_out.append(current_time)
                flux_out.append(current_flux)
                flux_err_out.append(current_flux_err)
                mask_out.append(current_mask)
                mask_fitted_planet_out.append(current_mask_fitted_planet)
                moving_median.append(np.nan)

        else:
            time_out.append(current_time)
            flux_out.append(current_flux)
            flux_err_out.append(current_flux_err)
            mask_out.append(current_mask)
            mask_fitted_planet_out.append(current_mask_fitted_planet)
            moving_median.append(np.nan)

    time_out = np.array(time_out)
    flux_out = np.array(flux_out)
    flux_err_out = np.array(flux_err_out)
    mask_out = np.array(mask_out)
    mask_fitted_planet_out = np.array(mask_fitted_planet_out)

    return (
        time_out,
        flux_out,
        flux_err_out,
        mask_out,
        mask_fitted_planet_out,
        moving_median,
    )


def reject_outliers_everywhere(
    time, flux, flux_err, time_window, npoints_window, sigma_window
):
    """
    rejects outliers via moving median and sigma clipping outside of transit mask
    
    input:
    time = array of time values
    flux = array of flux values
    flux_err = array of flux error values
    time_window = float, much time before and after data gaps (of length >time_window) for which we don't do outlier rejection
    npoints_window = int, how points around which to determine median on
    sigma_window = int, how many sigmas to clip
    
    
    returns:
    time_out = array of time values with outliers removed
    flux_out = array of flux values with outliers removed
    
    """

    if len(time) != len(flux):
        print("error, mismatched time and flux data length")

    # find the time right before and after data gaps, this is used to only do outlier rejection after 1 full time_window
    # data gap is considered anything greater than 1 time window wide
    time_gap_adjacent = [time[0], time[len(time) - 1]]

    for ii in range(1, len(time)):
        if time[ii] - time[ii - 1] > time_window:
            time_gap_adjacent.append(time[ii])
            time_gap_adjacent.append(time[ii - 1])

    time_out = []
    flux_out = []
    moving_median = []
    for ii in range(0, len(time)):
        current_time = time[ii]
        current_flux = flux[ii]
        current_flux_err = flux_err[ii]

        # only do outlier rejection if not within 1 time window of a data gap or transit
        near_time_gaps = False
        for time_gap in time_gap_adjacent:
            if np.abs(current_time - time_gap) < time_window:
                near_time_gaps = True

        if not near_time_gaps:

            # find indices that we want to include in the moving median
            # include if greater than minimum window value
            # and smaller than maximum window value
            n_around = int(np.floor(npoints_window / 2.0))
            indices = np.arange(ii - n_around, ii + n_around + 1)

            current_flux_median = np.nanmedian(flux[indices])

            if (
                current_flux + sigma_window * current_flux_err >= current_flux_median
                and current_flux - sigma_window * current_flux_err
                <= current_flux_median
            ):
                time_out.append(current_time)
                flux_out.append(current_flux)

            else:
                if not np.isnan(current_flux):
                    moving_median.append(current_flux_median)

                else:
                    time_out.append(current_time)
                    flux_out.append(current_flux)

        else:
            time_out.append(current_time)
            flux_out.append(current_flux)

    time_out = np.array(time_out)
    flux_out = np.array(flux_out)

    return time_out, flux_out
