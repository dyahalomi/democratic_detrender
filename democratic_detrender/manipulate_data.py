import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button

from .helper_functions import find_nearest


def split_around_problems(x, y, yerr, mask, mask_fitted_planet, problem_times):
    problem_split_x = []
    problem_split_y = []
    problem_split_yerr = []
    problem_split_mask = []
    problem_split_mask_fitted_planet = []

    split_x = []
    split_y = []
    split_yerr = []
    split_mask = []
    split_mask_fitted_planet = []

    split_times = []
    for time in problem_times:
        split_times.append(find_nearest(x, time))

    for ii in range(0, len(x)):
        time = x[ii]
        flux = y[ii]
        flux_err = yerr[ii]
        a_mask = mask[ii]
        a_mask_fitted_planet = mask_fitted_planet[ii]

        if time in split_times:

            problem_split_x.append(np.array(split_x))
            problem_split_y.append(np.array(split_y))
            problem_split_yerr.append(np.array(split_yerr))
            problem_split_mask.append(np.array(split_mask))
            problem_split_mask_fitted_planet.append(np.array(split_mask_fitted_planet))

            split_x = []
            split_y = []
            split_yerr = []
            split_mask = []
            split_mask_fitted_planet = []

        else:
            split_x.append(time)
            split_y.append(flux)
            split_yerr.append(flux_err)
            split_mask.append(a_mask)
            split_mask_fitted_planet.append(a_mask_fitted_planet)

    output = [
        np.array(problem_split_x, dtype=object),
        np.array(problem_split_y, dtype=object),
        np.array(problem_split_yerr, dtype=object),
        np.array(problem_split_mask, dtype=object),
        np.array(problem_split_mask_fitted_planet, dtype=object),
    ]
    return output


def add_nans_for_missing_data(
    sap_x_local,
    sap_detrended_lcs,
    sap_yerr_local,
    sap_mask_local,
    sap_mask_fitted_planet_local,
    pdc_x_local,
    pdc_detrended_lcs,
    pdc_yerr_local,
    pdc_mask_local,
    pdc_mask_fitted_planet_local,
):

    print("")
    print("")
    print("pdc length in: ", str(len(pdc_x_local)))
    print("sap length in: ", str(len(sap_x_local)))
    print("---")

    for ii in range(0, len(pdc_x_local)):
        time = pdc_x_local[ii]
        yerr = pdc_yerr_local[ii]
        mask = pdc_mask_local[ii]
        mask_fitted_planet = pdc_mask_fitted_planet_local[ii]
        if time not in sap_x_local:
            for kk in range(0, len(sap_x_local)):
                if sap_x_local[kk] > time:
                    sap_x_local = np.insert(sap_x_local, kk, time)
                    sap_yerr_local = np.insert(sap_yerr_local, kk, yerr)
                    sap_mask_local = np.insert(sap_mask_local, kk, mask)
                    sap_mask_fitted_planet_local = np.insert(
                        sap_mask_fitted_planet_local, kk, mask_fitted_planet
                    )
                    for jj in range(0, len(sap_detrended_lcs)):
                        sap_detrended_lcs[jj] = np.insert(
                            sap_detrended_lcs[jj], kk, np.nan
                        )

                    break

                elif kk + 1 == len(sap_x_local):
                    sap_x_local = np.insert(sap_x_local, kk + 1, time)
                    sap_yerr_local = np.insert(sap_yerr_local, kk + 1, yerr)
                    sap_mask_local = np.insert(sap_mask_local, kk + 1, mask)
                    sap_mask_fitted_planet_local = np.insert(
                        sap_mask_fitted_planet_local, kk + 1, mask_fitted_planet
                    )
                    for jj in range(0, len(sap_detrended_lcs)):
                        sap_detrended_lcs[jj] = np.insert(
                            sap_detrended_lcs[jj], kk + 1, np.nan
                        )

                    break

    for ii in range(0, len(sap_x_local)):
        time = sap_x_local[ii]
        yerr = sap_yerr_local[ii]
        mask = sap_mask_local[ii]
        mask_fitted_planet = sap_mask_fitted_planet_local[ii]
        if time not in pdc_x_local:
            for kk in range(0, len(pdc_x_local)):
                if pdc_x_local[kk] > time:
                    pdc_x_local = np.insert(pdc_x_local, kk, time)
                    pdc_yerr_local = np.insert(pdc_yerr_local, kk, yerr)
                    pdc_mask_local = np.insert(pdc_mask_local, kk, mask)
                    pdc_mask_fitted_planet_local = np.insert(
                        pdc_mask_fitted_planet_local, kk, mask_fitted_planet
                    )
                    for jj in range(0, len(pdc_detrended_lcs)):
                        pdc_detrended_lcs[jj] = np.insert(
                            pdc_detrended_lcs[jj], kk, np.nan
                        )

                    break

                elif kk + 1 == len(pdc_x_local):
                    pdc_x_local = np.insert(pdc_x_local, kk + 1, time)
                    pdc_yerr_local = np.insert(pdc_yerr_local, kk + 1, yerr)
                    pdc_mask_local = np.insert(pdc_mask_local, kk + 1, mask)
                    pdc_mask_fitted_planet_local = np.insert(
                        pdc_mask_fitted_planet_local, kk + 1, mask_fitted_planet
                    )
                    for jj in range(0, len(pdc_detrended_lcs)):
                        pdc_detrended_lcs[jj] = np.insert(
                            pdc_detrended_lcs[jj], kk + 1, np.nan
                        )

                    break

    print("pdc length out: ", str(len(pdc_x_local)))
    print("sap length out: ", str(len(sap_x_local)))

    print("")
    print("")
    print("")
    if (pdc_x_local == sap_x_local).all():
        x_detrended = pdc_x_local

    else:
        print("ERROR, pdc and sap x arrays aren't the same")

    yerr_detrended = np.nanmean([pdc_yerr_local, sap_yerr_local], axis=0)

    if (pdc_mask_local == sap_mask_local).all():
        mask_detrended = pdc_mask_local

    else:
        for ii in range(0, len(pdc_mask_local)):
            if pdc_mask_local[ii] != sap_mask_local[ii]:
                print(pdc_x_local[ii], pdc_mask_local[ii])
                print(sap_x_local[ii], sap_mask_local[ii])

                print("")
        print("ERROR, pdc and sap mask arrays aren't the same")

    if (pdc_mask_fitted_planet_local == sap_mask_fitted_planet_local).all():
        mask_fitted_planet_detrended = pdc_mask_fitted_planet_local

    else:
        print("ERROR, pdc and sap mask for fitted planet arrays aren't the same")

    return (
        x_detrended,
        sap_detrended_lcs,
        pdc_detrended_lcs,
        yerr_detrended,
        mask_detrended,
        mask_fitted_planet_detrended,
    )


def split_around_transits(
    x, y, yerr, mask, mask_fitted_planet, t0s, window, period, model="None"
):
    # x = time
    # y = flux
    # yerr = flux error
    # mask = mask
    # t0s = midtransits in data
    # window = what fraction of the period to plot on either side of transit (ie. xlim=1/2 means 1/2 period on either side)
    # period = planet period to define plotting limit

    xlims = []
    x_split = []
    y_split = []
    yerr_split = []
    mask_split = []
    mask_fitted_planet_split = []

    if type(model) is np.ndarray:
        model_split = []

    for t0 in t0s:

        xlims.append([float(t0 - (period * window)), float(t0 + (period * window))])
        x_split.append([])
        y_split.append([])
        yerr_split.append([])
        mask_split.append([])
        mask_fitted_planet_split.append([])

        if type(model) is np.ndarray:
            model_split.append([])

    split_index = 0
    for xlim in xlims:
        for ii in range(0, len(x)):
            time = x[ii]
            lc = y[ii]
            lc_err = yerr[ii]
            transit_mask = mask[ii]
            fitted_planet_mask = mask_fitted_planet[ii]

            if type(model) is np.ndarray:
                model_val = model[ii]

            if time >= xlim[0] and time <= xlim[1]:
                x_split[split_index].append(time)
                y_split[split_index].append(lc)
                yerr_split[split_index].append(lc_err)
                mask_split[split_index].append(transit_mask)
                mask_fitted_planet_split[split_index].append(fitted_planet_mask)

                if type(model) is np.ndarray:
                    model_split[split_index].append(model_val)

        x_split[split_index] = np.array(x_split[split_index])
        y_split[split_index] = np.array(y_split[split_index])
        yerr_split[split_index] = np.array(yerr_split[split_index])
        mask_split[split_index] = np.array(mask_split[split_index])
        mask_fitted_planet_split[split_index] = np.array(
            mask_fitted_planet_split[split_index]
        )

        split_index += 1

    x_split = np.array(x_split, dtype=object)
    y_split = np.array(y_split, dtype=object)
    yerr_split = np.array(yerr_split, dtype=object)
    mask_split = np.array(mask_split, dtype=object)
    mask_fitted_planet_split = np.array(mask_fitted_planet_split, dtype=object)

    if type(model) is np.ndarray:
        model_split = np.array(model_split, dtype=object)
        return (
            x_split,
            y_split,
            yerr_split,
            mask_split,
            mask_fitted_planet_split,
            model_split,
        )

    return x_split, y_split, yerr_split, mask_split, mask_fitted_planet_split


def find_quarters_with_transits(
    x_quarters,
    y_quarters,
    yerr_quarters,
    mask_quarters,
    mask_fitted_planet_quarters,
    t0s,
):
    x_transits = []
    y_transits = []
    yerr_transits = []
    mask_transits = []
    mask_fitted_planet_transits = []

    quarters_included = []
    for ii in range(0, len(x_quarters)):
        x_quarter = x_quarters[ii]
        y_quarter = y_quarters[ii]
        yerr_quarter = yerr_quarters[ii]
        mask_quarter = mask_quarters[ii]
        mask_fitted_planet_quarter = mask_fitted_planet_quarters[ii]

        xmin = np.min(x_quarter)
        xmax = np.max(x_quarter)

        for t0 in t0s:
            if t0 > xmin and t0 < xmax:

                # make sure this quarter hasn't already been added to the data
                # this ensures there aren't duplicates if multiple transits
                # exist in a single quarter or sector
                if ii not in quarters_included:
                    quarters_included.append(ii)
                    x_transits.append(x_quarter.astype("float64"))
                    y_transits.append(y_quarter.astype("float64"))
                    yerr_transits.append(yerr_quarter.astype("float64"))
                    mask_transits.append(mask_quarter.astype("bool"))
                    mask_fitted_planet_transits.append(
                        mask_fitted_planet_quarter.astype("bool")
                    )

    return (
        x_transits,
        y_transits,
        yerr_transits,
        mask_transits,
        mask_fitted_planet_transits,
    )
