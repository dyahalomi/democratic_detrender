""" This module contains functions to plot light curves and detrending results. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.widgets import Slider, Button
from democratic_detrender.helper_functions import bin_data

import matplotlib
matplotlib.rc("xtick", labelsize=27)
matplotlib.rc("ytick", labelsize=27)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rc("text", usetex=True)

def plot_transit(
    xs_star,
    ys_star,
    xs_transit,
    ys_transit,
    t0,
    period,
    title,
    bin_window,
    object_id,
    problem_times_input=None,
    dont_bin=False,
):
    # xs_star = time not in transit
    # ys_star = flux not in transit
    # xs_transit = times in transit
    # ys_tranist = fluxed in transit
    # t0 = midtransit
    # period = planet period to define plotting limit

    global problem_times

    if problem_times_input == None:
        problem_times = []

    else:
        problem_times_input.sort()
        problem_times = problem_times_input

    window = 1.0 / 2.0

    fig, ax = plt.subplots(1, 1, figsize=[9, 6])
    plt.subplots_adjust(left=0.2, bottom=0.3, hspace=0.3)

    if not dont_bin:
        xstar_bin, ystar_bin = bin_data(xs_star, ys_star, bin_window)
        xtransit_bin, ytransit_bin = bin_data(xs_transit, ys_transit, bin_window)
        bin_colors = ["#00008B", "#DC143C"]

    xmin, xmax = t0 - (period * window)[0], t0 + (period * window)[0]

    ymin_transit, ymax_transit = np.nanmin(ys_transit), np.nanmax(ys_transit)
    ymin_star, ymax_star = np.nanmin(ys_star), np.nanmax(ys_star)
    ymin, ymax = np.nanmin([ymin_transit, ymin_star]), np.max([ymax_transit, ymax_star])

    if ymin > 0:
        ymin = ymin / 1.2
    else:
        ymin = ymin * 1.2

    if ymax > 0:
        ymax = ymax * 1.2
    else:
        ymax = ymax / 1.2

    t_init = 0

    y = np.arange(ymin, ymax, 0.000001)
    t = t_init * np.ones(np.shape(y))
    l = ax.plot(t, y, lw=2, color="k")[0]

    ax.plot(xs_star, ys_star, ".", color="grey", alpha=0.3)
    ax.plot(xs_transit, ys_transit, ".", color="black", alpha=0.3)

    if not dont_bin:
        ax.plot(xstar_bin, ystar_bin, "o", color=bin_colors[1], alpha=0.9, markersize=7)
        ax.plot(
            xtransit_bin,
            ytransit_bin,
            "o",
            color=bin_colors[0],
            alpha=0.9,
            markersize=7,
        )

    ax.text(xmin + (xmax - xmin) * 0.05, 0.7 * ymax, title, fontsize=27)

    ax.axvline(t0, linewidth=1, color="k", ls="dashed")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("intensity")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.ticklabel_format(useOffset=False)  # disable scientific notation
    ax.set_title(object_id, fontsize=27)

    axtime = plt.axes([0.197, 0.1, 0.702, 0.09])
    stime = Slider(
        axtime, "time", xmin, xmax, valinit=t_init, orientation="horizontal", color="k"
    )

    def update(val):
        l.set_xdata(val * np.ones(np.shape(y)))

        fig.canvas.draw_idle()

    stime.on_changed(update)

    save_time = plt.axes([0.8, 0.025, 0.13, 0.04])
    button = Button(save_time, "save time", color="0.97", hovercolor="0.79")

    def save(event):
        global problem_times
        if stime.val not in problem_times:
            problem_times.append(stime.val)
            problem_times.sort()

    button.on_clicked(save)

    return (stime, button, problem_times)


def plot_transits(
    x_transits,
    y_transits,
    mask_transits,
    t0s,
    period,
    bin_window,
    object_id,
    problem_times_input=None,
    dont_bin=False,
    data_name=None,
):
    # xs = times
    # ys = fluxes
    # mask = masks for transit
    # t0s = midtransits in data
    # period = planet period to define plotting limits
    plt.close("all")
    sliders, buttons, problem_times = [], [], []

    if len(t0s) != len(x_transits):
        print("ERROR, length of t0s doesn't match length of x_transits")

    for ii in range(0, len(t0s)):
        t0 = t0s[ii]
        xs = x_transits[ii]
        ys = y_transits[ii]
        mask = mask_transits[ii]
        title = "epoch " + str(ii + 1)

        slider, button, problem_times_epoch = plot_transit(
            xs[~mask],
            ys[~mask],
            xs[mask],
            ys[mask],
            t0,
            period,
            title,
            bin_window,
            object_id,
            problem_times_input=problem_times_input,
            dont_bin=dont_bin,
        )
        sliders.append(slider)
        buttons.append(button)
        problem_times.append(problem_times_epoch)

        if data_name is not None:
            # saving detrend data as csv
            detrend_dict = {}

            detrend_dict["time"] = xs
            detrend_dict["flux"] = ys
            detrend_dict["mask"] = mask

            detrend_df = pd.DataFrame(detrend_dict)

            print("saving data to " + data_name + str(ii + 1) + ".csv")

            detrend_df.to_csv(data_name + str(ii + 1) + ".csv")

        plt.show()

    # turn list of lists into a flattened list
    problem_times_flat = [item for sublist in problem_times for item in sublist]

    return sliders, buttons, problem_times_flat


def plot_detrended_lc(
    xs,
    ys,
    detrend_labels,
    t0s_in_data,
    window,
    period,
    colors,
    duration,
    mask_width=1.3,
    depth=None,
    figname=None,
    title=None,
):
    """
    inputs:
    x = times
    ys = [detrended light curves] of length N number of detrendings
    detrend_labels = [detrending type] of length N number of detrendings
    t0s_in_data = midtransits in data
    window = what fraction of the period to plot on either side of transit (ie. window=1/2 means 1/2 period on either side)
    period = planet period to define plotting limit
    colors = [colors] of length N number of detrendings
    figname = Name of file if you want to save figure
    
    return:
    None
    
    
    
    
    
    """
    import math

    transit_windows = []
    for t0 in t0s_in_data:
        transit_windows.append(
            [
                t0 - mask_width * duration / (2 * 24.0),
                t0 + mask_width * duration / (2 * 24.0),
            ]
        )

    n_transit = np.arange(0, len(t0s_in_data), 1)

    if len(t0s_in_data) > 1:

        if len(ys) == 1:
            fig, ax = plt.subplots(
                ncols=3,
                nrows=math.ceil(len(t0s_in_data) / 3),
                figsize=[27, len(t0s_in_data) * len(ys)],
                sharey=True,
            )
        else:
            fig, ax = plt.subplots(
                ncols=3,
                nrows=math.ceil(len(t0s_in_data) / 3),
                figsize=[27, len(t0s_in_data) * len(ys) / 4],
                sharey=True,
            )

    else:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[18, 13], sharey=True)

    y_detrend = []

    if not depth:
        depth = 0.01

    if len(t0s_in_data) == 1:
        t0 = t0s_in_data[0]
        detrend_offset = 0
        for detrend_index in range(0, len(ys)):

            y_detrend = ys[detrend_index]
            x = xs

            ax.plot(
                x,
                y_detrend + detrend_offset,
                "o",
                color=colors[detrend_index],
                alpha=0.63,
                markersize=9,
            )

            ax.text(
                t0 - (period * window) + 0.01,
                detrend_offset + 0.013,
                detrend_labels[detrend_index],
                color=colors[detrend_index],
                fontsize=13,
            )

            detrend_offset += depth

        [
            ax.axvline(transit[0], linewidth=1.8, color="k", alpha=0.79, ls="--")
            for transit in transit_windows
        ]
        [
            ax.axvline(transit[1], linewidth=1.8, color="k", alpha=0.79, ls="--")
            for transit in transit_windows
        ]
        ax.set_xlabel("time [KBJD]", fontsize=27)
        ax.set_ylabel("intensity", fontsize=27)
        ax.set_xlim(t0 - (period * window), t0 + (period * window))

        ax.set_ylim(-1.2 * depth, depth * len(ys))
        ax.tick_params(axis="x", rotation=45)
        ax.ticklabel_format(useOffset=False)

    elif len(t0s_in_data) < 6:
        for ii in range(0, len(t0s_in_data)):
            ax_ii = ax[ii]
            t0 = t0s_in_data[ii]

            detrend_offset = 0
            for detrend_index in range(0, len(ys)):

                y_detrend = ys[detrend_index]
                x = xs

                ax_ii.plot(
                    x,
                    y_detrend + detrend_offset,
                    "o",
                    color=colors[detrend_index],
                    alpha=0.63,
                )

                ax_ii.text(
                    t0 - (period * window) + 0.18,
                    detrend_offset + 0.0018,
                    detrend_labels[detrend_index],
                    color=colors[detrend_index],
                    fontsize=18,
                )

                detrend_offset += depth

            ax_ii.axvline(
                transit_windows[ii][0], linewidth=1.8, color="k", alpha=0.79, ls="--"
            )
            ax_ii.axvline(
                transit_windows[ii][1], linewidth=1.8, color="k", alpha=0.79, ls="--"
            )

            ax_ii.set_xlabel("time [KBJD]", fontsize=18)
            ax_ii.set_ylabel("intensity", fontsize=18)
            ax_ii.set_xlim(t0 - (period * window), t0 + (period * window))
            ax_ii.set_ylim(-1.2 * depth, depth * len(ys))
            ax_ii.tick_params(axis="x", rotation=45)

    else:
        column = 0
        row = 0

        for ii in range(0, len(t0s_in_data)):
            ax_ii = ax[row][column]

            t0 = t0s_in_data[ii]

            detrend_offset = 0
            for detrend_index in range(0, len(ys)):

                y_detrend = ys[detrend_index]
                x = xs

                ax_ii.plot(
                    x,
                    y_detrend + detrend_offset,
                    "o",
                    color=colors[detrend_index],
                    alpha=0.63,
                )

                ax_ii.text(
                    t0 - (period * window) + 0.18,
                    detrend_offset + 0.0018,
                    detrend_labels[detrend_index],
                    color=colors[detrend_index],
                    fontsize=18,
                )

                detrend_offset += depth

            ax_ii.axvline(
                transit_windows[ii][0], linewidth=1.8, color="k", alpha=0.79, ls="--"
            )
            ax_ii.axvline(
                transit_windows[ii][1], linewidth=1.8, color="k", alpha=0.79, ls="--"
            )

            ax_ii.set_xlabel("time [KBJD]", fontsize=18)
            ax_ii.set_ylabel("intensity", fontsize=18)
            ax_ii.set_xlim(t0 - (period * window), t0 + (period * window))
            ax_ii.set_ylim(-1.2 * depth, depth * len(ys))
            ax_ii.tick_params(axis="x", rotation=45)

            if column == 2:
                column = 0
                row += 1
            else:
                column += 1

    if title:
        fig.suptitle(title, fontsize=36, y=1.01)

    # fig.delaxes(ax[1][4])
    fig.tight_layout()

    if figname:
        fig.savefig(figname, bbox_inches="tight")

    return None


def plot_phase_fold_lc(time, lc, period, t0s, xlim, figname):

    plt.close("all")

    fig = plt.figure(figsize=[18, 6])
    x_fold = (time - t0s[0] + 0.5 * period) % period - 0.5 * period
    plt.scatter(x_fold, lc, c=time, s=10)
    plt.xlabel("time since transit [days]")
    plt.ylabel("intensity [ppm]")
    plt.colorbar(label="time [days]")
    _ = plt.xlim(0.0 - (period / xlim), 0.0 + (period / xlim))

    if figname:
        fig.savefig(figname)

    return None


def plot_outliers(
    time, flux, time_out, flux_out, moving_median, kepler_quarters, figname, object_id
):
    """
    input:
    -------
    time = array of time values 
    flux = array of flux values 
    time_out = array of time values without outliers
    flux_out = array of flux values without outliers
    """

    plt.close("all")
    outlier_times = []
    outlier_fluxes = []
    n_outliers = len(flux) - len(flux_out)

    outliers_count = 0
    for ii in range(0, len(time)):
        if time[ii] not in time_out:
            outliers_count += 1
            outlier_times.append(time[ii])
            outlier_fluxes.append(flux[ii])

    if outliers_count != n_outliers:
        print("ERROR, didn't find all outliers")

    fig, ax = plt.subplots(1, 1, figsize=[18, 9])

    ax.plot(time_out, flux_out, "o", color="grey", alpha=0.7)
    ax.plot(outlier_times, outlier_fluxes, "o", color="red", alpha=1.0)
    # ax.plot(time, moving_median, '.', color = 'k')
    [ax.axvline(_x, linewidth=1, color="k", ls="--") for _x in kepler_quarters]

    ax.set_xlabel("time [days]", fontsize=27)
    ax.set_ylabel("intensity", fontsize=27)
    ax.set_title(object_id, fontsize=36)

    fig.tight_layout()

    fig.savefig(figname)

    return None


def plot_split_data(x_split, y_split, t0s, figname, object_id):

    plt.close("all")
    fig, ax = plt.subplots(nrows=len(x_split), figsize=[18, 3 * len(x_split)])

    if len(x_split) > 1:
        for ii in range(0, len(x_split)):
            xmin = np.min(x_split[ii])
            xmax = np.max(x_split[ii])

            ax_ii = ax[ii]

            ax_ii.plot(x_split[ii], y_split[ii], "o", color="grey", alpha=0.7)
            [ax_ii.axvline(_x, linewidth=1, color="k", ls="--") for _x in t0s]

            ax_ii.text(
                xmin + (xmax - xmin) * 0.05,
                0.7 * np.max(y_split[ii]),
                "quarter " + str(ii),
                fontsize=27,
            )
            ax_ii.set_xlim(xmin, xmax)
            ax_ii.set_ylabel("intensity", fontsize=27)

        ax[len(x_split) - 1].set_xlabel("time [days]", fontsize=27)
    else:
        xmin = np.min(x_split[0])
        xmax = np.max(x_split[0])

        ax_ii = ax

        ax_ii.plot(x_split[0], y_split[0], "o", color="grey", alpha=0.7)
        [ax_ii.axvline(_x, linewidth=1, color="k", ls="--") for _x in t0s]

        ax_ii.text(xmin + (xmax - xmin) * 0.05, 0, "quarter " + str(0), fontsize=27)
        ax_ii.set_xlim(xmin, xmax)
        ax_ii.set_xlabel("time [days]", fontsize=27)
        ax_ii.set_ylabel("intensity", fontsize=27)

    fig.suptitle(object_id, fontsize=36)

    fig.tight_layout()
    fig.savefig(figname)

    return None


def plot_individual_outliers(
    time, flux, time_out, flux_out, t0s, period, window, depth, figname
):
    """
    input:
    -------
    time = array of time values 
    flux = array of flux values 
    time_out = array of time values without outliers
    flux_out = array of flux values without outliers
    period = planet period to define plotting limit 
    window = what fraction of the period to plot on either side of transit (ie. window=1/2 means 1/2 period on either side)
    t0s = midtransits in data
    """

    plt.close("all")
    outlier_times = []
    outlier_fluxes = []
    n_outliers = len(flux) - len(flux_out)

    outliers_count = 0
    for ii in range(0, len(time)):
        if time[ii] not in time_out:
            outliers_count += 1
            outlier_times.append(time[ii])
            outlier_fluxes.append(flux[ii])

    if outliers_count != n_outliers:
        print("ERROR, didn't find all outliers")

    nplots = 0
    epochs_with_outliers = []
    for ii in range(0, len(t0s)):
        outlier_in_this_epoch = False
        t0 = t0s[ii]
        xmin, xmax = t0 - (period * window), t0 + (period * window)

        for outlier_time in outlier_times:
            if outlier_time > xmin and outlier_time < xmax:
                if not outlier_in_this_epoch:
                    nplots += 1
                    outlier_in_this_epoch = True
                    epochs_with_outliers.append(ii)

    if nplots > 0:
        fig, ax = plt.subplots(nrows=nplots, figsize=[18, 6 * nplots])
    else:
        return None
        print(figname)

    # CHANGED TO FIX NOT PERMANENT MARCH 16,2024
    plot_num = -1
    for ii in range(0, len(t0s)):
        if ii in epochs_with_outliers:
            plot_num += 1
            t0 = t0s[ii]
            xmin, xmax = t0 - (period * window), t0 + (period * window)

            if nplots > 1:
                ax[plot_num].plot(time_out, flux_out, "o", color="grey", alpha=0.7)
                ax[plot_num].plot(
                    outlier_times, outlier_fluxes, "o", color="red", alpha=1.0
                )
                ax[plot_num].text(
                    xmin + (xmax - xmin) * 0.05,
                    -depth * 0.2,
                    "epoch " + str(ii + 1),
                    fontsize=27,
                )
                ax[plot_num].set_xlim(xmin, xmax)

                ax[plot_num].set_xlabel("time [days]")
                ax[plot_num].set_ylabel("intensity")

            else:
                ax.plot(time_out, flux_out, "o", color="grey", alpha=0.7)
                ax.plot(outlier_times, outlier_fluxes, "o", color="red", alpha=1.0)
                ax.text(
                    xmin + (xmax - xmin) * 0.05,
                    -depth * 0.2,
                    "epoch " + str(ii + 1),
                    fontsize=27,
                )
                ax.set_xlim(xmin, xmax)

                ax.set_xlabel("time [days]")
                ax.set_ylabel("intensity")

    fig.tight_layout()

    fig.savefig(figname)

    return None
