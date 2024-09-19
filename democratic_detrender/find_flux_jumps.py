import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button
import ast



# print(f"exoplanet.__version__ = '{xo.__version__}'")

from .get_lc import *
from .helper_functions import *
from .outlier_rejection import *
from .manipulate_data import *
from .plot import *


def find_flux_jumps(
    star_id,
    flux_type,
    save_to_directory,
    show_plots,
    TESS=False,
    Kepler=False,
    user_periods=None,
    user_t0s=None,
    user_durations=None,
    planet_number=1,
    mask_width=1.3,
    no_pdc_problem_times=True,
    dont_bin=False,
    data_name=None,
    problem_times_default=None,
    user_light_curve='NO'
):

    """
    Allows the user to label discontinuities (ie flux jumps or problem times)
    in a light curve and return the necessary data for further analysis.

    Parameters:
    - star_id (str): Identifier for the star.
    - flux_type (str): Type of flux data (e.g., 'pdcsap_flux' or 'sap_flux').
    - save_to_directory (str): Directory path to save plots.
    - show_plots (bool): Whether to display plots.
    - TESS (bool, optional, default=False): Flag indicating whether TESS data is used.
    - Kepler (bool, optional, default=False): Flag indicating whether Kepler data is used.
    - user_periods (list, optional): List of user-defined planet periods.
    - user_t0s (list, optional): List of user-defined midtransit times.
    - user_durations (list, optional): List of user-defined transit durations.
    - planet_number (int, optional, default=1): Number of the planet being analyzed.
    - mask_width (float, optional, default=1.3): Width of the transit mask.
    - no_pdc_problem_times (bool, optional, default=True): Flag indicating whether PDC flux has no problem times.
    - dont_bin (bool, optional, default=False): Flag indicating whether to skip binning.
    - data_name (str, optional): Name of the data source.
    - problem_times_default (str, optional): Default problem times source.
    - user_light_curve (string, optional): path to folder with user light curve. Default: NO.


    Returns:
    Tuple containing the necessary data for further analysis:
    - x_epochs (array): Array of time values for each epoch.
    - y_epochs (array): Array of flux values for each epoch.
    - yerr_epochs (array): Array of flux error values for each epoch.
    - mask_epochs (array): Array of mask values for each epoch.
    - mask_fitted_planet_epochs (array): Array of mask values for each fitted planet epoch.
    - problem_times (list): List of jump times to trim structured noise.
    - t0s (list): List of midtransit times.
    - period (float): Planet period.
    - duration (float): Transit duration.
    - cadence (float): Cadence of observations.

    """

    if user_light_curve == 'NO':
        # pull in light curve
        (
            time,
            lc,
            lc_err,
            mask,
            mask_fitted_planet,
            t0s,
            period,
            duration,
            quarters,
            crowding,
            flux_fraction,
        ) = get_light_curve(
            star_id,
            flux_type,
            TESS,
            Kepler,
            user_periods,
            user_t0s,
            user_durations,
            planet_number,
            mask_width,
        )

    else:
        #pull in the user light curve and metadata
        # Load the CSV files into DataFrames
        print('LOADING USER LIGHT CURVE and METDATA FROM LOCAL DIRECTORY: ' + user_light_curve)
        lc_df = pd.read_csv(user_light_curve+'lc.csv')
        lc_metadata = pd.read_csv(user_light_curve+'lc_metadata.csv')
        orbital_data = pd.read_csv(user_light_curve+'orbital_data.csv')
        t0s_output = pd.read_csv(user_light_curve+'t0s_output.csv')


        time = lc_df['xs'].values
        lc = lc_df['ys'].values
        lc_err = lc_df['ys_err'].values
        mask = lc_df['mask'].values
        mask_fitted_planet = lc_df['mask_fitted_planet'].values
        t0s = t0s_output['t0s_in_data'].values
        period = orbital_data['period'].values[0]
        duration = orbital_data['duration'].values[0]
        quarters = list(lc_metadata['quarters'].apply(ast.literal_eval).values)
        crowding = list(lc_metadata['crowding'].values)
        flux_fraction = list(lc_metadata['flux_fraction'].values)

    # determine cadence of observation
    cadence = determine_cadence(time)

    # find end time of quarters
    quarters_end = [el[1] for el in quarters]

    (
        time_out,
        flux_out,
        flux_err_out,
        mask_out,
        mask_fitted_planet_out,
        moving_median,
    ) = reject_outliers_out_of_transit(
        time, lc, lc_err, mask, mask_fitted_planet, 30 * cadence, 4
    )

    plot_outliers(
        time,
        lc,
        time_out,
        flux_out,
        moving_median,
        quarters_end,
        save_to_directory + flux_type + "_" + "outliers.pdf",
        star_id,
    )
    if show_plots:
        plt.show()

    (
        x_quarters,
        y_quarters,
        yerr_quarters,
        mask_quarters,
        mask_fitted_planet_quarters,
    ) = split_around_problems(
        time_out, flux_out, flux_err_out, mask_out, mask_fitted_planet_out, quarters_end
    )

    plot_split_data(
        x_quarters,
        y_quarters,
        t0s,
        save_to_directory + flux_type + "_" + "quarters_split.pdf",
        star_id,
    )
    if show_plots:
        plt.show()

    (
        x_quarters_w_transits,
        y_quarters_w_transits,
        yerr_quarters_w_transits,
        mask_quarters_w_transits,
        mask_fitted_planet_quarters_w_transits,
    ) = find_quarters_with_transits(
        x_quarters,
        y_quarters,
        yerr_quarters,
        mask_quarters,
        mask_fitted_planet_quarters,
        t0s,
    )

    x_quarters_w_transits = np.concatenate(x_quarters_w_transits, axis=0, dtype=object)
    y_quarters_w_transits = np.concatenate(y_quarters_w_transits, axis=0, dtype=object)
    yerr_quarters_w_transits = np.concatenate(
        yerr_quarters_w_transits, axis=0, dtype=object
    )
    mask_quarters_w_transits = np.concatenate(
        mask_quarters_w_transits, axis=0, dtype=object
    )
    mask_fitted_planet_quarters_w_transits = np.concatenate(
        mask_fitted_planet_quarters_w_transits, axis=0, dtype=object
    )

    mask_quarters_w_transits = np.array(mask_quarters_w_transits, dtype=bool)
    mask_fitted_planet_quarters_w_transits = np.array(
        mask_fitted_planet_quarters_w_transits, dtype=bool
    )

    (
        x_transits,
        y_transits,
        yerr_transits,
        mask_transits,
        mask_fitted_planet_transits,
    ) = split_around_transits(
        x_quarters_w_transits,
        y_quarters_w_transits,
        yerr_quarters_w_transits,
        mask_quarters_w_transits,
        mask_fitted_planet_quarters_w_transits,
        t0s,
        1.0 / 2.0,
        period,
    )

    if len(mask_transits) == 1:
        mask_transits = np.array(mask_transits, dtype=bool)
        mask_fitted_planet_transits = np.array(mask_fitted_planet_transits, dtype=bool)

    x_epochs = np.concatenate(x_transits, axis=0, dtype=object)
    y_epochs = np.concatenate(y_transits, axis=0, dtype=object)
    yerr_epochs = np.concatenate(yerr_transits, axis=0, dtype=object)
    mask_epochs = np.concatenate(mask_transits, axis=0, dtype=object)
    mask_fitted_planet_epochs = np.concatenate(
        mask_fitted_planet_transits, axis=0, dtype=object
    )

    # check if problem times already exist
    if problem_times_default == "use_sap":
        print("using sap problem times for pdc also")
        problem_path = save_to_directory + "sap_flux_problem_times.txt"
    else:
        problem_path = save_to_directory + flux_type + "_problem_times.txt"

    if os.path.exists(problem_path):
        print(
            flux_type
            + " "
            + "problem times for "
            + star_id
            + " planet number "
            + str(planet_number)
            + " found"
        )
        with open(problem_path, "r") as problem_file:
            problem_times = json.load(problem_file)

    elif no_pdc_problem_times:
        if flux_type == "pdcsap_flux":
            print("assuming no pdc problem times")
            problem_times = []

        # if not, mark out problem times manually
        else:
            _, _, problem_times = plot_transits(
                x_transits,
                y_transits,
                mask_transits,
                t0s,
                period,
                cadence * 5,
                star_id,
                dont_bin=dont_bin,
                data_name=data_name,
            )
            # save problem times
            with open(problem_path, "w") as problem_file:
                json.dump(problem_times, problem_file)
            print(flux_type + " problem times saved as " + problem_path)

    # if not, mark out problem times manually for both pdc and sap
    else:
        _, _, problem_times = plot_transits(
            x_transits,
            y_transits,
            mask_transits,
            t0s,
            period,
            cadence * 5,
            star_id,
            dont_bin=dont_bin,
            data_name=data_name,
        )
        # save problem times
        with open(problem_path, "w") as problem_file:
            json.dump(problem_times, problem_file)
        print(flux_type + " problem times saved as " + problem_path)

    return (
        x_epochs,
        y_epochs,
        yerr_epochs,
        mask_epochs,
        mask_fitted_planet_epochs,
        problem_times,
        t0s,
        period,
        duration,
        cadence,
    )


def find_sap_and_pdc_flux_jumps(
    star_id,
    save_to_directory,
    show_plots,
    TESS=False,
    Kepler=False,
    user_periods=None,
    user_t0s=None,
    user_durations=None,
    planet_number=1,
    mask_width=1.3,
    dont_bin=False,
    data_name=None,
    problem_times_default=None,
    no_pdc_problem_times=True,
    user_light_curve='NO'
):

    """
    Wrapper to identify flux jumps in both SAP and PDC flux data and return the necessary data for further analysis.

    Parameters:
    - star_id (str): Identifier for the star.
    - save_to_directory (str): Directory path to save plots.
    - show_plots (bool): Whether to display plots.
    - TESS (bool, optional, default=False): Flag indicating whether TESS data is used.
    - Kepler (bool, optional, default=False): Flag indicating whether Kepler data is used.
    - user_periods (list, optional): List of user-defined planet periods.
    - user_t0s (list, optional): List of user-defined midtransit times.
    - user_durations (list, optional): List of user-defined transit durations.
    - planet_number (int, optional, default=1): Number of the planet being analyzed.
    - mask_width (float, optional, default=1.3): Width of the mask.
    - dont_bin (bool, optional, default=False): Flag indicating whether to skip binning.
    - data_name (str, optional): Name of the data source.
    - problem_times_default (str, optional): Default problem times source.
    - no_pdc_problem_times (bool, optional, default=True): Flag indicating whether PDC flux has no problem times.
    - user_light_curve (string, optional): path to folder with user light curve. Default: NO.


    Returns:
    Tuple containing the necessary data for further analysis for SAP flux:
    - x_epochs_sap (array): Array of time values for each SAP epoch.
    - y_epochs_sap (array): Array of flux values for each SAP epoch.
    - yerr_epochs_sap (array): Array of flux error values for each SAP epoch.
    - mask_epochs_sap (array): Array of mask values for each SAP epoch.
    - mask_fitted_planet_epochs_sap (array): Array of mask values for each fitted planet epoch in SAP flux.

    Tuple containing the necessary data for further analysis for PDC flux:
    - x_epochs_pdc (array): Array of time values for each PDC epoch.
    - y_epochs_pdc (array): Array of flux values for each PDC epoch.
    - yerr_epochs_pdc (array): Array of flux error values for each PDC epoch.
    - mask_epochs_pdc (array): Array of mask values for each PDC epoch.
    - mask_fitted_planet_epochs_pdc (array): Array of mask values for each fitted planet epoch in PDC flux.
    """

    sap_vals = find_flux_jumps(
        star_id,
        "sap_flux",
        save_to_directory,
        show_plots,
        TESS=TESS,
        Kepler=Kepler,
        user_periods=user_periods,
        user_t0s=user_t0s,
        user_durations=user_durations,
        planet_number=planet_number,
        mask_width=mask_width,
        dont_bin=dont_bin,
        data_name=data_name,
        problem_times_default=problem_times_default,
        no_pdc_problem_times=no_pdc_problem_times,
        user_light_curve=user_light_curve
    )

    pdc_vals = find_flux_jumps(
        star_id,
        "pdcsap_flux",
        save_to_directory,
        show_plots,
        TESS=TESS,
        Kepler=Kepler,
        user_periods=user_periods,
        user_t0s=user_t0s,
        user_durations=user_durations,
        planet_number=planet_number,
        mask_width=mask_width,
        dont_bin=dont_bin,
        data_name=data_name,
        problem_times_default=problem_times_default,
        no_pdc_problem_times=no_pdc_problem_times,
        user_light_curve=user_light_curve
    )

    return sap_vals, pdc_vals
