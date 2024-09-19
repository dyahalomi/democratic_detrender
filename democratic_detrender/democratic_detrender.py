from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from matplotlib.widgets import Slider, Button
import sys, argparse
import os
import warnings
import ast

warnings.simplefilter("ignore", np.RankWarning)

parser = argparse.ArgumentParser(
    description="Looks up light curves for TESS and Kepler objects and enables labeling of jump times.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "object", type=str, help='TESS or Kepler identifier. ex: "toi-2088"'
)
parser.add_argument(
    "--flux_type",
    type=str,
    default="both",
    help="Flux type as a string. Options: pdc, sap, both, or qlp.",
)
parser.add_argument(
    "--mission", type=str, default="TESS", help='Mission data select. ex: "TESS"'
)
parser.add_argument(
    "--planet_num", type=int, default=1, help="Which planet to look at in system. ex: 1"
)
parser.add_argument(
    "--save_to_dir",
    type=str,
    default="./",
    help="Directory path to save csvs and figures to.",
)
parser.add_argument(
    "-d",
    "--depth",
    default=0.01,
    help="Sets depth of detrended plots. Default is 0.02.",
)
parser.add_argument(
    "-p",
    "--period",
    default=None,
    help="Optionally input period. Otherwise defaults to what \
    Exoplanet Archive can find.",
)
parser.add_argument(
    "-t",
    "--t0",
    default=None,
    help="Optionally input t0. Otherwise defaults to what Exoplanet Archive can find.",
)
parser.add_argument(
    "-du",
    "--duration",
    default=None,
    help="Optionally input duration. Otherwise defaults to what \
    Exoplanet Archive can find.",
)
parser.add_argument(
    "-mw", "--mask_width", default=1.3, help="Sets mask width. Default is 1.3."
)
parser.add_argument(
    "-s",
    "--show_plots",
    default="True",
    help="Set whether to show non-problem-time plots.",
)
parser.add_argument(
    "--dont_bin",
    default="False",
    help="if True, then jump time plots data wont be binned.",
)
parser.add_argument(
    "--save_transit_data",
    default="False",
    help="if True, then will save data on each transit",
)
parser.add_argument(
    "--polyAM", default="True", help="detrend via polyAM...True or False"
)
parser.add_argument(
    "--CoFiAM", default="True", help="detrend via CoFiAM...True or False"
)
parser.add_argument("--local", default="True", help="detrend via local...True or False")
parser.add_argument("--GP", default="True", help="detrend via GP...True or False")
parser.add_argument(
    "--use_sap_problem_times",
    default="False",
    help="use SAP problem times for PDC. True or False.",
)
parser.add_argument(
    "--no_pdc_problem_times", default="True", help="assume PDC needs no problem times."
)
parser.add_argument(
    "--user_light_curve",
    type=str,
    default='NO',
    help="path to directory with user input light curve and metadata.",
)


args = vars(parser.parse_args())

# user input parameters

input_id = args["object"]
flux_type = args["flux_type"]
mission_select = args["mission"]
input_planet_number = int(args["planet_num"])
input_dir = args["save_to_dir"]
input_depth = float(args["depth"])
input_period = args["period"]
input_t0 = args["t0"]
input_duration = args["duration"]
input_mask_width = float(args["mask_width"])
input_show_plots = ast.literal_eval(args["show_plots"])
input_dont_bin = ast.literal_eval(args["dont_bin"])
input_save_transit_data = ast.literal_eval(args["save_transit_data"])
input_polyAM = args["polyAM"]
input_CoFiAM = args["CoFiAM"]
input_GP = args["GP"]
input_local = args["local"]
input_use_sap_problem_times = args["use_sap_problem_times"]
input_no_pdc_problem_times = args["no_pdc_problem_times"]
input_user_light_curve = args["user_light_curve"]


input_detrend_methods = []
if input_GP == "True":
    input_detrend_methods.append("GP")
if input_CoFiAM == "True":
    input_detrend_methods.append("CoFiAM")
if input_polyAM == "True":
    input_detrend_methods.append("polyAM")
if input_local == "True":
    input_detrend_methods.append("local")

if input_use_sap_problem_times == "True":
    input_problem_times_default = "use_sap"
else:
    input_problem_times_default = None


if input_no_pdc_problem_times == "True":
    input_no_pdc_problem_times = True
else:
    input_no_pdc_problem_times = False



# # # ---------------------------------------------- now the fun begins ! ! ! ------------------------------------------------ # # #


# print(f"exoplanet.__version__ = '{xo.__version__}'")

from find_flux_jumps import *
from get_lc import *
from helper_functions import *
from outlier_rejection import *
from manipulate_data import *
from plot import plot_detrended_lc
from detrend import *


# checks user arguments
if mission_select == "TESS":
    tess_bool = True
    kepler_bool = False
else:
    tess_bool = False  # assuming Kepler is selected
    kepler_bool = True


if input_dir == "./":

    # determining figname
    # determining today's date
    today = date.today()
    current_day = today.strftime("%B_%d_%Y")

    foldername = (
        input_id
        + "/"
        + input_id
        + ".0"
        + str(input_planet_number)
        + "/"
        + "detrending"
        + "/"
        + current_day
    )
    path = os.path.join(input_dir, foldername)

    os.makedirs(path, exist_ok=True)

else:

    path = input_dir


# check if we should run both pdc and sap flux
if flux_type == "both":

    """
    # check if detrended lc already exist
    detrendec_lc_saved = path + '/detrended.csv'


    if os.path.exists(detrendec_lc_saved):
        print('detrended lc for '+input_id+' planet number '+str(input_planet_number)+' found')
        detrended_df = pd.read_csv(detrendec_lc_saved)
        x_detrended = detrended_df['time']
        sap_detrend_sep_lc = [detrended_df['local SAP'], detrended_df['polyAM SAP'], detrended_df['GP SAP'], detrended_df['CoFiAM SAP']]
        pdc_detrend_sep_lc = [detrended_df['local PDCSAP'], detrended_df['polyAM PDCSAP'], detrended_df['GP PDCSAP'], detrended_df['CoFiAM PDCSAP']]
        yerr_detrended = detrended_df['yerr']
        mask_detrended = detrended_df['mask']



        #this just pulls in period and duration from exofop for plotting purposes
        _, _, _, _, _, pdc_t0s, pdc_period, pdc_duration, _, _, _ = \
        get_light_curve(input_id, 'pdcsap_flux', planet_number = input_planet_number, TESS=tess_bool, Kepler=kepler_bool)

    else:
    """
    # print('detrended lc for '+input_id+' planet number '+str(input_planet_number)+' not found')
    # pulls in light curve
    [
        [
            sap_x_epochs,
            sap_y_epochs,
            sap_yerr_epochs,
            sap_mask_epochs,
            sap_mask_fitted_planet_epochs,
            sap_problem_times,
            sap_t0s,
            sap_period,
            sap_duration,
            sap_cadence,
        ],
        [
            pdc_x_epochs,
            pdc_y_epochs,
            pdc_yerr_epochs,
            pdc_mask_epochs,
            pdc_mask_fitted_planet_epochs,
            pdc_problem_times,
            pdc_t0s,
            pdc_period,
            pdc_duration,
            pdc_cadence,
        ],
    ] = find_sap_and_pdc_flux_jumps(
        input_id,
        path + "/",
        show_plots=input_show_plots,
        TESS=tess_bool,
        Kepler=kepler_bool,
        planet_number=input_planet_number,
        user_periods=input_period,
        user_t0s=input_t0,
        user_durations=input_duration,
        mask_width=input_mask_width,
        dont_bin=input_dont_bin,
        problem_times_default=input_problem_times_default,
        no_pdc_problem_times=input_no_pdc_problem_times,
        user_light_curve=input_user_light_curve
    )

    # now for detrending!
    print("")
    print("")
    print("detrending now")
    print("--------------")
    print("")

    detrended_lc_all_vals = detrend_sap_and_pdc(
        sap_values=[
            sap_x_epochs,
            sap_y_epochs,
            sap_yerr_epochs,
            sap_mask_epochs,
            sap_mask_fitted_planet_epochs,
            sap_problem_times,
            sap_t0s,
            sap_period,
            sap_duration,
            sap_cadence,
        ],
        pdc_values=[
            pdc_x_epochs,
            pdc_y_epochs,
            pdc_yerr_epochs,
            pdc_mask_epochs,
            pdc_mask_fitted_planet_epochs,
            pdc_problem_times,
            pdc_t0s,
            pdc_period,
            pdc_duration,
            pdc_cadence,
        ],
        save_dir=path + "/",
        pop_out_plots=input_show_plots,
        detrend_methods=input_detrend_methods,
    )

    sap_detrend_methods_out = detrended_lc_all_vals[0]
    pdc_detrend_methods_out = detrended_lc_all_vals[1]

    # now to add nans, in order to make sure pdc and sap arrays are the same length
    # x_detrended,\
    # [sap_local_detrended2, sap_poly_detrended2, sap_cofiam_detrended2, sap_gp_detrended2],\
    # [pdc_local_detrended2, pdc_poly_detrended2, pdc_cofiam_detrended2, pdc_gp_detrended2],\
    # yerr_detrended, mask_detrended, mask_fitted_planet_detrended = \

    (
        x_detrended,
        sap_detrend_sep_lc,
        pdc_detrend_sep_lc,
        yerr_detrended,
        mask_detrended,
        mask_fitted_planet_detrended,
    ) = add_nans_for_missing_data(
        detrended_lc_all_vals[2][0],
        [
            detrended_lc_all_vals[2][5],
            detrended_lc_all_vals[2][8],
            detrended_lc_all_vals[2][11],
            detrended_lc_all_vals[2][14],
        ],
        detrended_lc_all_vals[2][2],
        detrended_lc_all_vals[2][3],
        detrended_lc_all_vals[2][4],
        detrended_lc_all_vals[3][0],
        [
            detrended_lc_all_vals[3][5],
            detrended_lc_all_vals[3][8],
            detrended_lc_all_vals[3][11],
            detrended_lc_all_vals[3][14],
        ],
        detrended_lc_all_vals[3][2],
        detrended_lc_all_vals[3][3],
        detrended_lc_all_vals[3][4],
    )

    ## now to plot and save data!!
    green2, green1 = "#355E3B", "#18A558"
    blue2, blue1 = "#000080", "#4682B4"
    purple2, purple1 = "#2E0854", "#9370DB"
    red2, red1 = "#770737", "#EC8B80"

    colors = [red1, red2, blue1, blue2, green1, green2, purple1, purple2]

    y_detrended = [
        sap_detrend_sep_lc[0],
        pdc_detrend_sep_lc[0],
        sap_detrend_sep_lc[1],
        pdc_detrend_sep_lc[1],
        sap_detrend_sep_lc[2],
        pdc_detrend_sep_lc[2],
        sap_detrend_sep_lc[3],
        pdc_detrend_sep_lc[3],
    ]

    detrend_label = [
        "local SAP",
        "local PDCSAP",
        "polyAM SAP",
        "polyAM PDCSAP",
        "GP SAP",
        "GP PDCSAP",
        "CoFiAM SAP",
        "CoFiAM PDCSAP",
    ]

    y_detrended = np.array(y_detrended)
    y_detrended_transpose = y_detrended.T

    method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
    MAD = median_abs_deviation(
        y_detrended_transpose, axis=1, scale=1 / 1.4826, nan_policy="omit"
    )

    yerr_detrended = np.sqrt(yerr_detrended.astype(float) ** 2 + MAD ** 2)

    # save detrend data as csv
    detrend_dict = {}

    detrend_dict["time"] = x_detrended
    detrend_dict["yerr"] = yerr_detrended
    detrend_dict["mask"] = mask_detrended
    detrend_dict["method marginalized"] = method_marg_detrended

    for ii in range(0, len(y_detrended)):
        detrend = y_detrended[ii]
        label = detrend_label[ii]
        detrend_dict[label] = detrend

    detrend_df = pd.DataFrame(detrend_dict)

    detrend_df.to_csv(path + "/" + "detrended.csv")

    # plot all detrended data
    plot_detrended_lc(
        x_detrended,
        y_detrended,
        detrend_label,
        pdc_t0s,
        float(6 * pdc_duration / (24.0)) / pdc_period,
        pdc_period,
        colors,
        pdc_duration,
        depth=input_depth,
        figname=path + "/" + "individual_detrended.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc(
        x_detrended,
        [method_marg_detrended],
        ["method marg"],
        pdc_t0s,
        float(6 * pdc_duration / (24.0)) / pdc_period,
        pdc_period,
        ["k"],
        pdc_duration,
        depth=input_depth,
        figname=path + "/" + "method_marg_detrended.pdf",
    )

    # plot binned phase folded lightcurve
    plot_phase_fold_lc(
        x_detrended,
        method_marg_detrended,
        pdc_period,
        pdc_t0s,
        20,
        figname=path + "/" + "phase_folded.pdf",
    )


# check if we should run just pdc
elif flux_type == "pdc":

    # pulls in light curve
    [
        pdc_x_epochs,
        pdc_y_epochs,
        pdc_yerr_epochs,
        pdc_mask_epochs,
        pdc_mask_fitted_planet_epochs,
        pdc_problem_times,
        pdc_t0s,
        pdc_period,
        pdc_duration,
        pdc_cadence,
    ] = find_flux_jumps(
        input_id,
        "pdcsap_flux",
        path + "/",
        show_plots=input_show_plots,
        TESS=tess_bool,
        Kepler=kepler_bool,
        planet_number=input_planet_number,
        user_periods=input_period,
        user_t0s=input_t0,
        user_durations=input_duration,
        mask_width=input_mask_width,
        no_pdc_problem_times=input_no_pdc_problem_times,
        dont_bin=input_dont_bin,
        problem_times_default=input_problem_times_default,
        user_light_curve=input_user_light_curve
    )

    # now for detrending!
    print("")
    print("")
    print("detrending now")
    print("--------------")
    print("")

    detrended_lc_all_vals = detrend_one_lc(
        lc_values=[
            pdc_x_epochs,
            pdc_y_epochs,
            pdc_yerr_epochs,
            pdc_mask_epochs,
            pdc_mask_fitted_planet_epochs,
            pdc_problem_times,
            pdc_t0s,
            pdc_period,
            pdc_duration,
            pdc_cadence,
        ],
        save_dir=path + "/",
        pop_out_plots=input_show_plots,
        detrend_methods=input_detrend_methods,
    )

    [
        pdc_local_x,
        pdc_local_y,
        pdc_local_yerr,
        pdc_local_mask,
        pdc_local_mask_fitted_planet,
        pdc_local_detrended,
        pdc_local_x_no_outliers,
        pdc_local_detrended_no_outliers,
        pdc_poly_detrended,
        pdc_poly_x_no_outliers,
        pdc_poly_detrended_no_outliers,
        pdc_gp_detrended,
        pdc_gp_x_no_outliers,
        pdc_gp_detrended_no_outliers,
        pdc_cofiam_detrended,
        pdc_cofiam_x_no_outliers,
        pdc_cofiam_detrended_no_outliers,
    ] = detrended_lc_all_vals[1]

    x_detrended = pdc_local_x

    ## now to plot and save data!

    green2, green1 = "#355E3B", "#18A558"
    blue2, blue1 = "#000080", "#4682B4"
    purple2, purple1 = "#2E0854", "#9370DB"
    red2, red1 = "#770737", "#EC8B80"

    colors = [red2, blue2, green2, purple2]

    y_detrended = [
        pdc_local_detrended,
        pdc_poly_detrended,
        pdc_gp_detrended,
        pdc_cofiam_detrended,
    ]

    yerr_detrended = pdc_local_yerr
    mask_detrended = pdc_local_mask

    detrend_label = ["local PDCSAP", "polyAM PDCSAP", "GP PDCSAP", "CoFiAM PDCSAP"]

    y_detrended = np.array(y_detrended)
    y_detrended_transpose = y_detrended.T

    method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
    MAD = median_abs_deviation(
        y_detrended_transpose, axis=1, scale=1 / 1.4826, nan_policy="omit"
    )

    yerr_detrended = np.sqrt(yerr_detrended.astype(float) ** 2 + MAD ** 2)

    # save detrend data as csv
    detrend_dict = {}

    detrend_dict["time"] = x_detrended
    detrend_dict["yerr"] = yerr_detrended
    detrend_dict["mask"] = mask_detrended
    detrend_dict["method marginalized"] = method_marg_detrended

    for ii in range(0, len(y_detrended)):
        detrend = y_detrended[ii]
        label = detrend_label[ii]
        detrend_dict[label] = detrend

    detrend_df = pd.DataFrame(detrend_dict)

    detrend_df.to_csv(path + "/" + "detrended_PDC.csv")

    # plot all detrended data
    plot_detrended_lc(
        x_detrended,
        y_detrended,
        detrend_label,
        pdc_t0s,
        float(6 * pdc_duration / (24.0)) / pdc_period,
        pdc_period,
        colors,
        pdc_duration,
        depth=input_depth,
        figname=path + "/" + "individual_detrended_PDC.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc(
        x_detrended,
        [method_marg_detrended],
        ["method marg"],
        pdc_t0s,
        float(6 * pdc_duration / (24.0)) / pdc_period,
        pdc_period,
        ["k"],
        pdc_duration,
        depth=input_depth,
        figname=path + "/" + "method_marg_detrended_PDC.pdf",
    )

    # plot binned phase folded lightcurve
    plot_phase_fold_lc(
        x_detrended,
        method_marg_detrended,
        pdc_period,
        pdc_t0s,
        20,
        figname=path + "/" + "phase_folded_PDC.pdf",
    )


# check if we should run just sap
elif flux_type == "sap":

    # pulls in light curve
    [
        sap_x_epochs,
        sap_y_epochs,
        sap_yerr_epochs,
        sap_mask_epochs,
        sap_mask_fitted_planet_epochs,
        sap_problem_times,
        sap_t0s,
        sap_period,
        sap_duration,
        sap_cadence,
    ] = find_flux_jumps(
        input_id,
        "sap_flux",
        path + "/",
        show_plots=input_show_plots,
        TESS=tess_bool,
        Kepler=kepler_bool,
        planet_number=input_planet_number,
        user_periods=input_period,
        user_t0s=input_t0,
        user_durations=input_duration,
        mask_width=input_mask_width,
        dont_bin=input_dont_bin,
        user_light_curve=input_user_light_curve
    )

    # now for detrending!
    print("")
    print("")
    print("detrending now")
    print("--------------")
    print("")

    detrended_lc_all_vals = detrend_one_lc(
        lc_values=[
            sap_x_epochs,
            sap_y_epochs,
            sap_yerr_epochs,
            sap_mask_epochs,
            sap_mask_fitted_planet_epochs,
            sap_problem_times,
            sap_t0s,
            sap_period,
            sap_duration,
            sap_cadence,
        ],
        save_dir=path + "/",
        pop_out_plots=input_show_plots,
        detrend_methods=input_detrend_methods,
    )

    [
        sap_local_x,
        sap_local_y,
        sap_local_yerr,
        sap_local_mask,
        sap_local_mask_fitted_planet,
        sap_local_detrended,
        sap_local_x_no_outliers,
        sap_local_detrended_no_outliers,
        sap_poly_detrended,
        sap_poly_x_no_outliers,
        sap_poly_detrended_no_outliers,
        sap_gp_detrended,
        sap_gp_x_no_outliers,
        sap_gp_detrended_no_outliers,
        sap_cofiam_detrended,
        sap_cofiam_x_no_outliers,
        sap_cofiam_detrended_no_outliers,
    ] = detrended_lc_all_vals[1]

    x_detrended = sap_local_x

    # # now to plot!

    green2, green1 = "#355E3B", "#18A558"
    blue2, blue1 = "#000080", "#4682B4"
    purple2, purple1 = "#2E0854", "#9370DB"
    red2, red1 = "#770737", "#EC8B80"

    colors = [red1, blue1, green1, purple1]

    y_detrended = [
        sap_local_detrended,
        sap_poly_detrended,
        sap_gp_detrended,
        sap_cofiam_detrended,
    ]

    yerr_detrended = sap_local_yerr
    mask_detrended = sap_local_mask

    detrend_label = ["local SAP", "polyAM SAP", "GP SAP", "CoFiAM SAP"]

    y_detrended = np.array(y_detrended)
    y_detrended_transpose = y_detrended.T

    method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
    MAD = median_abs_deviation(
        y_detrended_transpose, axis=1, scale=1 / 1.4826, nan_policy="omit"
    )

    yerr_detrended = np.sqrt(yerr_detrended.astype(float) ** 2 + MAD ** 2)

    # saving detrend data as csv
    detrend_dict = {}

    detrend_dict["time"] = x_detrended
    detrend_dict["yerr"] = yerr_detrended
    detrend_dict["mask"] = mask_detrended
    detrend_dict["method marginalized"] = method_marg_detrended

    for ii in range(0, len(y_detrended)):
        detrend = y_detrended[ii]
        label = detrend_label[ii]
        detrend_dict[label] = detrend

    detrend_df = pd.DataFrame(detrend_dict)

    detrend_df.to_csv(path + "/" + "detrended_SAP.csv")

    # plot all detrended data
    plot_detrended_lc(
        x_detrended,
        y_detrended,
        detrend_label,
        sap_t0s,
        float(6 * sap_duration / (24.0)) / sap_period,
        sap_period,
        colors,
        sap_duration,
        depth=input_depth,
        figname=path + "/" + "individual_detrended_SAP.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc(
        x_detrended,
        [method_marg_detrended],
        ["method marg"],
        sap_t0s,
        float(6 * sap_duration / (24.0)) / sap_period,
        sap_period,
        ["k"],
        sap_duration,
        depth=input_depth,
        figname=path + "/" + "method_marg_detrended_SAP.pdf",
    )

    # plot binned phase folded lightcurve
    plot_phase_fold_lc(
        x_detrended,
        method_marg_detrended,
        sap_period,
        sap_t0s,
        20,
        figname=path + "/" + "phase_folded_SAP.pdf",
    )


else:
    print("ERROR!")
    print("invalid flux_type value entered...options are: pdc, sap, or both")
