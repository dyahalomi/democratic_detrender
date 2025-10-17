from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import os
import warnings

#surpress np.RankWarning
warnings.simplefilter("ignore", np.RankWarning)

from democratic_detrender.find_flux_jumps import find_flux_jumps, find_sap_and_pdc_flux_jumps
from democratic_detrender.get_lc import get_light_curve
from democratic_detrender.manipulate_data import add_nans_for_missing_data
from democratic_detrender.plot import plot_detrended_lc, plot_phase_fold_lc
from democratic_detrender.detrend import detrend_sap_and_pdc, detrend_one_lc
from democratic_detrender.dw_rejection_functions import reject_via_DW, dw_rejection_plots
from democratic_detrender.binning_rejection_functions import reject_via_binning, binning_rejection_plots
from democratic_detrender.method_rejection_functions import ensemble_step, merge_epochs, reject_epochs_by_white_noise_tests

def democratic_all(
    input_id, mission, flux_type='both', input_planet_number=1, input_dir='./',
    input_depth=0.01, input_period=None, input_t0=None, input_duration=None, input_mask_width=1.1, 
    input_show_plots=False, input_dont_bin=False, input_use_sap_problem_times=False, 
    input_no_pdc_problem_times=True, input_user_light_curve=None,
    input_polyAM=True, input_CoFiAM=True, input_GP=True, input_local=True):


    # which detrending methods will we use?
    input_detrend_methods = []
    if input_GP:
         input_detrend_methods.append("GP")
    if input_CoFiAM:
         input_detrend_methods.append("CoFiAM")
    if input_polyAM:
         input_detrend_methods.append("polyAM")
    if input_local:
         input_detrend_methods.append("local")


    #determine how to handle PDC if no problem times identified
    if input_use_sap_problem_times:
        input_problem_times_default = "use_sap"
    else:
        input_problem_times_default = None


    # checks mission input arguments
    if mission == "TESS":
        tess_bool = True
        kepler_bool = False
    elif mission == "Kepler":
        tess_bool = False  
        kepler_bool = True
    else:
        print('ERROR, CURRENTLY FULL DEMOCRATIC_DETREND FUNCTION ONLY SETUP TO RUN FOR KEPLER OR TESS LCs')
        print('SEE TUTORIAL FOR LOADING YOUR OWN LCs INTO THE DETRENDER')
        return None 



    # determine the path to directory to load and save files
    if input_dir == "./":
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
            #pulls in light curve
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

            # now add nans, in order to make sure pdc and sap arrays are the same length

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
                mask_width=input_mask_width
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
                mask_width=input_mask_width
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
            mask_width=input_mask_width
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
            mask_width=input_mask_width
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
            mask_width=input_mask_width
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
            mask_width=input_mask_width
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


    return detrend_df


def democratic_detrend(input_id, mission, flux_type='both', input_planet_number=1, input_dir='./',
    input_depth=0.01, input_period=None, input_t0=None, input_duration=None, input_mask_width=1.1, 
    input_show_plots=False, input_dont_bin=False, input_use_sap_problem_times=False, 
    input_no_pdc_problem_times=True, input_user_light_curve=None,
    input_polyAM=True, input_CoFiAM=True, input_GP=True, input_local=True):
    
    df = detrend_all(input_id, mission, flux_type, input_planet_number, input_dir,
    input_depth, input_period, input_t0, input_duration, input_mask_width, 
    input_show_plots, input_dont_bin, input_use_sap_problem_times, 
    input_no_pdc_problem_times, input_user_light_curve,
    input_polyAM, input_CoFiAM, input_GP, input_local)



    # Initialize sublists
    time_epochs = []
    y_epochs = []
    yerr_epochs = []

    # Temporary variables to hold data for current sublist
    time_temp = []
    y_temp = []
    yerr_temp = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if len(time_temp) == 0:  # If it's the first data point
            # Check if all values in the specified columns are not NaN for the current row
            if row[['local SAP', 'local PDCSAP', 
                    'polyAM SAP', 'polyAM PDCSAP', 
                    'GP SAP', 'GP PDCSAP', 
                    'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                
                time_temp.append(row['time'])
                y_temp.append(row[[
                    'local SAP', 'local PDCSAP', 
                    'polyAM SAP', 'polyAM PDCSAP', 
                    'GP SAP', 'GP PDCSAP', 
                    'CoFiAM SAP', 'CoFiAM PDCSAP']])
                yerr_temp.append(row['yerr'])
        else:
            time_diff = row['time'] - time_temp[-1]
            if time_diff > 5:  # If there is a gap greater than 50 in time
                # Check if all values in the specified columns are not NaN for the current row
                if row[['local SAP', 'local PDCSAP', 
                        'polyAM SAP', 'polyAM PDCSAP', 
                        'GP SAP', 'GP PDCSAP', 
                        'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                    # Append current sublist to the main list
                    time_epochs.append(time_temp)
                    y_epochs.append(pd.DataFrame(y_temp))
                    yerr_epochs.append(yerr_temp)
                    # Reset temporary variables for the new sublist
                    time_temp = [row['time']]
                    y_temp = [row[[
                        'local SAP', 'local PDCSAP', 
                        'polyAM SAP', 'polyAM PDCSAP', 
                        'GP SAP', 'GP PDCSAP', 
                        'CoFiAM SAP', 'CoFiAM PDCSAP']]]
                    yerr_temp = [row['yerr']]
            else:
                # Check if all values in the specified columns are not NaN for the current row
                if row[['local SAP', 'local PDCSAP', 
                        'polyAM SAP', 'polyAM PDCSAP', 
                        'GP SAP', 'GP PDCSAP', 
                        'CoFiAM SAP', 'CoFiAM PDCSAP']].notna().all():
                    time_temp.append(row['time'])
                    y_temp.append(row[[
                        'local SAP', 'local PDCSAP', 
                        'polyAM SAP', 'polyAM PDCSAP', 
                        'GP SAP', 'GP PDCSAP', 
                        'CoFiAM SAP', 'CoFiAM PDCSAP']])
                    yerr_temp.append(row['yerr'])

    # Append the last sublist
    time_epochs.append(time_temp)
    y_epochs.append(pd.DataFrame(y_temp))
    yerr_epochs.append(yerr_temp)

    detrending_methods = ['local PDCSAP', 'polyAM PDCSAP','GP PDCSAP', 'CoFiAM PDCSAP']

    # START OF METHOD REJECTION TESTS!!!!
    method_reject_figpath = path + "/" + "method_rejection_figures/"
    
    # DW method rejection test
    dw_sigma_test, DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs = reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=10000)
    dw_rejection_plots(DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs, detrending_methods, method_reject_figpath)


    # binning vs. RMS method rejection test
    binning_sigma_test, beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs = reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=1000)
    binning_rejection_plots(beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs, detrending_methods, method_reject_figpath)


    # method rejection step
    y_epochs_post_rej = reject_epochs_by_white_noise_tests(y_epochs, dw_sigma_test, binning_sigma_test, detrending_methods)
    times_all_post_rej, y_all_post_rej, yerr_all_post_rej = merge_epochs(time_epochs, y_epochs_post_rej, yerr_epochs)
    detrend_df_post_rej = ensemble_step(times_all_post_rej, y_all_post_rej, yerr_all_post_rej, detrending_methods)


    ## now to plot and save data!!
    green2, green1 = '#355E3B', '#18A558'
    blue2, blue1 = '#000080', '#4682B4'
    purple2, purple1 = '#2E0854','#9370DB'
    red2, red1 = '#770737', '#EC8B80'


    colors = [red1, red2,
              blue1, blue2,
              green1, green2,
              purple1, purple2]

        
    # plot all detrended data
    plot_detrended_lc(times_all_post_rej, y_all_post_rej, detrending_methods,
                      t0s, float(6*duration)/period, period,
                      colors, duration*24., depth=0.01, mask_width=1,
                      figname = './individual_detrended_post_rejection.pdf')


    #save post method rejection as csv
    detrend_df_post_rej.to_csv(path + "/" + "detrended_post_method_rejection.csv")

    return detrend_df_post_rej


