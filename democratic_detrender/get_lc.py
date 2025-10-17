""" This module contains functions to obtain light curve data and transit information from the NASA Exoplanet Archive. """

import numpy as np
import pandas as pd

from democratic_detrender.helper_functions import determine_cadence, find_nearest

import lightkurve as lk


def tic_id_from_simbad(other_id):
    """
    Queries Simbad to obtain the TIC ID corresponding to the given identifier.

    Note: This function requires the astroquery and astropy packages to be installed.

    Parameters:
        other_id (str): Identifier for which TIC ID needs to be obtained.

    Returns:
        str: TIC ID obtained from Simbad. Returns None if no TIC ID is found.

    Examples:
        >>> tic_id = tic_id_from_simbad('HD 12345')
        >>> print(tic_id)

    """

    # Import necessary libraries
    from astroquery.simbad import Simbad
    import astropy

    # Query Simbad to obtain the object IDs
    ID_table = Simbad.query_objectids(other_id)

    # Check if the result is a valid table
    if type(ID_table) is not astropy.table.table.Table:
        return None

    # Convert the 'ID' column to string
    ID_table["ID"] = ID_table["ID"].astype(str)

    # Convert the table to a Pandas DataFrame
    ID_pandas = ID_table.to_pandas()

    # Filter for TIC IDs
    tic_id = ID_pandas[ID_pandas["ID"].str.contains("TIC")]

    # Return the TIC ID, if found
    return tic_id["ID"].values[0]


def tic_id_from_exoplanet_archive(other_id):
    """
    Parameters:
        other_id (str): Identifier for which the TIC ID needs to be obtained.

    Returns:
        str: TIC ID obtained from the Exoplanet Archive. Returns None if no TIC ID is found.
    """

    # if SIMBAD can't get TIC ID, looks for it in exoplanet archive
    # most of this is from transit_info_from_exoplanet_archive tbh; hesitant to try to merge functions

    # SIMBAD should be able to grab data for already confirmed systems
    # this is primarily for TOI candidates

    # Exoplanet Archive URL for TOI data
    a = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=toipfx,tid,pl_tranmid,pl_orbper,pl_trandurh&format=csv"

    # Read TOI data from the Exoplanet Archive
    exoplanets = pd.read_csv(a)

    # Rename columns for consistency
    column_dict = {
        "toipfx": "toi_host",
        "tid": "tic_id",
        "pl_tranmid": "t0 [BJD]",
        "pl_orbper": "period [days]",
        "pl_trandurh": "duration [hours]",
    }

    exoplanets.rename(columns=column_dict, inplace=True)

    exoplanets["toi_host"] = "TOI-" + exoplanets["toi_host"].astype(str)
    exoplanets["tic_id"] = "TIC " + exoplanets["tic_id"].astype(str)

    # replacing any chars that don't match with ones that do; standardizes user input

    other_id = other_id.lstrip("toi- TOI")
    other_id = "TOI-" + other_id  # there's probably a smarter way to do this

    tic_id = exoplanets["tic_id"].loc[exoplanets["toi_host"] == other_id]

    return tic_id.values[0]


def transit_info_from_exoplanet_archive(tic_id):

    """
    Queries the Exoplanet Archive to obtain transit information (t0, period, and duration) for a given TIC ID.

    Parameters:
        tic_id (str): TIC ID for which transit information needs to be obtained.

    Returns:
        result: DataFrame containing transit information (t0, period, and duration) for the specified TIC ID.
        Returns an empty DataFrame if no information is found.

    """
    import pyvo as vo

    # Connect to the Exoplanet Archive TAP service
    service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")

    # Query the planetary comparison table for transit information
    a = service.search(
        "SELECT \
                       tic_id, pl_tranmid, pl_orbper, pl_trandur\
                       FROM pscomppars"
    )

    exoplanets = a.to_table()

    exoplanets = exoplanets.to_pandas()

    # Rename columns for consistency
    column_dict = {
        "pl_tranmid": "t0 [BJD]",
        "pl_orbper": "period [days]",
        "pl_trandur": "duration [hours]",
    }

    exoplanets.rename(columns=column_dict, inplace=True)

    # Filter for the specified TIC ID
    result = exoplanets[exoplanets["tic_id"] == tic_id]

    # if there's no row in the planetary comparison table, check TOI table
    if result.empty:
        print("Exoplanet Archive: TOI Table")
        print("----------------------------")
        a = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=tid,pl_tranmid,pl_orbper,pl_trandurh&format=csv"

        exoplanets = pd.read_csv(a)

        # Rename columns for consistency
        column_dict = {
            "tid": "tic_id",
            "pl_tranmid": "t0 [BJD]",
            "pl_orbper": "period [days]",
            "pl_trandurh": "duration [hours]",
        }

        exoplanets.rename(columns=column_dict, inplace=True)
        exoplanets["tic_id"] = "TIC " + exoplanets["tic_id"].astype(str)

        # Filter for the specified TIC ID in the TOI table
        result = exoplanets[exoplanets["tic_id"] == tic_id]

    else:
        print("Exoplanet Archive: Planet Comparison Table")
        print("------------------------------------------")

    return result


def get_transit_info(object_id):

    """
    Takes an object ID, queries Simbad to get the matching TIC ID, then queries the Exoplanet Archive
    to extract transit information including t0, period, and duration.
    
    if no Simbad match found, then returns None and prints error message
    if no exoplanet archive match found, then returns None and prints error message
    
    Parameters:
        object_id (str): Object ID for which transit information needs to be obtained.

    Returns:
        pandas.DataFrame or str or None:
            Returns a DataFrame containing transit information if found.
            Returns the TIC ID as a string if no transit information is found but a TIC ID is retrieved.
            Returns None if neither a TIC ID nor transit information is found.

    """

    tic_id = tic_id_from_simbad(object_id)

    if tic_id == None:
        # search on exoplanet archive
        print("No TIC ID match found on Simbad")
        print("Checking on exoplanet archive")

        tic_id = tic_id_from_exoplanet_archive(object_id)

        if tic_id == None:
            print("No TIC ID match found on exoplanet archive either.")
            return None

        else:
            print("Found " + tic_id)
            transit_info = transit_info_from_exoplanet_archive(tic_id)

            if transit_info.empty:
                print("No PC found with that TIC ID match on exoplanet archive")
                return tic_id

            else:
                return transit_info

    else:  # Simbad does find TIC ID
        transit_info = transit_info_from_exoplanet_archive(tic_id)

        if transit_info.empty:
            print("No PC found with that TIC ID match on exoplanet archive")
            return tic_id

        else:
            return transit_info


def get_light_curve(
    object_id,
    flux_type,
    TESS=False,
    Kepler=False,
    user_period=None,
    user_t0=None,
    user_duration=None,
    planet_number=1,
    mask_width=1.3,
    remove_PDCSAP_blend=True
):
    """
    Obtains light curve data based on the object ID, flux type, and optional user-provided transit parameters.

    Parameters:
        object_id (str): Object ID for which the light curve needs to be obtained.
        flux_type (str): Type of flux data to retrieve (e.g., 'sap_flux', 'pdcsap_flux').
        TESS (bool, optional): Whether the object is observed by TESS. Defaults to False.
        Kepler (bool, optional): Whether the object is observed by Kepler. Defaults to False.
        user_period (float, optional): User-provided period for the transit. Defaults to None.
        user_t0 (float, optional): User-provided transit midpoint. Defaults to None.
        user_duration (float, optional): User-provided transit duration. Defaults to None.
        planet_number (int, optional): Number of the planet in the system. Defaults to 1.
        mask_width (float, optional): Width multiplier for creating transit masks. Defaults to 1.3.
        remove_PDCSAP_blend (bool, optional). Whether to remove the assumed blend factor from PDCSAP data. Defaults to True.

    Returns:
        tuple: A tuple containing the light curve data:
            np.array: Time values (xs).
            np.array: Flux values (ys).
            np.array: Flux error values (ys_err).
            np.array: Transit mask (mask).
            np.array: Fitted planet mask (mask_fitted_planet).
            np.array: Transit midpoints (t0s_in_data).
            np.array: Transit periods (period).
            np.array: Transit durations (duration).
            list: Quarters of observation (quarters).
            list: Crowding information (crowding).
            list: Flux fraction information (flux_fraction).

    """

    transit_info = get_transit_info(object_id)

    if transit_info is None:
        print("no TIC ID found")
        return None

    # if transit_info is a string then we just returned tic id bc no planet info found
    if isinstance(transit_info, str):

        if user_period == None or user_t0 == None or user_duration == None:
            print(
                "no transit info found, so you must enter all 3 of period, t0, and duration"
            )
            return None

        else:
            transit_dic = {
                "tic_id": [transit_info],
                "t0 [BJD]": [float(user_t0)],
                "period [days]": [float(user_period)],
                "duration [hours]": [float(user_duration)],
            }

            transit_info = pd.DataFrame(transit_dic)

    print("NASA Exoplanet Archive planet parameters:")
    print("planet #, [    tic_id      ,    t0 [BJD]    ,  P [days]   , tdur [hrs]]")
    transit_info_list = transit_info.astype(str).values.tolist()
    for ii in range(0, len(transit_info_list)):
        print("planet " + str(ii + 1) + ", " + str(transit_info_list[ii]))

    tic_id = str(transit_info["tic_id"].values[0])

    if user_period != None:
        periods = np.array(transit_info["period [days]"].values, dtype=float)
        periods[planet_number - 1] = user_period

        print("using periods = " + str(periods))

    else:
        periods = np.array(transit_info["period [days]"].values, dtype=float)

    if user_t0 != None:
        t0s = np.array(transit_info["t0 [BJD]"].values, dtype=float)
        t0s[planet_number - 1] = user_t0

        print("using t0s = " + str(t0s))

    else:
        t0s = np.array(transit_info["t0 [BJD]"].values, dtype=float)

    if user_duration != None:
        durations = np.array(transit_info["duration [hours]"].values, dtype=float)
        durations[planet_number - 1] = user_duration

        print("using durations = " + str(durations))

    else:
        durations = np.array(transit_info["duration [hours]"].values, dtype=float)

    # if no duration values input, just assume a 2 hour duration
    for ii in range(0, len(durations)):
        if np.isnan(durations[ii]):
            print(
                "no duration information on exoplanet archive for the "
                + str(ii + 1)
                + "th planet"
            )
            print("assuming 2 hours for " + str(ii + 1) + "th planet duration!")
            durations[ii] = 2.0

    print("")
    print("")

    nplanets = len(periods)

    if TESS:
        # switch to TESS BJD
        t0s = t0s - 2457000

        if flux_type == "qlp":
            lc_files = lk.search_lightcurve(
                tic_id, mission="TESS", author="qlp"
            ).download_all(quality_bitmask="default")

        else:
            # pull in short cadence TESS SPOC LC
            lc_files_short_cadence = lk.search_lightcurve(
                tic_id, mission="TESS", author="SPOC", cadence="short"
            ).download_all(quality_bitmask="default", flux_column=flux_type)

            # pull in long cadence TESS SPOC LC
            lc_files_long_cadence = lk.search_lightcurve(
                tic_id, mission="TESS", author="SPOC", cadence="long"
            ).download_all(quality_bitmask="default", flux_column=flux_type)

            # use short cadence TESS data if if exists, else use long cadence
            if lc_files_short_cadence == []:
                lc_files = lc_files_long_cadence
            else:
                lc_files = lc_files_short_cadence

    if Kepler:
        # switch to Kepler BJD
        t0s = t0s - 2454833

        # pull in Kepler LC
        # QUICK FIX NOT LONG TERM USING LONG CADENCE DATA!!!!
        # REMOVE THIS CHANGE BEFORE PUSHING
        # ISSUE IS WITH OUTLIER REJECTION, BUT WILL LIKELY NEED TO MAKE OTHER SPEED UPS....
        # DYAHALOMI MARCH 16, 2024
        lc_files = lk.search_lightcurve(
            tic_id, mission="Kepler", author="Kepler", cadence="long"
        ).download_all(quality_bitmask="default", flux_column=flux_type)

    quarters = []
    crowding = []
    flux_fraction = []

    try:
        for file in lc_files:
            quarters.append([np.min(file.time.value), np.max(file.time.value)])

            if flux_type != "qlp":
                crowding.append(file.CROWDSAP)
                flux_fraction.append(file.FLFRCSAP)

    except TypeError:
        if TESS:
            mission = "TESS"
        else:
            mission = "Kepler"
        error_message = (
            "no "
            + mission
            + " "
            + flux_type
            + " data found for "
            + object_id
            + ", so the code will break..."
        )

        print("")
        print("")
        print("")
        print("")
        print(error_message)
        print("")
        print("")
        print("")
        print("")

        return None


    #remove blend factor from PDCSAP data
    if remove_PDCSAP_blend:

        if flux_type == 'pdcsap_flux':
            xs = []
            ys = []
            ys_err = []
            for ii in range(0, len(lc_files)):
                #normalize flux value so you can add the blend factors
                lc = lc_files[ii].normalize().remove_nans() 
                f = flux_fraction[ii]
                c = crowding[ii]

                # Retrieve flux for the individual light curve
                x = lc.time.value
                y = lc.flux.value
                y_err = lc.flux_err.value


                # Apply the transformation to ys (flux)
                y_transformed = y / c


                # Append the modified light curve to the list
                xs.extend(x)
                ys.extend(y_transformed)
                ys_err.extend(y_err)


            # make the lists np arrays
            xs = np.array(xs)
            ys = np.array(ys)
            ys_err = np.array(ys_err)


        else:
            lc = lc_files.stitch().remove_nans()

            xs = lc.time.value
            ys = lc.flux.value
            ys_err = lc.flux_err.value


    else:
        lc = lc_files.stitch().remove_nans()

        xs = lc.time.value
        ys = lc.flux
        ys_err = lc.flux_err


    #define lc_mask for mask determination 
    lc_mask = lc_files.stitch().remove_nans() 
    mask = np.zeros(np.shape(xs), dtype=bool)
    for ii in range(0, nplanets):
        masks = lc_mask.create_transit_mask(
            period=periods[ii],
            duration=durations[ii] / 24.0 * mask_width,
            transit_time=t0s[ii],
        )
        mask += masks

    mask_fitted_planet = lc_mask.create_transit_mask(
        period=periods[planet_number - 1],
        duration=durations[planet_number - 1] / 24.0 * mask_width,
        transit_time=t0s[planet_number - 1],
    )

    # save the period, duration, and t0 for the planet we are fitting for...
    period = np.array([periods[planet_number - 1]])
    t0 = t0s[planet_number - 1]
    duration = np.array([durations[planet_number - 1]])

    nan_values = []
    if np.isnan(period[0]):
        nan_values.append("period")
    if np.isnan(t0):
        nan_values.append("t0")
    if np.isnan(duration[0]):
        nan_values.append("duration")

    if nan_values != []:
        print("")
        print("")
        print("")
        print("")
        print(
            str(nan_values)
            + " input is (are) not a number(s), so the code will break..."
        )
        print("")
        print("")
        print("")
        print("")

        return None

    print("using the following params for the planet we are fitting")
    print("--------------------------------------------------------")

    if TESS:
        print("[  t0 [TESS BJD]  , P [days], tdur [hrs]")
    if Kepler:
        print("[ t0 [Kepler BJD] , P [days], tdur [hrs]")
    print("[" + str(t0) + ", " + str(period[0]) + ", " + str(duration[0]) + "]")

    min_time = xs.min()
    max_time = xs.max()
    t0s_all = []
    while t0 > min_time:
        t0 -= period[0]

    while t0 < max_time:
        t0s_all.append(t0)
        t0 += period[0]

    cadence = determine_cadence(xs)

    t0s_in_data = []
    for t0 in t0s_all:
        nearest_lc_time = find_nearest(xs, t0)

        # if there is a data point within the cadence (times some uncertainty lets say 3) of
        # expected transit then there should be transit data
        if np.abs(t0 - nearest_lc_time) < 3 * cadence:
            t0s_in_data.append(t0)

    print("")
    print(str(len(t0s_in_data)) + " transits (or epochs) in total")
    print("")

    mu = np.median(ys)
    ys = ys / mu - 1
    ys_err = ys_err / mu

    return (
        np.array(xs),
        np.array(ys),
        np.array(ys_err),
        mask,
        mask_fitted_planet,
        np.array(t0s_in_data),
        period,
        duration,
        quarters,
        crowding,
        flux_fraction,
    )
