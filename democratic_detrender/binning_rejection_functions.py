from democratic_detrender.helper_functions import determine_cadence


def binning_monte_carlo(x, y, yerr, niter=1000):
    
    
    beta_dist = []
    sigma_1 = np.std(y)
    for ii in range(0, niter):
        fake_residuals = np.random.normal(loc=0.0, 
                                          scale=yerr, 
                                          size=np.shape(yerr))


        bmin = 2
        bmax = int(len(x)/10) # Maximum bin size to try
        binsize = [0 for i in range(bmin,bmax+1)] # Initialize binsize array
        rms = [0 for i in range(bmin,bmax+1)] # Initialize rms array
        rmstheory = [0 for i in range(bmin,bmax+1)] # Initialize rmstheory array
        beta_n = [0 for i in range(bmin,bmax+1)] # Initialize beta_n array

        for b in range(bmin,bmax+1):
            # Create a binned time series of the data = "binned"
            imax = int( np.floor(len(fake_residuals)/b) ) # Length of the new binned array

            binned = [0 for i in range(imax)]      # Initialize this new array
            for i in range(imax):
                binned[i] = np.mean(fake_residuals[b*i+1:(i+1)*b])

            # Compute the r.m.s. of this binned time series
            rms[b-bmin] = np.std(binned)
            binsize[b-bmin] = b
            rmstheory[b-bmin] = (sigma_1*b**(-0.5)*(imax/(imax-1))**(0.5))
            beta_n[b-bmin] = rms[b-bmin]/rmstheory[b-bmin]


        rms = np.array(rms)
        binsize = np.array(binsize)
        rmstheory = np.array(rmstheory)
        beta = np.mean(np.array(beta_n))

        
        # add beta to beta_dist array
        beta_dist.append(beta)
    
    return beta_dist
    
    
    
    





def mask_transits(x, t0, duration):
    # Calculate the start and end times of the transit window
    transit_start = t0 - duration / 2.0
    transit_end = t0 + duration / 2.0
    
    # Create a boolean mask for the transit region
    mask = (x >= transit_start) & (x <= transit_end)
    
    return mask




def pad_with_nans(x, y, yerr):
    cadence = determine_cadence(x)
    
    # Determine the expected number of points based on cadence
    expected_points = int(np.ceil((x[-1] - x[0]) / cadence)) + 1
    
    
    # Create arrays to store padded data
    padded_x = np.full(expected_points, np.nan)
    padded_y = np.full(expected_points, np.nan)
    padded_yerr = np.full(expected_points, np.nan)
    

    
    # Find indices to insert observed data into the padded arrays
    indices = ((x - x[0]) / cadence).astype(int)
    
    # Insert observed data into padded arrays
    padded_x[indices] = x
    padded_y[indices] = y
    padded_yerr[indices] = yerr
    
    return padded_x, padded_y, padded_yerr







def binning_outlier_detection(x, y, yerr, beta_dist_MC, max_sigma=3):
    """
    test how much the DW statistic for a detrended LC is an outlier (in terms of sigma) 
    for a single epoch LC in time series data.

    Parameters:
    - x (array): Time values.
    - y (array): Detrended flux values assuming centered on zero.

    Returns:
    - sigma (float): what sigma value is the DW for a given epoch vs. white noise
    """
    
    
    
    bmin = 2
    bmax = int(len(x)/10) # Maximum bin size to try
    binsize = [0 for i in range(bmin,bmax+1)] # Initialize binsize array
    rms = [0 for i in range(bmin,bmax+1)] # Initialize rms array
    rmstheory = [0 for i in range(bmin,bmax+1)] # Initialize rmstheory array
    beta_n = [0 for i in range(bmin,bmax+1)] # Initialize beta_n array
    
    sigma_1 = np.std(y)
    
    for b in range(bmin,bmax+1):
        # Create a binned time series of the data = "binned"
        imax = int( np.floor(len(x)/b) ) # Length of the new binned array

        binned = [0 for i in range(imax)]      # Initialize this new array
        for i in range(imax):
            binned[i] = np.mean(y[b*i+1:(i+1)*b])

        # Compute the r.m.s. of this binned time series
        rms[b-bmin] = np.std(binned)
        binsize[b-bmin] = b
        rmstheory[b-bmin] = (sigma_1*b**(-0.5)*(imax/(imax-1))**(0.5))
        beta_n[b-bmin] = rms[b-bmin]/rmstheory[b-bmin]

    rms = np.array(rms)
    binsize = np.array(binsize)
    rmstheory = np.array(rmstheory)
    beta = np.mean(np.array(beta_n))
    
    
    
    
    # Calculate the error function term
    # determine how far from 50th percentile we allow for based on max sigma
    erf_term = 0.5 * erf(max_sigma / np.sqrt(2))

    # Calculate the quantile bounds
    #lower_bound = np.quantile(beta_dist_MC, 0.5 - erf_term)
    upper_bound = np.quantile(beta_dist_MC, 0.5 + erf_term)

    # Test whether r_squared is within the bounds
    is_within_bounds = beta <= upper_bound
    
    #print(beta)
    #print(beta_dist_MC)
    #print('')


        
    
    return is_within_bounds, beta, upper_bound




def reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter):
    print('')
    print("starting individual model rejection via binning vs. RMS test:")
    print("--------------------------------------------------")
    start_time = time.time()


    mask_epochs = []
    for ii in range(0, len(time_epochs)):
        mask = mask_transits(time_epochs[ii], t0s[ii], duration)
        mask_epochs.append(mask)

    beta_dist_MC_epochs = []    
    for ii in range(0, len(time_epochs)):
        mask = mask_epochs[ii]


        beta_dist_MC = binning_monte_carlo(
            np.array(time_epochs[ii])[~mask], 
            np.array(y_epochs[ii])[~mask], 
            np.array(yerr_epochs[ii])[~mask], 
            niter=niter)

        beta_dist_MC_epochs.append(beta_dist_MC)



    is_within_bounds_epochs = []
    beta_detrended_epochs = []
    upper_bound_epochs = []
    for ii, file in enumerate(time_epochs):
        mask = mask_epochs[ii]
        is_within_bounds_detrended = []
        beta_detrended = []
        upper_bound_detrended = []
        for jj, col in enumerate(y_epochs[ii].columns):
            binning_outlier_estimate = binning_outlier_detection(np.array(time_epochs[ii])[~mask], 
                                                                 np.array(y_epochs[ii][col])[~mask],
                                                                 np.array(yerr_epochs[ii])[~mask],
                                                                 beta_dist_MC_epochs[ii], 
                                                                 max_sigma=3)

            is_within_bounds_detrended.append(binning_outlier_estimate[0])
            beta_detrended.append(binning_outlier_estimate[1])
            upper_bound_detrended = binning_outlier_estimate[2] #this is the same for each method, only varies per epoch

        is_within_bounds_epochs.append(is_within_bounds_detrended)
        beta_detrended_epochs.append(beta_detrended)
        upper_bound_epochs.append(upper_bound_detrended)



    end_time = time.time()

    # Calculate the elapsed time
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")


    columns = ['local SAP', 'local PDCSAP', 'polyAM SAP', 'polyAM PDCSAP', 'GP SAP', 'GP PDCSAP', 'CoFiAM SAP', 'CoFiAM PDCSAP']
    sigma_test = pd.DataFrame(is_within_bounds_epochs, columns=columns)
    
    return sigma_test, beta_dist_MC_epochs, beta_detrended_epochs, upper_bound_epochs




