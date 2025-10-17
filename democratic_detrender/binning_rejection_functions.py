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


def binning_rejection_plots(beta_dist_MC_epochs, beta_detrended_epochs, upper_bound_epochs, columns):

    green2, green1 = '#355E3B', '#18A558'
    blue2, blue1 = '#000080', '#4682B4'
    purple2, purple1 = '#2E0854','#9370DB'
    red2, red1 = '#770737', '#EC8B80'


    colors = [red1, red2,
              blue1, blue2,
              green1, green2,
              purple1, purple2]


    for ii in range(0, len(beta_dist_MC_epochs)):
        plt.figure(figsize=[9,7])
        plt.hist(beta_dist_MC_epochs[ii], bins=100, density=True, histtype='step', color='k', linewidth=1, alpha=.7, label = r'White Noise Monte Carlo $R^2$')

        for jj in range(0, len(beta_detrended_epochs[ii])):

            #if sigma_test[columns[jj]][ii]:
            plt.axvline(beta_detrended_epochs[ii][jj], color=colors[jj], lw=3, label=columns[jj]+ r' $\hat{\beta}$ = ' + str(np.round(beta_detrended_epochs[ii][jj], 2)))
            #else:
            #    plt.axvline(DWdetrend_epochs[ii][jj], color=colors[jj], lw=3, label=columns[jj]+' DW = ' + str(np.round(DWdetrend_epochs[ii][jj], 2)), ls='dotted')


        # Plot vertical lines at median, median ± 1 std dev, and median ± 2 std dev
        # plt.axvspan(0., upper_bound_epochs[ii], color='g', alpha=0.3)
        plt.axvspan(upper_bound_epochs[ii], 2, color='r', alpha=0.1)
        plt.text(1.5, 0.5, r'$\hat{\beta}$ $>$ 3$\sigma$ outlier', 
                 verticalalignment='bottom', horizontalalignment='left', fontsize=27, color='r')

        plt.title(r'Bootstrap MC $\hat{\beta}$ Distribution vs. Detrended $\hat{\beta}$s, epoch \#'+str(epochs[ii]+1), fontsize=23)
        plt.xlabel(r'$\hat{\beta}$ Statistic for RMS vs. Bin Size Test of Light Curve', fontsize=23)
        plt.ylabel('Density', fontsize=23)
        plt.legend(fontsize=13, loc=1)

        #plt.xlim(np.min(r_squared_MC_epochs[ii]), np.max(r_squared_MC_epochs[ii]))
        plt.xlim(0.3,2)

        plt.tight_layout()

        plt.savefig('./figures/reject_via_binning_epoch'+str(ii+1)+'.pdf')

        plt.show()

