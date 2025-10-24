from democratic_detrender.helper_functions import determine_cadence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import erf



def binning_monte_carlo(x, y, yerr, niter=1000):
    """
    Vectorized MC for beta distribution (RMS vs. bin test).
    - Keeps your signature and semantics.
    - Uses bmin = 1, bmax = N//10.
    - Uses np.nanmedian across bin sizes.
    """
    yerr = np.asarray(yerr, dtype=float)
    N = yerr.shape[0]
    if N == 0:
        return []

    # Ensure at least one candidate bin size (avoids empty ranges for short series)
    bmax = max(1, N // 10)

    # Generate all MC draws at once: shape (niter, N)
    rng = np.random.default_rng()
    fake = rng.normal(0.0, yerr, size=(niter, N))
    # Per-realization unbinned std (σ1) — consistent with each realization
    sigma1 = np.nanstd(fake, axis=1)  # shape (niter,)

    betas_per_b = []

    # Loop over bin sizes (fast inner ops via reshape)
    for b in range(1, bmax + 1):
        imax = N // b  # number of bins
        if imax < 2:
            continue  # avoid undefined sqrt(imax/(imax-1)) for imax=1

        # Truncate and reshape to (niter, imax, b), then mean over bin members
        trunc = fake[:, :imax * b]
        bins = trunc.reshape(niter, imax, b)
        binned_means = np.nanmean(bins, axis=2)        # (niter, imax)
        rms = np.nanstd(binned_means, axis=1)          # (niter,)

        # White-noise theoretical RMS
        rmstheory = sigma1 / np.sqrt(b) * np.sqrt(imax / (imax - 1))
        beta_n = rms / rmstheory                        # (niter,)
        betas_per_b.append(beta_n)

    if not betas_per_b:
        # No valid bin sizes (e.g., ultra-short segments)
        return [np.nan] * niter

    # Stack over bin sizes → (niter, n_b) and take nanmedian over bin sizes
    betas_per_b = np.stack(betas_per_b, axis=1)
    beta = np.nanmedian(betas_per_b, axis=1)           # (niter,)

    # Return as a plain list to match your downstream usage
    return beta.tolist()

    
    
    





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
    Compute beta for the real series and compare to a one-sided Gaussian cutoff
    derived from the MC beta distribution.
    - Always bmin = 1, bmax = N//10
    - Uses np.nanmedian across bin sizes
    - One-sided cutoff via erf (e.g., 3σ ⇒ 99.865th percentile)
    """
    y = np.asarray(y, dtype=float)
    N = y.shape[0]
    if N == 0:
        return False, np.nan, np.nan

    bmax = max(1, N // 10)
    sigma_1 = np.nanstd(y)

    beta_n_list = []
    for b in range(1, bmax + 1):
        imax = N // b
        if imax < 2:
            continue

        trunc = y[:imax * b]
        bins = trunc.reshape(imax, b)
        binned_means = np.nanmean(bins, axis=1)
        rms = np.nanstd(binned_means)
        rmstheory = sigma_1 / np.sqrt(b) * np.sqrt(imax / (imax - 1))
        beta_n_list.append(rms / rmstheory)

    if not beta_n_list:
        beta = np.nan
    else:
        beta = np.nanmedian(np.array(beta_n_list))

    # One-sided N-sigma percentile via erf (e.g., N=3 ⇒ q≈0.99865)
    erf_term = 0.5 * erf(max_sigma / np.sqrt(2.0))
    q = 0.5 + erf_term

    # Robust to empty/NaN MC input
    beta_dist_MC = np.asarray(beta_dist_MC, dtype=float)
    if beta_dist_MC.size == 0 or not np.isfinite(beta_dist_MC).any():
        upper_bound = np.nan
        is_within_bounds = False
    else:
        upper_bound = np.quantile(beta_dist_MC, q)
        is_within_bounds = (beta <= upper_bound)

    return is_within_bounds, beta, upper_bound





def reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000):
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


    columns = y_epochs[ii].columns
    sigma_test = pd.DataFrame(is_within_bounds_epochs, columns=columns)
    
    return sigma_test, beta_dist_MC_epochs, beta_detrended_epochs, upper_bound_epochs


def binning_rejection_plots(beta_dist_MC_epochs, beta_detrended_epochs, upper_bound_epochs, columns, figpath):

    green2, green1 = '#355E3B', '#18A558'
    blue2, blue1 = '#000080', '#4682B4'
    purple2, purple1 = '#2E0854','#9370DB'
    red2, red1 = '#770737', '#EC8B80'

    colors = [red1, red2,
              blue1, blue2,
              green1, green2,
              purple1, purple2]

    for ii in range(0, len(beta_dist_MC_epochs)):

        # --- skip plotting if there is no finite data (prevents ValueError: autodetected range of [nan, nan]) ---
        dist = np.asarray(beta_dist_MC_epochs[ii], dtype=float)
        if dist.size == 0 or not np.isfinite(dist).any():
            print(f"[binning_rejection_plots] Skipping epoch {ii+1}: no finite Monte Carlo data to plot.")
            continue

        plt.figure(figsize=[9,7])
        plt.hist(beta_dist_MC_epochs[ii], bins=100, density=True, histtype='step', color='k', linewidth=1, alpha=.7,
                 label = r'White Noise Monte Carlo $\hat{\beta}$')

        for jj in range(0, len(beta_detrended_epochs[ii])):

            #if sigma_test[columns[jj]][ii]:
            plt.axvline(beta_detrended_epochs[ii][jj], color=colors[jj], lw=3,
                        label=columns[jj]+ r' $\hat{\beta}$ = ' + str(np.round(beta_detrended_epochs[ii][jj], 2)))
            #else:
            #    plt.axvline(DWdetrend_epochs[ii][jj], color=colors[jj], lw=3,
            #                label=columns[jj]+' DW = ' + str(np.round(DWdetrend_epochs[ii][jj], 2)), ls='dotted')

        # Plot vertical lines at median, median ± 1 std dev, and median ± 2 std dev
        # plt.axvspan(0., upper_bound_epochs[ii], color='g', alpha=0.3)
        plt.axvspan(upper_bound_epochs[ii], 2, color='r', alpha=0.1)
        plt.text(1.5, 0.5, r'$\hat{\beta}$ $>$ 3$\sigma$ outlier', 
                 verticalalignment='bottom', horizontalalignment='left', fontsize=27, color='r')

        plt.title(r'Bootstrap MC $\hat{\beta}$ Distribution vs. Detrended $\hat{\beta}$s, epoch \#'+str(ii+1), fontsize=23)
        plt.xlabel(r'$\hat{\beta}$ Statistic for RMS vs. Bin Size Test of Light Curve', fontsize=23)
        plt.ylabel('Density', fontsize=23)
        plt.legend(fontsize=13, loc=1)

        #plt.xlim(np.min(r_squared_MC_epochs[ii]), np.max(r_squared_MC_epochs[ii]))
        plt.xlim(0.3,2)

        plt.tight_layout()

        plt.savefig(figpath+'/reject_via_binning_epoch'+str(ii+1)+'.pdf')

    return None


