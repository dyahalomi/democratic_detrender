def reject_epochs_by_white_noise_tests(y_epochs, dw_sigma_test, binning_sigma_test, detrending_methods):
    
    for col in detrending_methods:
        for epoch in range(0, len(dw_sigma_test[col])):

            #if a certain detrending method failed at a certain epoch
            #then change all values to NaNs
            if not dw_sigma_test[col][epoch]:
                print(str(col) + ' epoch ' + str(epoch+1) + ' failed dw test')
                y_epochs[epoch][col] = np.nan
                
            if not binning_sigma_test[col][epoch]:
                print(str(col) + ' epoch ' + str(epoch+1) + ' failed binning test')
                y_epochs[epoch][col] = np.nan
                
    return y_epochs


def merge_epochs(time_epochs, y_epochs, yerr_epochs):
    # Merging the list of lists into a single list using list comprehension
    times_all = [item for sublist in time_epochs for item in sublist]
    yerr_all = [item for sublist in yerr_epochs for item in sublist]


    # Merging the list of DataFrames into a single DataFrame
    merged_y = pd.concat(y_epochs, ignore_index=True)

    # Convert each column into a NumPy array and store them in a list
    y_all = [merged_y[col].to_numpy() for col in merged_y.columns]

    return times_all, y_all, yerr_all
                

    
def ensemble_step(times_all, y_all, yerr_all, detrending_methods):

    x_detrended = np.array(times_all)
    y_detrended = np.array(y_all)
    yerr_detrended = np.array(yerr_all)

    y_detrended_transpose = y_detrended.T

    method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
    MAD = median_abs_deviation(y_detrended_transpose, axis=1, scale=1/1.4826, nan_policy = 'omit')

    yerr_detrended = np.sqrt(yerr_detrended.astype(float)**2 + MAD**2)


    # save detrend data as csv
    detrend_dict = {}

    detrend_dict["time"] = x_detrended
    detrend_dict["yerr"] = yerr_detrended
    detrend_dict["mask"] = df['mask']
    detrend_dict["method marginalized"] = method_marg_detrended

    detrend_label = detrending_methods
    for ii in range(0, len(y_detrended)):
            detrend = y_detrended[ii]
            label = detrend_label[ii]
            detrend_dict[label] = detrend


    detrend_df = pd.DataFrame(detrend_dict)

    detrend_df.to_csv('detrended_post_rejection.csv')
    
    
    return detrend_df