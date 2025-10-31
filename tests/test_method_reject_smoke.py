import numpy as np
import pandas as pd
import pytest
import os

import democratic_detrender.method_reject as mr


@pytest.fixture
def fake_input_dir(tmp_path):
    """
    Create a temp directory that looks like an existing detrend_all() run output.
    We will create:
    - detrended.csv
    - orbital_data.csv
    - t0s.csv
    which method_reject() expects.
    """
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Fake detrended.csv ---
    # multiple segments separated by gaps >5 in 'time'
    time_arr = np.array([0.0, 2.5, 5.0, 12.0, 14.5, 17.0])
    yerr_arr = np.linspace(0.01, 0.02, len(time_arr))
    mask_arr = np.zeros(len(time_arr), dtype=bool)

    local_sap = np.linspace(-0.001, 0.001, len(time_arr))
    local_pdc = np.linspace(0.002, 0.003, len(time_arr))
    poly_sap = np.linspace(-0.002, 0.002, len(time_arr))
    poly_pdc = np.linspace(0.003, 0.004, len(time_arr))
    gp_sap = np.linspace(-0.003, 0.003, len(time_arr))
    gp_pdc = np.linspace(0.004, 0.005, len(time_arr))
    cofiam_sap = np.linspace(-0.004, 0.004, len(time_arr))
    cofiam_pdc = np.linspace(0.005, 0.006, len(time_arr))

    method_marg = np.mean(
        np.vstack(
            [
                local_sap,
                local_pdc,
                poly_sap,
                poly_pdc,
                gp_sap,
                gp_pdc,
                cofiam_sap,
                cofiam_pdc,
            ]
        ),
        axis=0,
    )

    detrended_df = pd.DataFrame(
        {
            "time": time_arr,
            "yerr": yerr_arr,
            "mask": mask_arr,
            "method marginalized": method_marg,
            "local SAP": local_sap,
            "local PDCSAP": local_pdc,
            "polyAM SAP": poly_sap,
            "polyAM PDCSAP": poly_pdc,
            "GP SAP": gp_sap,
            "GP PDCSAP": gp_pdc,
            "CoFiAM SAP": cofiam_sap,
            "CoFiAM PDCSAP": cofiam_pdc,
        }
    )
    detrended_df.to_csv(target_dir / "detrended.csv", index=False)

    # --- Fake orbital_data.csv ---
    # method_reject reads 'period' and 'duration', and then takes [0]
    orbital_df = pd.DataFrame(
        {
            "period": [5.0],      # days
            "duration": [2.0],    # hours
        }
    )
    orbital_df.to_csv(target_dir / "orbital_data.csv", index=False)

    # --- Fake t0s.csv ---
    # method_reject reads this column as list(...) of "t0s_in_data"
    t0s_df = pd.DataFrame({"t0s_in_data": [1.5, 6.5]})
    t0s_df.to_csv(target_dir / "t0s.csv", index=False)

    return target_dir


def test_method_reject_runs_with_monkeypatch(fake_input_dir, monkeypatch):
    """
    Smoke test for method_reject().

    We:
    - Monkeypatch heavy helper functions and plotting
    - Run method_reject() with minimal args it actually accepts
    - Assert structure of return value and output artifacts
    """

    # Helper: build deterministic dummy outputs in the shape
    # method_reject() expects from its helpers.

    # reject_via_DW returns:
    #   dw_sigma_test, DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs
    # We'll just make simple placeholders with matching shapes.
    def _fake_reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000):
        # sigma thresholds (per epoch, per detrending method) -> ones
        dw_sigma_test = [np.ones(len(y_epochs[i].columns)) for i in range(len(y_epochs))]
        # MC epochs, detrended epochs, upper bound epochs: mimic y_epochs shapes
        return dw_sigma_test, y_epochs, y_epochs, y_epochs

    monkeypatch.setattr(mr, "reject_via_DW", _fake_reject_via_DW, raising=True)

    # reject_via_binning returns:
    #   binning_sigma_test, beta_MC_epochs, beta_detrended_epochs, binning_upper_bound_epochs
    def _fake_reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000):
        binning_sigma_test = [np.ones(len(y_epochs[i].columns)) for i in range(len(y_epochs))]
        return binning_sigma_test, y_epochs, y_epochs, y_epochs

    monkeypatch.setattr(mr, "reject_via_binning", _fake_reject_via_binning, raising=True)

    # binning_rejection_plots / dw_rejection_plots are just plotting helpers
    def _noop_plot(*args, **kwargs):
        return None

    monkeypatch.setattr(mr, "dw_rejection_plots", _noop_plot, raising=False)
    monkeypatch.setattr(mr, "binning_rejection_plots", _noop_plot, raising=False)

    # reject_epochs_by_white_noise_tests:
    # returns filtered y_epochs. We'll just pass through unchanged.
    def _fake_reject_epochs_by_white_noise_tests(y_epochs, dw_sigma_test, binning_sigma_test, detrending_methods):
        return y_epochs

    monkeypatch.setattr(
        mr,
        "reject_epochs_by_white_noise_tests",
        _fake_reject_epochs_by_white_noise_tests,
        raising=True,
    )

    # merge_epochs:
    # returns (times_all_post_rej, y_all_post_rej, yerr_all_post_rej)
    # We'll just flatten epochs into 1D arrays / list of arrays.
    def _fake_merge_epochs(time_epochs, y_epochs_post_rej, yerr_epochs):
        # concatenate along epochs
        all_times = np.concatenate([np.array(ep) for ep in time_epochs])
        # y_epochs_post_rej[i] is a DataFrame of detrend methods at those times
        # we'll vertically stack these frames, then split columns into list
        y_concat_df = pd.concat(y_epochs_post_rej, ignore_index=True)
        # list of per-method arrays (column-wise)
        y_all_post_rej = [y_concat_df[col].to_numpy() for col in y_concat_df.columns]
        # concatenate yerr epochs
        all_yerr = np.concatenate([np.array(ep) for ep in yerr_epochs])
        return all_times, y_all_post_rej, all_yerr

    monkeypatch.setattr(mr, "merge_epochs", _fake_merge_epochs, raising=True)

    # ensemble_step:
    # In your code, the real ensemble_step(...) returns a final dataframe.
    # Actually in method_reject() it does:
    #   detrend_df_post_rej = ensemble_step(times_all_post_rej,
    #                                       y_all_post_rej,
    #                                       yerr_all_post_rej,
    #                                       detrending_methods,
    #                                       df['mask'])
    #
    # So we'll build and return a DataFrame that looks like the expected output
    # columns: time, yerr, mask, method marginalized, plus each method column.
    def _fake_ensemble_step(times_all_post_rej,
                            y_all_post_rej,
                            yerr_all_post_rej,
                            detrending_methods,
                            mask_series):
        times_all_post_rej = np.array(times_all_post_rej)
        yerr_all_post_rej = np.array(yerr_all_post_rej)

        # y_all_post_rej is list-of-arrays, one per method in detrending_methods,
        # in the same order. We'll stack them so we can compute "method marginalized"
        # as a median across methods.
        method_matrix = np.vstack(y_all_post_rej)  # shape (n_methods, n_points)
        method_marg = np.nanmedian(method_matrix, axis=0)

        out = {
            "time": times_all_post_rej,
            "yerr": yerr_all_post_rej,
            "mask": np.array(mask_series[: len(times_all_post_rej)]),
            "method marginalized": method_marg,
        }
        for i, mname in enumerate(detrending_methods):
            out[mname] = method_matrix[i, :]

        return pd.DataFrame(out)

    monkeypatch.setattr(mr, "ensemble_step", _fake_ensemble_step, raising=True)

    # plot_detrended_lc is called twice at the end, just stub it
    monkeypatch.setattr(mr, "plot_detrended_lc", _noop_plot, raising=False)

    # ---- Call method_reject with ONLY the args it actually accepts
    out_df = mr.method_reject(
        path=str(fake_input_dir),
        input_depth=0.01,
        input_period=None,
        input_duration=None,
        input_mask_width=1.1,
    )

    # ---- Assertions on returned dataframe
    assert isinstance(out_df, pd.DataFrame)

    required_cols = [
        "time",
        "yerr",
        "mask",
        "method marginalized",
        "local SAP",
        "local PDCSAP",
        "polyAM SAP",
        "polyAM PDCSAP",
        "GP SAP",
        "GP PDCSAP",
        "CoFiAM SAP",
        "CoFiAM PDCSAP",
    ]

    for col in required_cols:
        assert col in out_df.columns, f"missing '{col}' in method_reject() output df"

    assert np.all(np.isfinite(out_df["method marginalized"].to_numpy()))
    assert np.all(np.isfinite(out_df["yerr"].to_numpy()))

    # ---- Side effects
    # It should write:
    #   method_rejection_figures/  (directory)
    #   individual_detrended_post_rejection.pdf
    #   method_marg_detrended_post_rejection.pdf
    #   detrended_post_method_rejection.csv
    #
    # We'll just assert on the CSV since it's simple and portable.
    out_csv_path = os.path.join(str(fake_input_dir), "detrended_post_method_rejection.csv")
    assert os.path.exists(out_csv_path), "detrended_post_method_rejection.csv should be written"

    out_csv_df = pd.read_csv(out_csv_path)
    for col in required_cols:
        assert col in out_csv_df.columns, (
            f"Column '{col}' should appear in detrended_post_method_rejection.csv"
        )
    assert np.all(np.isfinite(out_csv_df["method marginalized"].to_numpy()))
    assert np.all(np.isfinite(out_csv_df["yerr"].to_numpy()))

