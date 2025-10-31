import numpy as np
import pandas as pd
import pytest
import os

import democratic_detrender.detrend as detrend_mod


@pytest.fixture
def fake_env(tmp_path):
    """
    Temp directory for detrend_all() to treat as input_dir.
    We'll inspect files that get written there.
    """
    return tmp_path


def test_detrend_all_runs_with_monkeypatch_and_writes_csv(fake_env, monkeypatch):
    """
    Smoke test for detrend_all() with flux_type='pdc'.

    We monkeypatch heavy/external helpers inside democratic_detrender.detrend:
    - find_flux_jumps
    - detrend_one_lc
    - ensemble_step
    - plotting helpers
    - (shim) pandas.DataFrame inside detrend_mod to tolerate scalar metadata

    Then we assert detrend_all():
    - returns a DataFrame with expected columns
    - writes detrended.csv / orbital_data.csv / t0s.csv
    - returns orbital info consistent with our stubs
    """

    N = 5
    fake_time = np.arange(float(N))
    fake_flux = np.linspace(0.0, 0.02, N)
    fake_yerr = np.full(N, 0.01)
    fake_mask = np.zeros(N, dtype=bool)

    # --- stub find_flux_jumps
    def _fake_find_flux_jumps(*args, **kwargs):
        """
        Return epochs and metadata in the exact structure detrend_all() expects
        for the 'pdc' path.

        We return scalars for period/duration/etc so the math in detrend_all
        (for plotting) doesn't blow up.
        """
        x_epochs = [fake_time]
        y_epochs = [fake_flux]
        yerr_epochs = [fake_yerr]
        mask_epochs = [fake_mask]
        mask_fitted_planet_epochs = [fake_mask]
        problem_times = [[]]

        t0s = 1.5        # mid-transit time
        period = 5.0     # days
        duration = 2.0   # hours
        cadence = 0.02   # days per point

        return [
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
        ]

    monkeypatch.setattr(
        detrend_mod,
        "find_flux_jumps",
        _fake_find_flux_jumps,
        raising=True,
    )

    # --- stub detrend_one_lc
    def _fake_detrend_one_lc(*args, **kwargs):
        """
        Return structure matching what detrend_all() expects in the 'pdc' branch.

        pdc_block must have exactly 17 items in a specific order.
        """
        local_det = fake_flux - 0.001
        poly_det = fake_flux - 0.002
        gp_det = fake_flux - 0.003
        cofiam_det = fake_flux - 0.004

        pdc_block = [
            fake_time,         # pdc_local_x
            fake_flux,         # pdc_local_y
            fake_yerr,         # pdc_local_yerr
            fake_mask,         # pdc_local_mask
            fake_mask,         # pdc_local_mask_fitted_planet
            local_det,         # pdc_local_detrended
            fake_time,         # pdc_local_x_no_outliers
            local_det,         # pdc_local_detrended_no_outliers
            poly_det,          # pdc_poly_detrended
            fake_time,         # pdc_poly_x_no_outliers
            poly_det,          # pdc_poly_detrended_no_outliers
            gp_det,            # pdc_gp_detrended
            fake_time,         # pdc_gp_x_no_outliers
            gp_det,            # pdc_gp_detrended_no_outliers
            cofiam_det,        # pdc_cofiam_detrended
            fake_time,         # pdc_cofiam_x_no_outliers
            cofiam_det,        # pdc_cofiam_detrended_no_outliers
        ]

        sap_block = ["unused_sap_block_for_smoke_test_only"]

        return [sap_block, pdc_block]

    monkeypatch.setattr(
        detrend_mod,
        "detrend_one_lc",
        _fake_detrend_one_lc,
        raising=True,
    )

    # --- stub plotting to no-op
    def _noop_plot(*args, **kwargs):
        return None

    monkeypatch.setattr(detrend_mod, "plot_detrended_lc", _noop_plot, raising=False)
    monkeypatch.setattr(detrend_mod, "plot_phase_fold_lc", _noop_plot, raising=False)

    # --- stub ensemble_step to produce a single combined curve + uncertainty
    def _fake_ensemble_step(y_detrended_T, yerr_detrended, method="median"):
        if method == "median":
            combined = np.nanmedian(y_detrended_T, axis=1)
        else:
            combined = np.nanmean(y_detrended_T, axis=1)
        # combined: array (n_times,)
        # yerr_detrended: input per-point uncertainty
        return combined, np.array(yerr_detrended)

    monkeypatch.setattr(
        detrend_mod,
        "ensemble_step",
        _fake_ensemble_step,
        raising=True,
    )

    # --- shim pandas.DataFrame INSIDE detrend_mod
    # Save the real class *before* patching:
    real_DataFrame = pd.DataFrame

    def _safe_DataFrame(data=None, *args, **kwargs):
        # If detrend_all passes a dict of scalars, wrap them in lists
        if isinstance(data, dict):
            fixed = {}
            for k, v in data.items():
                if np.isscalar(v):
                    fixed[k] = [v]
                else:
                    fixed[k] = v
            return real_DataFrame(fixed, *args, **kwargs)
        # Otherwise behave like normal
        return real_DataFrame(data, *args, **kwargs)

    # Patch only inside detrend_mod so production pandas stays untouched elsewhere
    monkeypatch.setattr(detrend_mod.pd, "DataFrame", _safe_DataFrame, raising=True)

    # ---- run the real detrend_all
    detrend_df, t0s_out, period_out, duration_out = detrend_mod.detrend_all(
        input_id="FAKE-ID",
        mission="TESS",
        flux_type="pdc",
        input_planet_number=1,
        input_dir=str(fake_env),
        input_depth=0.01,
        input_period=None,
        input_t0=None,
        input_duration=None,
        input_mask_width=1.1,
        input_show_plots=False,
        input_dont_bin=True,
        input_use_sap_problem_times=False,
        input_no_pdc_problem_times=True,
        input_user_light_curve=None,
        ensemble_statistic="median",
        input_polyAM=True,
        input_CoFiAM=True,
        input_GP=True,
        input_local=True,
    )

    # --- Assertions on return values
    assert isinstance(detrend_df, real_DataFrame)
    assert len(detrend_df) == N

    # For flux_type="pdc", detrend_all writes these columns:
    required_cols = [
        "time",
        "yerr",
        "mask",
        "method marginalized",
        "local PDCSAP",
        "polyAM PDCSAP",
        "GP PDCSAP",
        "CoFiAM PDCSAP",
    ]

    for col in required_cols:
        assert col in detrend_df.columns, f"missing '{col}' in detrend_all() output df"

    # These are scalars from _fake_find_flux_jumps
    assert np.isclose(t0s_out, 1.5)
    assert np.isclose(period_out, 5.0)
    assert np.isclose(duration_out, 2.0)

    # The combined curve and the propagated uncertainties should both be finite
    assert np.all(np.isfinite(detrend_df["method marginalized"].to_numpy()))
    assert np.all(np.isfinite(detrend_df["yerr"].to_numpy()))

    # --- Files written by detrend_all
    detrended_path = os.path.join(str(fake_env), "detrended.csv")
    orbital_path = os.path.join(str(fake_env), "orbital_data.csv")
    t0s_path = os.path.join(str(fake_env), "t0s.csv")

    assert os.path.exists(detrended_path), "detrend_all() should write detrended.csv"
    assert os.path.exists(orbital_path), "detrend_all() should write orbital_data.csv"
    assert os.path.exists(t0s_path), "detrend_all() should write t0s.csv"

    detrended_written = pd.read_csv(detrended_path)
    orbital_written = pd.read_csv(orbital_path)
    t0s_written = pd.read_csv(t0s_path)

    # detrended.csv sanity
    assert len(detrended_written) == N
    for col in required_cols:
        assert col in detrended_written.columns, (
            f"Column '{col}' should be present in detrended.csv"
        )

    # orbital_data.csv sanity
    assert "period" in orbital_written.columns
    assert "duration" in orbital_written.columns
    assert np.allclose(orbital_written["period"].to_numpy(), [5.0])
    assert np.allclose(orbital_written["duration"].to_numpy(), [2.0])

    # t0s.csv sanity
    assert "t0s_in_data" in t0s_written.columns
    assert np.isclose(t0s_written["t0s_in_data"].to_numpy()[0], 1.5)

