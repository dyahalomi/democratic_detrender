
import numpy as np
import pandas as pd
import pytest
import os

import democratic_detrender.democratic_detrend as dt


@pytest.fixture
def fake_env(tmp_path):
    """
    Makes a temp directory for detrend_all() to treat as input_dir.
    We'll inspect the files that get written there.
    """
    return tmp_path


def test_detrend_all_runs_with_monkeypatch_and_writes_csv(fake_env, monkeypatch):
    """
    Smoke test for detrend_all() with flux_type='pdc'.
    """

    N = 5
    fake_time = np.arange(float(N))
    fake_flux = np.linspace(0.0, 0.02, N)
    fake_yerr = np.full(N, 0.01)
    fake_mask = np.zeros(N, dtype=bool)

    def _fake_find_flux_jumps(*args, **kwargs):
        x_epochs = [fake_time]
        y_epochs = [fake_flux]
        yerr_epochs = [fake_yerr]
        mask_epochs = [fake_mask]
        mask_fitted_planet_epochs = [fake_mask]
        problem_times = [[]]
        t0s = [1.5]
        period = [5.0]
        duration = [2.0]
        cadence = [0.02]
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

    monkeypatch.setattr(dt, "find_flux_jumps", _fake_find_flux_jumps, raising=True)

    def _fake_detrend_one_lc(*args, **kwargs):
        local_det = fake_flux - 0.001
        poly_det = fake_flux - 0.002
        gp_det = fake_flux - 0.003
        cofiam_det = fake_flux - 0.004

        pdc_block = [
            fake_time,            # pdc_local_x
            fake_flux,            # pdc_local_y
            fake_yerr,            # pdc_local_yerr
            fake_mask,            # pdc_local_mask
            np.zeros(N),          # pdc_local_periodic_model
            local_det,            # pdc_local_detrended
            fake_time,            # pdc_poly_x
            fake_flux,            # pdc_poly_y
            fake_yerr,            # pdc_poly_yerr
            fake_mask,            # pdc_poly_mask
            np.zeros(N),          # pdc_poly_periodic_model
            poly_det,             # pdc_poly_detrended
            fake_time,            # pdc_gp_x
            gp_det,               # pdc_gp_detrended
            fake_time,            # pdc_cofiam_x
            cofiam_det,           # pdc_cofiam_detrended
            fake_time,            # pdc_gp_x_no_outliers
            gp_det,               # pdc_gp_detrended_no_outliers
            fake_time,            # pdc_cofiam_x_no_outliers
            cofiam_det,           # pdc_cofiam_detrended_no_outliers
        ]

        sap_block = ["unused_sap"]
        return [sap_block, pdc_block]

    monkeypatch.setattr(dt, "detrend_one_lc", _fake_detrend_one_lc, raising=True)

    def _noop_plot(*args, **kwargs):
        return None

    monkeypatch.setattr(dt, "plot_detrended_lc", _noop_plot, raising=False)
    monkeypatch.setattr(dt, "plot_phase_fold_lc", _noop_plot, raising=False)

    def _fake_ensemble_step(y_detrended_T, yerr_detrended, method="median"):
        if method == "median":
            combined = np.nanmedian(y_detrended_T, axis=1)
        else:
            combined = np.nanmean(y_detrended_T, axis=1)
        return combined, np.array(yerr_detrended)

    monkeypatch.setattr(dt, "ensemble_step", _fake_ensemble_step, raising=True)

    detrend_df, t0s_out, period_out, duration_out = dt.detrend_all(
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

    assert isinstance(detrend_df, pd.DataFrame)
    assert len(detrend_df) == N

    for col in [
        "time",
        "yerr",
        "mask",
        "local PDCSAP",
        "polyAM PDCSAP",
        "GP PDCSAP",
        "CoFiAM PDCSAP",
        "method marginalized",
        "method marginalized uncertainty",
    ]:
        assert col in detrend_df.columns, f"missing '{col}' in detrend_all() output df"

    assert np.allclose(np.array(t0s_out), [1.5])
    assert np.allclose(np.array(period_out), [5.0])
    assert np.allclose(np.array(duration_out), [2.0])

    assert np.all(np.isfinite(detrend_df["method marginalized"].to_numpy()))
    assert np.all(np.isfinite(detrend_df["method marginalized uncertainty"].to_numpy()))

    detrended_path = os.path.join(str(fake_env), "detrended.csv")
    orbital_path = os.path.join(str(fake_env), "orbital_data.csv")
    t0s_path = os.path.join(str(fake_env), "t0s.csv")

    assert os.path.exists(detrended_path), "detrend_all() should write detrended.csv"
    assert os.path.exists(orbital_path), "detrend_all() should write orbital_data.csv"
    assert os.path.exists(t0s_path), "detrend_all() should write t0s.csv"

    detrended_written = pd.read_csv(detrended_path)
    orbital_written = pd.read_csv(orbital_path)
    t0s_written = pd.read_csv(t0s_path)

    assert len(detrended_written) == N
    for col in [
        "time",
        "yerr",
        "mask",
        "local PDCSAP",
        "polyAM PDCSAP",
        "GP PDCSAP",
        "CoFiAM PDCSAP",
        "method marginalized",
        "method marginalized uncertainty",
    ]:
        assert col in detrended_written.columns, f"Column '{col}' should be present in detrended.csv"

    assert "period" in orbital_written.columns
    assert "duration" in orbital_written.columns
    assert np.allclose(orbital_written["period"].to_numpy(), [5.0])
    assert np.allclose(orbital_written["duration"].to_numpy(), [2.0])

    assert "t0s_in_data" in t0s_written.columns
    assert np.allclose(
        np.sort(t0s_written["t0s_in_data"].to_numpy()),
        np.sort([1.5]),
    )
