
import numpy as np
import pandas as pd
import pytest
import os

import democratic_detrender.method_reject as mr


def _make_fake_input_dir(tmp_path):
    """
    Create the minimal CSV inputs method_reject() expects in `path`:
      - detrended.csv
      - orbital_data.csv
      - t0s.csv
    """
    path = tmp_path / "target"
    path.mkdir(parents=True, exist_ok=True)

    time_vals = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    n = len(time_vals)

    local_sap = np.linspace(0.0, 0.05, n)
    local_pdc = np.linspace(0.01, 0.06, n)
    poly_sap = np.linspace(-0.01, 0.02, n)
    poly_pdc = np.linspace(-0.02, 0.03, n)
    gp_sap = np.linspace(0.002, 0.004, n)
    gp_pdc = np.linspace(0.003, 0.005, n)
    cofiam_sap = np.linspace(-0.005, 0.005, n)
    cofiam_pdc = np.linspace(-0.006, 0.006, n)

    yerr_vals = np.full(n, 0.01)
    mask_vals = np.zeros(n, dtype=bool)

    detrended_df = pd.DataFrame(
        {
            "time": time_vals,
            "yerr": yerr_vals,
            "mask": mask_vals,
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
    detrended_df.to_csv(path / "detrended.csv", index=False)

    orbital_df = pd.DataFrame(
        {
            "period": [5.0],
            "duration": [2.0],
        }
    )
    orbital_df.to_csv(path / "orbital_data.csv", index=False)

    t0s_df = pd.DataFrame(
        {
            "t0s_in_data": [0.5, 5.5],
        }
    )
    t0s_df.to_csv(path / "t0s.csv", index=False)

    return path


@pytest.fixture
def fake_input_dir(tmp_path):
    return _make_fake_input_dir(tmp_path)


def test_method_reject_runs_with_monkeypatch(fake_input_dir, monkeypatch):
    """
    Smoke test for method_reject():
    - returns a DataFrame
    - writes detrended_post_method_rejection.csv
    with expected columns and finite values.

    Heavy steps (MC sims, plotting) are monkeypatched.
    """

    def _noop_plot(*args, **kwargs):
        return None

    monkeypatch.setattr(mr, "plot_detrended_lc", _noop_plot, raising=False)
    monkeypatch.setattr(mr, "dw_rejection_plots", _noop_plot, raising=False)
    monkeypatch.setattr(mr, "binning_rejection_plots", _noop_plot, raising=False)

    def _fake_reject_via_DW(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000):
        methods = list(y_epochs[0].columns)
        n_epochs = len(time_epochs)
        dw_sigma_test = {m: [True] * n_epochs for m in methods}
        DWMC_epochs = [[]] * n_epochs
        DWdetrend_epochs = [[]] * n_epochs
        DWupper_bound_epochs = [[]] * n_epochs
        return dw_sigma_test, DWMC_epochs, DWdetrend_epochs, DWupper_bound_epochs

    def _fake_reject_via_binning(time_epochs, y_epochs, yerr_epochs, t0s, period, duration, niter=100000):
        methods = list(y_epochs[0].columns)
        n_epochs = len(time_epochs)
        binning_sigma_test = {m: [True] * n_epochs for m in methods}
        beta_MC_epochs = [[]] * n_epochs
        beta_detrended_epochs = [[]] * n_epochs
        binning_upper_bound_epochs = [[]] * n_epochs
        return (
            binning_sigma_test,
            beta_MC_epochs,
            beta_detrended_epochs,
            binning_upper_bound_epochs,
        )

    monkeypatch.setattr(mr, "reject_via_DW", _fake_reject_via_DW, raising=True)
    monkeypatch.setattr(mr, "reject_via_binning", _fake_reject_via_binning, raising=True)

    out_df = mr.method_reject(
        path=str(fake_input_dir),
        input_depth=0.01,
        input_period=None,
        input_duration=None,
        input_mask_width=1.1,
    )

    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) > 0
    for col in ["time", "yerr", "mask"]:
        assert col in out_df.columns

    saved_file = fake_input_dir / "detrended_post_method_rejection.csv"
    assert saved_file.exists(), "Output CSV from method_reject() was not written"

    written_df = pd.read_csv(saved_file)
    assert len(written_df) == len(out_df)

    assert "method marginalized" in written_df.columns
    assert "method marginalized uncertainty" in written_df.columns

    assert np.all(np.isfinite(written_df["method marginalized"].to_numpy()))
    assert np.all(np.isfinite(written_df["method marginalized uncertainty"].to_numpy()))

    assert os.path.isdir(fake_input_dir)
