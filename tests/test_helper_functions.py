
import numpy as np

# we assume your package directory is named "democratic_detrender"
# and lives next to the "tests" folder.
from democratic_detrender import helper_functions as hf


def test_durbin_watson_basic():
    """
    durbin_watson should return a finite scalar >= 0 for a simple residual sequence.
    """
    residuals = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    dw = hf.durbin_watson(residuals)

    assert np.isscalar(dw), "durbin_watson() should return a scalar"
    assert np.isfinite(dw), "durbin_watson() should return a finite numeric value"
    assert dw >= 0.0, "Durbin-Watson statistic should be non-negative"


def test_get_detrended_lc_shapes_and_math():
    """
    get_detrended_lc(y, model) implements:
        detrended = ((y + 1) / (model + 1)) - 1
    We check shape and correctness on a toy example.
    """
    y = np.array([0.0, 0.1, -0.05])
    model = np.array([0.0, 0.05, -0.02])

    detrended = hf.get_detrended_lc(y, model)

    assert detrended.shape == y.shape

    expected_idx1 = ((y[1] + 1.0) / (model[1] + 1.0)) - 1.0
    assert np.isclose(detrended[1], expected_idx1, rtol=1e-10, atol=1e-12)


def test_determine_cadence_reasonable():
    """
    determine_cadence(times):
    - looks at diffs
    - returns the most common gap
    """
    times = np.array([0.00, 0.02, 0.04, 0.06, 0.50, 0.52, 0.54])
    cadence = hf.determine_cadence(times)
    assert np.isclose(cadence, 0.02, atol=1e-8), (
        f"Expected cadence ~0.02, got {cadence}"
    )


def test_find_nearest_finds_closest_value():
    arr = np.array([10.0, 10.5, 11.0, 20.0])
    target = 10.6
    got = hf.find_nearest(arr, target)
    assert got == 10.5


def test_ensemble_step_output_shapes_and_no_crash():
    """
    ensemble_step(y_detrended, yerr_detrended, method=...)
    should:
    - return two 1D arrays (flux, flux_err)
    - length == n_times
    - work for 'median' and 'mean'
    - raise on invalid method
    """
    y_detrended = np.array([
        [0.0,   0.1,   np.nan],
        [0.05,  0.1,   0.15  ],
        [-0.02, np.nan, -0.01],
        [0.0,   0.0,   0.0   ],
        [0.2,   0.25,  0.30  ],
    ])  # shape (5,3)

    yerr = np.array([0.01, 0.01, 0.02, 0.02, 0.05])

    flux_med, fluxerr_med = hf.ensemble_step(
        y_detrended=y_detrended,
        yerr_detrended=yerr,
        method="median",
    )

    assert flux_med.shape == (5,)
    assert fluxerr_med.shape == (5,)
    assert np.all(np.isfinite(fluxerr_med))

    flux_mean, fluxerr_mean = hf.ensemble_step(
        y_detrended=y_detrended,
        yerr_detrended=yerr,
        method="mean",
    )

    assert flux_mean.shape == (5,)
    assert fluxerr_mean.shape == (5,)
    assert np.all(np.isfinite(fluxerr_mean))

    try:
        hf.ensemble_step(y_detrended, yerr, method="not_a_valid_option")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "ensemble_step should raise ValueError if method is not 'median' or 'mean'"
        )
