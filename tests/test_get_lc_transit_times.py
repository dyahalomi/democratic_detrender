import numpy as np
import pandas as pd
import pytest

import democratic_detrender.get_lc as get_lc


def test_normalize_transit_times_supports_flat_and_multi_planet_inputs():
    flat = get_lc._normalize_transit_times([3.0, 1.0], planet_number=2, nplanets=2)
    assert list(flat) == [2]
    assert np.array_equal(flat[2], [1.0, 3.0])

    multiple = get_lc._normalize_transit_times(
        {1: [1.0, 2.0], 2: [4.0, 5.0]}, planet_number=1, nplanets=2
    )
    assert np.array_equal(multiple[1], [1.0, 2.0])
    assert np.array_equal(multiple[2], [4.0, 5.0])


@pytest.mark.parametrize(
    "value",
    [
        {},
        {0: [1.0]},
        {3: [1.0]},
        {1: []},
        {1: [np.nan]},
        {"1": [1.0]},
    ],
)
def test_normalize_transit_times_rejects_invalid_inputs(value):
    with pytest.raises(ValueError):
        get_lc._normalize_transit_times(value, planet_number=1, nplanets=2)


def test_mask_from_transit_times_uses_each_individual_center():
    times = np.array([0.0, 0.95, 1.0, 1.05, 2.0, 3.95, 4.0, 4.05])
    mask = get_lc._mask_from_transit_times(times, [1.0, 4.0], duration_days=0.2)

    assert np.array_equal(mask, [False, True, True, True, False, True, True, True])


def test_get_light_curve_combines_explicit_times_for_multiple_planets(monkeypatch):
    relative_times = np.round(np.arange(99.8, 103.21, 0.05), 8)

    class Values:
        def __init__(self, value):
            self.value = value

    class FakeLightCurve:
        def __init__(self):
            self.time = Values(relative_times)
            self.flux = Values(np.ones(len(relative_times)))
            self.flux_err = Values(np.full(len(relative_times), 0.01))
            self.CROWDSAP = 1.0
            self.FLFRCSAP = 1.0

        def remove_nans(self):
            return self

        def create_transit_mask(self, **kwargs):
            raise AssertionError("periodic masking should not run for supplied planets")

    class FakeCollection(list):
        def stitch(self):
            return self[0]

    class FakeSearch:
        def download_all(self, **kwargs):
            return FakeCollection([FakeLightCurve()])

    transit_info = pd.DataFrame(
        {
            "tic_id": ["TIC 1", "TIC 1"],
            "t0 [BJD]": [2457100.0, 2457103.0],
            "period [days]": [10.0, 20.0],
            "duration [hours]": [4.8, 4.8],
        }
    )
    monkeypatch.setattr(get_lc, "get_transit_info", lambda object_id: transit_info)
    monkeypatch.setattr(get_lc.lk, "search_lightcurve", lambda *args, **kwargs: FakeSearch())

    result = get_lc.get_light_curve(
        "target",
        "sap_flux",
        TESS=True,
        planet_number=1,
        mask_width=1.0,
        user_transit_times={1: [2457100.0], 2: [2457103.0]},
    )

    times, _, _, mask, fitted_mask, t0s_in_data, *_ = result
    expected_first = get_lc._mask_from_transit_times(times, [100.0], 0.2)
    expected_second = get_lc._mask_from_transit_times(times, [103.0], 0.2)

    assert np.array_equal(fitted_mask, expected_first)
    assert np.array_equal(mask, expected_first | expected_second)
    assert np.array_equal(t0s_in_data, [100.0])
