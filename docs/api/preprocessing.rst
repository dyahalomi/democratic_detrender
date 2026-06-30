Obtaining and pre-processing data
=================================

Obtaining archival light curves
-------------------------------

Supplying individual transit times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``input_transit_times`` when measured transit centers should define the
mask instead of a strictly periodic ephemeris. Times are supplied in BJD. A
flat sequence applies to the planet selected by ``input_planet_number``::

    democratic_detrend(
        "Kepler-1513",
        "Kepler",
        input_planet_number=1,
        input_transit_times=[2455001.2, 2455161.7, 2455322.4],
    )

For a multi-planet system, use a mapping whose keys are the one-based planet
numbers used by the archive results::

    democratic_detrend(
        "example target",
        "TESS",
        input_planet_number=1,
        input_transit_times={
            1: [2459001.2, 2459011.4],
            2: [2459005.1, 2459023.7],
        },
    )

The combined mask contains the supplied transits for both planets, while the
fitted-planet mask and returned epoch centers use planet 1. Planets omitted
from the mapping continue to use their catalog or user-supplied periodic
ephemeris. Period and duration values are still required by the downstream
detrending and plotting procedures.

Optional outlier rejection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Outlier rejection is enabled by default. Set ``input_reject_outliers=False``
to bypass both the pre-detrending moving-median rejection and the outlier pass
applied after each detrending method::

    democratic_detrend(
        "example target",
        "TESS",
        flux_type="pdc",
        input_reject_outliers=False,
    )

Disabling rejection can substantially accelerate short-cadence light curves,
but unflagged anomalous points will be passed to the detrending methods.

.. automodule:: democratic_detrender.get_lc
    :members:
    :undoc-members:
    :show-inheritance:

Rejecting outliers
------------------

.. automodule:: democratic_detrender.outlier_rejection
    :members:
    :undoc-members:
    :show-inheritance:

Identifying flux jumps
----------------------

.. automodule:: democratic_detrender.find_flux_jumps
    :members:
    :undoc-members:
    :show-inheritance:
    
Manipulating light curve data
-----------------------------

.. automodule:: democratic_detrender.manipulate_data
    :members:
    :undoc-members:
    :show-inheritance:
