Glossary
========

.. glossary::

   Stellar Activity
      Stars are active and fluctuating objects. As they rotate, their brightness changes over time.
      In time series data, the flux (i.e. brightness) values fluctuate accordingly.

   Time-Series Data
      A collection (or "series") of data points measured at successive values of time.

   Photometry
      An astronomical technique that gathers photons emitted by stars, translated into electrons
      stored within each pixel of a telescope's CCD camera. The number of electrons per pixel
      corresponds to pixel brightness. Photon flux fluctuates as a star's brightness fluctuates.

   Epoch
      An arbitrary "cycle" of time within a star's observation period, defined by some arbitrarily
      chosen starting time. For this project, an epoch is always centered at the middle of a
      transit, with ½ of a period on either side.

   Transit
      A dip in the star's measured brightness due to a planet moving across it in our line of
      sight, causing the overall image to dim.

   Cadence
      The amount of time taken to detect photons emitted by a star for one observation.

   Light Curve
      Tables containing data showing the variation in a star's brightness over the observation
      period.

   Exoplanet Archive
      Databases compiled by NASA containing information on stars and their respective exoplanets
      from several missions including Kepler and TESS. Includes sections for confirmed exoplanets
      and sections containing both confirmed and candidate planets.

   SIMBAD
      A comprehensive database containing information on various stars, exoplanets, comets, and
      most other celestial objects outside our Solar System. Often lacking for TESS candidates,
      which is why the Exoplanet Archive is used to obtain TESS data.

   Detrending
      The process of removing stellar activity from data (or "flattening" the light curve) in order
      to model transits. Necessary before looking for planets in time series photometry.

   Problem Times
      Timestamped discontinuities in time series data. When strong enough, they interfere with
      detrending. They must be flagged at the time value where they start.

   Jump Times
      Same as :term:`Problem Times`.

   TESS
      Transiting Exoplanet Survey Satellite. Launched in 2018, its primary mission was to survey
      ~200,000 of the brightest nearby stars for transiting exoplanets. Covers a larger portion of
      the sky than Kepler.

   TOI
      TESS Object of Interest. Naming schema used for TESS stars and exoplanets
      (e.g., star TOI 1796, planet TOI 1796.01).

   TIC
      TESS Input Catalog. Reconciles pre-existing sky data with TESS targets by assigning each
      object a unique TIC number. A TIC number corresponds to both the star and all planets it
      hosts.

   Kepler
      A NASA telescope launched in 2009 to search for transiting exoplanets and determine their
      prevalence; it ran out of fuel in 2018.

   KOI
      Kepler Object of Interest.

   MAST
      Database of astronomical datasets queried by Lightkurve to pull light curve data for TESS
      and Kepler objects. The primary data archive for TESS data.

   ExoFOP
      Site that compiles known data and publications on a Kepler/TESS star of your choice.

   Jupyter
      Programming interface that allows writing Python code in a live, modular setting.

   SAP
      Simple Aperture Photometry.

   PDC-SAP
      Pre-search Data Conditioning SAP. Attempts to remove common spacecraft-level trends;
      each target uses one of six modes.

   SPOC
      Science Processing Operations Center. Group at NASA Ames that calibrates TESS data,
      which is then archived on MAST.

   Transit Timing Variation (TTV)
      Variation in the timing of a planet's transits, often caused by gravitational interactions
      with other planets in the system.

   Eccentricity
      How elliptical a celestial object's orbit is.

   CoFIAM
      Detrending method that fits cosines to the light curve.

   PolyAM
      Detrending method that fits polynomials to the light curve.

   GP Method
      Detrending method that fits a Gaussian Process model to the light curve.
