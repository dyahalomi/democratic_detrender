democratic detrender
====================

Welcome to the documentation for the ``democratic detrender``!

*The documentation is currently under construction, with the API to be updated and tutorials added soon.*

Stellar time-series photometry is a combination of periodic, quasi-periodic, and non-periodic variations 
caused by both physical and instrumental factors. There is thus no "perfect" model for this nuisance signal. 
To mitigate model dependency, ``democratic detrender`` that performs stellar detrending using a novel 
ensemble-based "voting" approach via a community-of-models. This detrending package has been extensively 
tested on thousands of *TESS* and *Kepler* light curves. For more information, please see the 
`paper describing the methodology <https://arxiv.org/abs/2411.09753>`_.

If you use the `democratic detrender` in your research, please cite us!
See the `repository <https://github.com/dyahalomi/democratic_detrender#citation>`_ for the citation.

For help, submit a bug report or feature request on the
`issues page <https://github.com/dyahalomi/democratic_detrender/issues>`_. 
For anything else, please don’t hesitate to `reach out <daniel.yahalomi@columbia.edu>`_ with any questions.

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: User Guide

    Installation <installation>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Tutorials

    xx <tutorials/blank_tutorial.ipynb>

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: API

    api/preprocessing
    api/detrending
    api/plotting

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Reference

    Index <genindex>
    Changelog <https://github.com/dyahalomi/democratic_detrender/blob/main/HISTORY.rst>
    GitHub <https://github.com/dyahalomi/democratic_detrender>
