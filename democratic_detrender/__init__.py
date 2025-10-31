"""
Democratic Detrender: An ensemble-based approach to removing nuisance signals.

This package provides tools for removing nuisance signals from stellar 
time-series photometry using an ensemble of different detrending methods.
"""

from .democratic_detrend import democratic_detrend
from .detrend_only import detrend_only
from .problem_times_only import problem_times_only

__all__ = ["democratic_detrend", "detrend_only", "problem_times_only"]
__license__ = "MIT"
__description__ = "An ensemble-based approach to removing nuisance signals from stellar time-series photometry."
__version__ = "0.0.1"