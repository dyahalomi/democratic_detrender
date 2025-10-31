from .democratic_detrend import democratic_detrend
from .detrend_only import detrend_only
from .problem_times_only import problem_times_only

try:
    from ._version import version as __version__
except ImportError:
    # fallback for when setuptools_scm hasn't generated the version file yet
    __version__ = "unknown"

__all__ = ["democratic_detrend"]
__license__ = "MIT"
__description__ = "An ensemble-based approach to removing nuisance signals from stellar time-series photometry."