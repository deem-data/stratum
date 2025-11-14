from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import Any, List

from .config import set_config, get_config

def _first_version(*names):
    for n in names:
        try:
            return _dist_version(n)
        except PackageNotFoundError:
            pass
    return "0+unknown"

# Our own version
__version__ = _first_version("stratum-ai", "stratum")

# Apply Stratum patching to upstream skrub automatically
from .patching import patch_skrub
patch_skrub()

# Import original skrub
_skrub = import_module("skrub")
__skrub_version__ = _dist_version("skrub")

# stratum specific config knobs
from . import config as config

# Expose our subclasses under the same names
from .adapters.string_encoder import RustyStringEncoder as StringEncoder
from .adapters.one_hot_encoder import RustyOneHotEncoder as OneHotEncoder

# __all__ passthrough. Keep __all__ minimal
__all__ = [
    "StringEncoder",
    "OneHotEncoder",
    "config",
    "__version__",
    "__skrub_version__",
    "versions",
    "set_config",
    "get_config",
]

# Delegate everything else to real skrub lazily
def __getattr__(name: str) -> Any:
    return getattr(_skrub, name)

def __dir__() -> List[str]:
    # So IDEs and dir() look decent
    return sorted(set(__all__) | set(dir(_skrub)))

def versions():
    return {"stratum": __version__, "skrub": __skrub_version__}
