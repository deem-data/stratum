import stratum
import os
from stratum._config import _env_bool

def test_versions_contains_strings():
    versions = stratum.versions()
    assert set(versions.keys()) == {"stratum", "skrub"}
    assert all(isinstance(v, str) and v for v in versions.values())


def test_env_bool_true_values():
    """Test various true values."""
    for val in ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]:
        os.environ["TEST_BOOL"] = val
        assert _env_bool("TEST_BOOL", False) is True
        del os.environ["TEST_BOOL"]

def test_env_bool_false_values():
    """Test various false values."""
    for val in ["0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"]:
        os.environ["TEST_BOOL"] = val
        assert _env_bool("TEST_BOOL", True) is False
        del os.environ["TEST_BOOL"]


