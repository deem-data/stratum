import stratum
import os
from stratum._config import _env_bool, _env_str
from stratum._config import FLAGS

def test_versions_contains_strings():
    versions = stratum.versions()
    assert set(versions.keys()) == {"stratum", "skrub"}
    assert all(isinstance(v, str) and v for v in versions.values())
    module_dir = stratum.__dir__()


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

def test_config_scheduler_parallelism():
    with stratum.config(scheduler_parallelism="threading"):
        assert FLAGS.scheduler_parallelism == "threading"
    with stratum.config(scheduler_parallelism="process"):
        assert FLAGS.scheduler_parallelism == "process"
    with stratum.config(scheduler_parallelism="auto"):
        assert FLAGS.scheduler_parallelism == "auto"
    try:
        with stratum.config(scheduler_parallelism="invalid"):
            assert False
    except ValueError as e:
        assert str(e) == "scheduler_parallelism must be None, 'threading', 'process', or 'auto', got invalid"
    os.environ["STRATUM_SCHEDULER_PARALLELISM"] = "threading"
    assert _env_str("STRATUM_SCHEDULER_PARALLELISM") == "threading"
    os.environ["STRATUM_SCHEDULER_PARALLELISM"] = "none"
    assert _env_str("STRATUM_SCHEDULER_PARALLELISM") is None
    del os.environ["STRATUM_SCHEDULER_PARALLELISM"]


