import os
import pytest
from stratum._config import (
    _env_bool,
    _env_int,
    set_config,
    get_config,
    FLAGS,
)


class TestEnvBool:
    """Test _env_bool function."""
    
    def test_env_bool_true_values(self):
        """Test various true values."""
        for val in ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]:
            os.environ["TEST_BOOL"] = val
            assert _env_bool("TEST_BOOL", False) is True
            del os.environ["TEST_BOOL"]
    
    def test_env_bool_false_values(self):
        """Test various false values."""
        for val in ["0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"]:
            os.environ["TEST_BOOL"] = val
            assert _env_bool("TEST_BOOL", True) is False
            del os.environ["TEST_BOOL"]
    
    def test_env_bool_not_set_default_false(self):
        """Test default value when env var not set."""
        if "TEST_BOOL" in os.environ:
            del os.environ["TEST_BOOL"]
        assert _env_bool("TEST_BOOL", False) is False
    
    def test_env_bool_not_set_default_true(self):
        """Test default value when env var not set."""
        if "TEST_BOOL" in os.environ:
            del os.environ["TEST_BOOL"]
        assert _env_bool("TEST_BOOL", True) is True
    
    def test_env_bool_whitespace(self):
        """Test that whitespace is stripped."""
        os.environ["TEST_BOOL"] = "  true  "
        assert _env_bool("TEST_BOOL", False) is True
        del os.environ["TEST_BOOL"]
    
    def test_env_bool_unknown_value(self):
        """Test unknown value falls back to comparison with 'true'."""
        os.environ["TEST_BOOL"] = "unknown"
        # Should return False since "unknown" != "true"
        assert _env_bool("TEST_BOOL", False) is False
        del os.environ["TEST_BOOL"]


class TestEnvInt:
    """Test _env_int function."""
    
    def test_env_int_valid_values(self):
        """Test valid integer values."""
        for val in ["0", "1", "42", "100", "-5"]:
            os.environ["TEST_INT"] = val
            assert _env_int("TEST_INT", 0) == int(val)
            del os.environ["TEST_INT"]
    
    def test_env_int_not_set_default(self):
        """Test default value when env var not set."""
        if "TEST_INT" in os.environ:
            del os.environ["TEST_INT"]
        assert _env_int("TEST_INT", 42) == 42
        assert _env_int("TEST_INT", 0) == 0
    
    def test_env_int_invalid_raises(self):
        """Test that invalid integer values raise ValueError."""
        os.environ["TEST_INT"] = "not_a_number"
        with pytest.raises(ValueError):
            _env_int("TEST_INT", 0)
        del os.environ["TEST_INT"]


class TestSetConfig:
    """Test set_config function."""
    
    def setup_method(self):
        """Reset config to defaults before each test."""
        # Save original env values
        self.original_env = {}
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING", 
                   "SKRUB_RUST_ALLOW_MONKEYPATCH", "STRATUM_STATS"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
        
        # Reset to defaults
        set_config(rust_backend=False, num_threads=0, debug_timing=False, 
                  allow_patch=True, stratum_stats=False)
    
    def teardown_method(self):
        """Restore original environment after each test."""
        # Clear test env vars
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING",
                   "SKRUB_RUST_ALLOW_MONKEYPATCH", "STRATUM_STATS"]:
            if key in os.environ:
                del os.environ[key]
        
        # Restore original values
        for key, value in self.original_env.items():
            os.environ[key] = value
    
    def test_set_config_rust_backend(self):
        """Test setting rust_backend."""
        set_config(rust_backend=True)
        assert FLAGS.rust_backend is True
        assert os.environ["SKRUB_RUST"] == "1"
        
        set_config(rust_backend=False)
        assert FLAGS.rust_backend is False
        assert os.environ["SKRUB_RUST"] == "0"
    
    def test_set_config_num_threads(self):
        """Test setting num_threads."""
        set_config(num_threads=4)
        assert FLAGS.num_threads == 4
        assert os.environ["SKRUB_RUST_THREADS"] == "4"
        
        set_config(num_threads=0)
        assert FLAGS.num_threads == 0
        assert os.environ["SKRUB_RUST_THREADS"] == "0"
        
        set_config(num_threads=8)
        assert FLAGS.num_threads == 8
        assert os.environ["SKRUB_RUST_THREADS"] == "8"
    
    def test_set_config_num_threads_invalid(self):
        """Test invalid num_threads raises ValueError."""
        with pytest.raises(ValueError, match="num_threads must be an int >= 0"):
            set_config(num_threads=-1)
        
        with pytest.raises(ValueError, match="num_threads must be an int >= 0"):
            set_config(num_threads="invalid")
    
    def test_set_config_debug_timing(self):
        """Test setting debug_timing."""
        set_config(debug_timing=True)
        assert FLAGS.debug_timing is True
        assert os.environ["SKRUB_RUST_DEBUG_TIMING"] == "1"
        
        set_config(debug_timing=False)
        assert FLAGS.debug_timing is False
        assert os.environ["SKRUB_RUST_DEBUG_TIMING"] == "0"
    
    def test_set_config_allow_patch(self):
        """Test setting allow_patch."""
        set_config(allow_patch=True)
        assert FLAGS.allow_patch is True
        assert os.environ["SKRUB_RUST_ALLOW_MONKEYPATCH"] == "1"
        
        set_config(allow_patch=False)
        assert FLAGS.allow_patch is False
        assert os.environ["SKRUB_RUST_ALLOW_MONKEYPATCH"] == "0"
    
    def test_set_config_stratum_stats(self):
        """Test setting stratum_stats."""
        set_config(stratum_stats=True)
        assert FLAGS.stratum_stats is True
        assert os.environ["STRATUM_STATS"] == "1"
        
        set_config(stratum_stats=False)
        assert FLAGS.stratum_stats is False
        assert os.environ["STRATUM_STATS"] == "0"
    
    def test_set_config_multiple_params(self):
        """Test setting multiple parameters at once."""
        set_config(rust_backend=True, num_threads=4, debug_timing=True, 
                  allow_patch=False, stratum_stats=True)
        assert FLAGS.rust_backend is True
        assert FLAGS.num_threads == 4
        assert FLAGS.debug_timing is True
        assert FLAGS.allow_patch is False
        assert FLAGS.stratum_stats is True
        assert os.environ["SKRUB_RUST"] == "1"
        assert os.environ["SKRUB_RUST_THREADS"] == "4"
        assert os.environ["SKRUB_RUST_DEBUG_TIMING"] == "1"
        assert os.environ["SKRUB_RUST_ALLOW_MONKEYPATCH"] == "0"
        assert os.environ["STRATUM_STATS"] == "1"
    
    def test_set_config_partial_update(self):
        """Test that None values don't update config."""
        # Set initial values
        set_config(rust_backend=True, num_threads=4)
        assert FLAGS.rust_backend is True
        assert FLAGS.num_threads == 4
        
        # Update only one value
        set_config(rust_backend=False)
        assert FLAGS.rust_backend is False
        assert FLAGS.num_threads == 4  # Should remain unchanged
    
    def test_set_config_bool_coercion(self):
        """Test that non-bool values are coerced to bool."""
        # Test with truthy values
        set_config(rust_backend=1)
        assert FLAGS.rust_backend is True
        
        set_config(rust_backend="truthy")
        assert FLAGS.rust_backend is True
        
        # Test with falsy values
        set_config(rust_backend=0)
        assert FLAGS.rust_backend is False
        
        set_config(rust_backend="")
        assert FLAGS.rust_backend is False


class TestGetConfig:
    """Test get_config function."""
    
    def setup_method(self):
        """Reset config to defaults before each test."""
        self.original_env = {}
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING",
                   "SKRUB_RUST_ALLOW_MONKEYPATCH", "STRATUM_STATS"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
        
        set_config(rust_backend=False, num_threads=0, debug_timing=False,
                  allow_patch=True, stratum_stats=False)
    
    def teardown_method(self):
        """Restore original environment after each test."""
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING",
                   "SKRUB_RUST_ALLOW_MONKEYPATCH", "STRATUM_STATS"]:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            os.environ[key] = value
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)
    
    def test_get_config_keys(self):
        """Test that get_config returns expected keys."""
        config = get_config()
        expected_keys = {"rust_backend", "num_threads", "debug_timing", "allow_patch"}
        assert set(config.keys()) == expected_keys
    
    def test_get_config_values(self):
        """Test that get_config returns current flag values."""
        set_config(rust_backend=True, num_threads=8, debug_timing=True, allow_patch=False)
        config = get_config()
        assert config["rust_backend"] is True
        assert config["num_threads"] == 8
        assert config["debug_timing"] is True
        assert config["allow_patch"] is False
    
    def test_get_config_shallow_copy(self):
        """Test that get_config returns a shallow copy."""
        config1 = get_config()
        config2 = get_config()
        # Should be different dict objects
        assert config1 is not config2
        # But should have same values
        assert config1 == config2
        
        # Modifying returned dict shouldn't affect FLAGS
        config1["rust_backend"] = "modified"
        assert FLAGS.rust_backend != "modified"


class TestFlagsInitialization:
    """Test _Flags dataclass initialization."""
    
    def test_flags_exists(self):
        """Test that FLAGS object exists."""
        assert FLAGS is not None
        assert hasattr(FLAGS, "rust_backend")
        assert hasattr(FLAGS, "num_threads")
        assert hasattr(FLAGS, "debug_timing")
        assert hasattr(FLAGS, "allow_patch")
        assert hasattr(FLAGS, "stratum_stats")
    
    def test_flags_types(self):
        """Test that FLAGS have correct types."""
        assert isinstance(FLAGS.rust_backend, bool)
        assert isinstance(FLAGS.num_threads, int)
        assert isinstance(FLAGS.debug_timing, bool)
        assert isinstance(FLAGS.allow_patch, bool)
        assert isinstance(FLAGS.stratum_stats, bool)

