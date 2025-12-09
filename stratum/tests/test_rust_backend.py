import os
import time
import pytest
import pandas as pd


class TestGetAttr:
    """Test __getattr__ function for dynamic attribute access."""
    
    def setup_method(self):
        """Save original environment and reset config."""
        from stratum.config import get_config, set_config
        
        self.original_env = {}
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING", 
                   "SKRUB_RUST_ALLOW_PATCH"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]
        
        # Save original config state
        original_config = get_config()
        self.original_config = {
            'rust_backend': original_config.get('rust_backend', False),
            'num_threads': original_config.get('num_threads', 0),
            'debug_timing': original_config.get('debug_timing', False),
            'allow_patch': original_config.get('allow_patch', False),
        }
        
        # Reset config
        set_config(rust_backend=False, num_threads=0, debug_timing=False, allow_patch=False)
    
    def teardown_method(self):
        """Restore original environment and config."""
        from stratum.config import set_config
        
        # Restore original config
        set_config(
            rust_backend=self.original_config['rust_backend'],
            num_threads=self.original_config['num_threads'],
            debug_timing=self.original_config['debug_timing'],
            allow_patch=self.original_config['allow_patch'],
        )
        
        # Restore original environment
        for key in ["SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING",
                   "SKRUB_RUST_ALLOW_PATCH"]:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            os.environ[key] = value
    
    def test_getattr_use_rust_from_env(self):
        """Test USE_RUST from environment variable."""
        os.environ["SKRUB_RUST"] = "1"
        from stratum import _rust_backend
        assert _rust_backend.USE_RUST is True
        
        os.environ["SKRUB_RUST"] = "0"
        # Clear any cached attribute
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('USE_RUST', None)
        assert _rust_backend.USE_RUST is False
    
    def test_getattr_use_rust_from_config(self):
        """Test USE_RUST from config."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(rust_backend=True)
        # Clear any cached attribute
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('USE_RUST', None)
        assert _rust_backend.USE_RUST is True
        
        set_config(rust_backend=False)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('USE_RUST', None)
        assert _rust_backend.USE_RUST is False
    
    def test_getattr_use_rust_env_overrides_config(self):
        """Test that environment variable overrides config."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(rust_backend=False)
        os.environ["SKRUB_RUST"] = "1"
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('USE_RUST', None)
        assert _rust_backend.USE_RUST is True
    
    def test_getattr_use_rust_syncs_env_vars(self):
        """Test that USE_RUST syncs environment variables when enabled."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(rust_backend=True, debug_timing=True, num_threads=4)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('USE_RUST', None)
        _ = _rust_backend.USE_RUST  # Trigger __getattr__
        
        assert os.environ.get("SKRUB_RUST_DEBUG_TIMING") == "1"
        assert os.environ.get("SKRUB_RUST_THREADS") == "4"
    
    def test_getattr_num_threads_from_env(self):
        """Test NUM_THREADS from environment variable."""
        os.environ["SKRUB_RUST_THREADS"] = "8"
        from stratum import _rust_backend
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('NUM_THREADS', None)
        assert _rust_backend.NUM_THREADS == 8
    
    def test_getattr_num_threads_from_config(self):
        """Test NUM_THREADS from config."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(num_threads=4)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('NUM_THREADS', None)
        assert _rust_backend.NUM_THREADS == 4
    
    def test_getattr_num_threads_env_overrides_config(self):
        """Test that environment variable overrides config for NUM_THREADS."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(num_threads=2)
        os.environ["SKRUB_RUST_THREADS"] = "6"
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('NUM_THREADS', None)
        assert _rust_backend.NUM_THREADS == 6
    
    def test_getattr_debug_timing_from_env(self):
        """Test DEBUG_TIMING from environment variable."""
        os.environ["SKRUB_RUST_DEBUG_TIMING"] = "1"
        from stratum import _rust_backend
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        assert _rust_backend.DEBUG_TIMING is True
        
        os.environ["SKRUB_RUST_DEBUG_TIMING"] = "0"
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        assert _rust_backend.DEBUG_TIMING is False
    
    def test_getattr_debug_timing_from_config(self):
        """Test DEBUG_TIMING from config."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(debug_timing=True)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        assert _rust_backend.DEBUG_TIMING is True
        
        set_config(debug_timing=False)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        assert _rust_backend.DEBUG_TIMING is False
    
    def test_getattr_allow_patch_from_env(self):
        """Test ALLOW_PATCH from environment variable."""
        os.environ["SKRUB_RUST_ALLOW_PATCH"] = "1"
        from stratum import _rust_backend
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('ALLOW_PATCH', None)
        assert _rust_backend.ALLOW_PATCH is True
        
        os.environ["SKRUB_RUST_ALLOW_PATCH"] = "0"
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('ALLOW_PATCH', None)
        assert _rust_backend.ALLOW_PATCH is False
    
    def test_getattr_allow_patch_from_config(self):
        """Test ALLOW_PATCH from config."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(allow_patch=True)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('ALLOW_PATCH', None)
        assert _rust_backend.ALLOW_PATCH is True
        
        set_config(allow_patch=False)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('ALLOW_PATCH', None)
        assert _rust_backend.ALLOW_PATCH is False
    
    def test_getattr_unknown_attribute(self):
        """Test that unknown attributes raise AttributeError."""
        from stratum import _rust_backend
        
        with pytest.raises(AttributeError):
            _ = _rust_backend.UNKNOWN_ATTR


class TestTimingFunctions:
    """Test timing utility functions."""
    
    def setup_method(self):
        """Save original environment and reset config."""
        from stratum.config import get_config, set_config
        
        self.original_env = {}
        for key in ["SKRUB_RUST_DEBUG_TIMING"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]
        
        # Save original config state
        original_config = get_config()
        self.original_config = {
            'debug_timing': original_config.get('debug_timing', False),
        }
        
        # Reset config
        set_config(debug_timing=False)
    
    def teardown_method(self):
        """Restore original environment and config."""
        from stratum.config import set_config
        
        # Restore original config
        set_config(debug_timing=self.original_config['debug_timing'])
        
        # Restore original environment
        for key in ["SKRUB_RUST_DEBUG_TIMING"]:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            os.environ[key] = value
    
    def test_start_timing_when_debug_disabled(self):
        """Test start_timing when DEBUG_TIMING is False."""
        from stratum import _rust_backend
        # Clear cached DEBUG_TIMING if it exists
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        
        result = _rust_backend.start_timing()
        assert result is None
    
    def test_start_timing_when_debug_enabled(self):
        """Test start_timing when DEBUG_TIMING is True."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(debug_timing=True)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        
        result = _rust_backend.start_timing()
        assert result is not None
        assert isinstance(result, float)
    
    def test_print_timing_when_debug_disabled(self, capfd):
        """Test print_timing when DEBUG_TIMING is False."""
        from stratum import _rust_backend
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        
        start_time = time.perf_counter()
        _rust_backend.print_timing("test", start_time)
        
        captured = capfd.readouterr()
        assert "[python]" not in captured.out
    
    def test_print_timing_when_debug_enabled(self, capfd):
        """Test print_timing when DEBUG_TIMING is True."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(debug_timing=True)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        
        start_time = _rust_backend.start_timing()
        time.sleep(0.01)  # Small delay to ensure timing > 0
        _rust_backend.print_timing("test_message", start_time)
        
        captured = capfd.readouterr()
        assert "[python]" in captured.out
        assert "test_message" in captured.out
    
    def test_print_timing_with_none_start_time(self, capfd):
        """Test print_timing with None start_time."""
        from stratum.config import set_config
        from stratum import _rust_backend
        
        set_config(debug_timing=True)
        if hasattr(_rust_backend, '__dict__'):
            _rust_backend.__dict__.pop('DEBUG_TIMING', None)
        
        _rust_backend.print_timing("test", None)
        
        captured = capfd.readouterr()
        assert "[python]" not in captured.out


class TestToList:
    """Test _to_list function."""
    
    def test_to_list_with_pandas_series(self):
        """Test _to_list with pandas Series."""
        from stratum import _rust_backend
        
        series = pd.Series([1, 2, 3, 4, 5])
        result = _rust_backend._to_list(series)
        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)
    
    def test_to_list_with_pandas_series_strings(self):
        """Test _to_list with pandas Series containing strings."""
        from stratum import _rust_backend
        
        series = pd.Series(["foo", "bar", "baz"])
        result = _rust_backend._to_list(series)
        assert result == ["foo", "bar", "baz"]
    
    def test_to_list_with_regular_list(self):
        """Test _to_list with regular list."""
        from stratum import _rust_backend
        
        data = [1, 2, 3, 4, 5]
        result = _rust_backend._to_list(data)
        assert result == [1, 2, 3, 4, 5]
        assert result is not data  # Should be a new list
    
    def test_to_list_with_tuple(self):
        """Test _to_list with tuple."""
        from stratum import _rust_backend
        
        data = (1, 2, 3, 4, 5)
        result = _rust_backend._to_list(data)
        assert result == [1, 2, 3, 4, 5]
    
    def test_to_list_with_polars_series(self):
        """Test _to_list with polars Series."""
        import polars as pl
        from stratum import _rust_backend
        
        series = pl.Series([1, 2, 3, 4, 5])
        result = _rust_backend._to_list(series)
        assert result == [1, 2, 3, 4, 5]

