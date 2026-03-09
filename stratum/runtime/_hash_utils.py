import hashlib
from sklearn.base import BaseEstimator
from skrub import TableVectorizer

def _stable_hash_tuple(items):
    """
    Hash a tuple/sequence of items deterministically across processes.
    """
    # Create a deterministic hash by hashing the stable hashes of each item
    hash_values = tuple(stable_hash(item) for item in items)
    # Convert tuple of integers to bytes for hashing
    # Use struct.pack or a simple byte representation
    # For simplicity, use a delimiter-separated string representation
    byte_data = b'|'.join(str(h).encode('utf-8') for h in hash_values)
    return int.from_bytes(hashlib.sha256(byte_data).digest()[:8], byteorder='big')


def hash_estimator(est: BaseEstimator) -> int:
    """
    Hash an estimator.
    """
    param_hashes = []
    items = list(est.get_params().items())
    for key, value in items:
        if key != "fitted_":
            if isinstance(value, BaseEstimator):
                param_hashes.append((key, hash_estimator(value)))
            else:
                param_hashes.append(((key, stable_hash(value))))
    if "fitted_" in items:
        param_hashes.append(("fitted_", stable_hash(est.fitted_)))
    return _stable_hash_tuple(param_hashes)

def stable_hash(obj):
    if isinstance(obj, str):
        # Use SHA256 for stable hashing across processes
        return int.from_bytes(hashlib.sha256(obj.encode('utf-8')).digest()[:8], byteorder='big')
    elif isinstance(obj, BaseEstimator):
        return hash_estimator(obj)
    elif isinstance(obj, (int, float, bool, type(None))):
        # These types have stable representations
        # Convert to string and hash for consistency
        return int.from_bytes(hashlib.sha256(repr(obj).encode('utf-8')).digest()[:8], byteorder='big')
    elif isinstance(obj, list):
        return _stable_hash_tuple(obj)
    elif isinstance(obj, tuple):
        return _stable_hash_tuple(obj)
    elif isinstance(obj, dict):
        # Sort items by key hash for deterministic ordering
        sorted_items = sorted(obj.items(), key=lambda x: stable_hash(x[0]))
        return _stable_hash_tuple((stable_hash(key), stable_hash(value)) for key, value in sorted_items)
    else:
        # For other types, use repr() to get a stable string representation
        return int.from_bytes(hashlib.sha256(repr(obj).encode('utf-8')).digest()[:8], byteorder='big')