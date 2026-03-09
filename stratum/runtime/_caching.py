from fileinput import filename
import json
import os
import logging
import polars as pl
import pandas as pd
from time import perf_counter


logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".stratum", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
INTERMEDIATES_DIR = os.path.join(CACHE_DIR, "intermediates")
if not os.path.exists(INTERMEDIATES_DIR):
    os.makedirs(INTERMEDIATES_DIR)


class Cache:
    def __init__(self):
        self.cache = {}
        self.timings = []
        # try to load cache from file
        if os.path.exists(os.path.join(CACHE_DIR, "cache.json")):
            logger.info(f"Loading cache from {os.path.join(CACHE_DIR, 'cache.json')}")
            t0 = perf_counter()
            with open(os.path.join(CACHE_DIR, "cache.json"), "r") as f:
                tmp_cache = json.load(f)
                for key,(file_name, converted) in tmp_cache.items():
                    key = int(key)
                    self.cache[key] = read_value(file_name, converted)
            t1 = perf_counter()
            duration = t1 - t0
            logger.info(f"Cache loaded in {duration} seconds")
            self.timings.append(("load_cache", duration))
            
        # Cache operation counters
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.hit_time = 0.0
        self.set_time = 0.0

    def get(self, key):
        t0 = perf_counter()
        result = self.cache.get(key)
        t1 = perf_counter()
        duration = t1 - t0
        
        if result is not None:
            self.hit_count += 1
            self.hit_time += duration
        else:
            self.miss_count += 1
        
        return result

    def set(self, key, value):
        t0 = perf_counter()
        self.cache[key] = value
        t1 = perf_counter()
        duration = t1 - t0
        self.set_count += 1
        self.set_time += duration

    def persist(self):
        logger.info(f"Saving cache to {os.path.join(CACHE_DIR, 'cache.json')}")
        t0 = perf_counter()
        file_name_cache = {}
        for key, value in self.cache.items():
            converted = isinstance(value, pd.DataFrame)
            if not check_if_intermediate_exists(key):
                write_value(key, value)
            else:
                logger.debug(f"Intermediate {key} already exists, skipping write")
            file_name_cache[key] = (make_intermediate_file_name(key), converted)
        # clear existing cache file
        if os.path.exists(os.path.join(CACHE_DIR, "cache.json")):
            os.remove(os.path.join(CACHE_DIR, "cache.json"))
        # write new cache file
        with open(os.path.join(CACHE_DIR, "cache.json"), "w") as f:
            json.dump(file_name_cache, f)
        t1 = perf_counter()
        duration = t1 - t0
        logger.info(f"Cache saved in {duration} seconds")
        self.timings.append(("save_cache", duration))
        del self.cache


def make_intermediate_file_name(key):
    return os.path.join(INTERMEDIATES_DIR, f"{key}.parquet")

def check_if_intermediate_exists(key):
    return os.path.exists(make_intermediate_file_name(key))

def read_value(file_name, converted=False):
    if not os.path.exists(file_name):
        raise RuntimeError(f"Intermediate {file_name} not found. Cache is corrupted. Please do 'rm -rf {CACHE_DIR}' and run your code again.")
    df = pl.read_parquet(file_name)
    if converted:
        df = df.to_pandas()
    return df

def write_value(key, value):
    if isinstance(value, pd.DataFrame):
        value = pl.from_pandas(value)
    if isinstance(value, pl.DataFrame):
        with open(make_intermediate_file_name(key), "wb") as f:
            value.write_parquet(f)
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")