import logging
from time import perf_counter

logger = logging.getLogger(__name__)

# Utility function to start timer
def start_time():
   return perf_counter()

# Utility function to log time
def log_time(msg, start_time):
    logger.info(f"{msg}: {(perf_counter() - start_time):.2f} seconds")