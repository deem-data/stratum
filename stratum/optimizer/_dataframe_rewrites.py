from __future__ import annotations

from dataclasses import dataclass

from stratum.optimizer._map_rewrites import fuse_assign_maps
from stratum.optimizer.logical._ops import Op
from stratum.utils._utils import start_time, log_time


@dataclass(frozen=True, slots=True)
class DataframeRewritesConfig:
    fuse_assign_maps: bool = True


def dataframe_rewrites(root: Op, config: DataframeRewritesConfig) -> Op:
    start = start_time()
    if config.fuse_assign_maps:
        root = fuse_assign_maps(root)
    log_time("dataframe_rewrites", start)
    return root
