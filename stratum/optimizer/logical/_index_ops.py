"""Logical operations for row labels and proven aggregation keys."""

from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._ops import Op


class IndexAccessOp(Op):
    """Read the actual index of a pandas object.

    This is intentionally pandas-specific: an arbitrary Polars object has no
    index whose labels could be recovered later.
    """

    logical_family = "IndexAccess"
    fields = []

    def __init__(self, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.output_type = OutputType.SERIES


class GroupKeysOp(Op):
    """Return the values of one proven grouping key from an aggregate relation."""

    logical_family = "GroupKeys"
    fields = ["key"]

    def __init__(self, key: str, inputs=None, outputs=None):
        super().__init__(name=key, inputs=inputs, outputs=outputs)
        self.key = key
        self.output_type = OutputType.SERIES
