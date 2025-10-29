import logging

from skrub import DataOp

from stratum.logical_optimizer.utils import update_data_op, hash_data_op, equals_data_op

logger = logging.getLogger(__name__)

def apply_cse(data_op: DataOp, nodes: dict[int, DataOp], order: list[int],
              parents: dict[int, list[int]]):
    table = CSETable()

    for current_node in order:
        current_parent_ids = parents.get(current_node, [])
        current_op = nodes[current_node]
        cs_id, cs_node = table.get(current_op)

        if cs_node is None:
            table.put(current_node, current_op)
        elif cs_id != current_node:
            logger.debug(f"We can eliminate the current node [{current_op}] with equivalent node [{cs_node}]")
            for parent_id in current_parent_ids:
                update_data_op(nodes[parent_id], current_op, cs_node)

            if logger.isEnabledFor(logging.DEBUG):
                data_op.skb.draw_graph().open()


class CSETable:
    """
    Common Subexpression Elimination (CSE) table for Skrub DataOps.

    This table stores previously encountered DataOp nodes keyed by a
    structure-aware hash (via `hash_data_op`). It enables detection of
    equivalent nodes during bottom-up traversal of a computation graph.

    If two DataOps are semantically equivalent (per `equals_data_op`),
    they will map to the same table entry and can be merged.
    """
    def __init__(self):
        # Maps hash -> list[DataOp] (to handle hash collisions safely)
        self.table: dict[int, list[tuple[int, DataOp]]] = {}

    def put(self, id: int, op: DataOp):
        """
        Add a DataOp to the table.

        If an equivalent op is already present, this does nothing.
        """
        h = hash_data_op(op)
        bucket = self.table.setdefault(h, [])
        # Avoid duplicates if equivalent op already exists
        for _, existing in bucket:
            if equals_data_op(existing, op):
                return  # Already in table
        bucket.append((id, op))

    def get(self, op: DataOp):
        """
        Retrieve an equivalent DataOp from the table, if it exists.

        Parameters
        ----------
        op : DataOp
            Node to look up.

        Returns
        -------
        DataOp | None
            The equivalent DataOp, or None if not found.
        """
        h = hash_data_op(op)
        bucket = self.table.get(h, [])
        for id, existing in bucket:
            if equals_data_op(existing, op):
                return id, existing
        return None, None

    def delete(self, op: DataOp):
        """
        Remove an equivalent DataOp from the table, if it exists.

        Parameters
        ----------
        op : DataOp
            Node to remove.

        Returns
        -------
        bool
            True if an equivalent DataOp was found and removed, False otherwise.
        """
        h = hash_data_op(op)
        bucket = self.table.get(h)
        if not bucket:
            return False

        # Iterate and remove first equivalent match
        for i, (id, existing) in enumerate(bucket):
            if equals_data_op(existing, op):
                del bucket[i]
                # If bucket becomes empty, clean up the hash key
                if not bucket:
                    del self.table[h]
                return True
        return False
