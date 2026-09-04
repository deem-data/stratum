# Implementation selection mutates the operator's class in place

Implementation selection replaces an abstract physical operator with a concrete one by
assigning `op.__class__ = impl.impl_class` rather than constructing a replacement node.
Node identity is the key the rest of the system runs on: the buffer pool indexes
intermediates by operator identity, and every `inputs`/`outputs` edge holds a direct
reference. Mutating the class preserves identity, so nothing has to be rewired.

## Considered options

**Construct a new node and rewire.** The conventional approach, and rejected because it
makes selection O(edges) instead of O(nodes) and forces a remap of every buffer-pool
entry and edge reference. It also loses the property that concrete implementations
sharing the abstract operator's fields need no constructor of their own.

**Keep the abstract operator and dispatch on backend at execution time.** Rejected
because it puts backend-selection control flow into the hot path. Removing that control
flow from execution is the reason the physical layer exists: a concrete physical
operator runs on exactly one backend and its `process` contains no selection logic.

## Consequences

Concrete implementations must be field-compatible with their abstract operator, since
they inherit its instance state as-is. An implementation needing extra plan-time state
precomputes it in `on_impl_selected`. Assigning `__class__` is exotic Python and reads
as a bug on first encounter, which is why it is recorded here as well as in
`stratum/optimizer/physical/_physical_ops.py`.
