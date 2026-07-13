"""
Rust physical operator registration.

How to add a new Rust kernel:

1. Keep the Python/Rust bridge in `stratum/adapters/<name>.py`.
   The adapter file should expose a small Python class or function that calls the
   compiled Rust backend and a support check such as:
   `supports_rust_<name>(estimator_or_op) -> tuple[bool, str]`.

2. Add an execution wrapper in this file.
   The wrapper receives the logical op, runtime mode, and resolved inputs:
   `execute(op, mode, inputs)`. It should adapt the logical op to the Rust bridge
   and then call the existing op execution path or the Rust bridge directly.

3. Add one `RustKernelRegistration` entry to `RUST_KERNELS`.
   The entry must name the logical op type, input/output formats, support check,
   and execute wrapper. Use `backend_name="rust"` only through this file.

4. Keep fallback behavior out of the Rust `execute` wrapper.
   `supports` decides whether a Rust candidate is valid. If `execute` is called,
   the selector has chosen Rust and unexpected fallback should be treated as a
   bug unless the adapter has a documented numerical fallback.
"""

#TODO: Cost and memory are placeholders for now. Replace `default_cost`
# and `default_exec_mem` with kernel-specific estimates as the cost model lands.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from stratum.adapters.one_hot_encoder import (
    RustyOneHotEncoder,
    supports_rust_one_hot_encoder,
)
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    supports_rust_string_encoder,
)
from stratum.optimizer.ir._ops import Op, TransformerOp
from stratum.optimizer.physical._registry import PhysicalImpl, PhysicalRegistry


CostFn = Callable[[Op, Any], float]
ExecMemFn = Callable[[Op, Any], int]
ExecuteFn = Callable[[Op, str, list[Any]], Any]
SupportsFn = Callable[[Op], bool]


@dataclass(frozen=True, slots=True)
class RustKernelRegistration:
    name: str
    logical_op_type: type[Op]
    input_format: str
    output_format: str
    supports: SupportsFn # Function to check if the kernel supports the input parameters
    execute: ExecuteFn   # Function to execute the kernel
    cost: CostFn
    exec_mem: ExecMemFn

    def as_physical_impl(self) -> PhysicalImpl:
        return PhysicalImpl(
            logical_op_type=self.logical_op_type,
            backend_name="rust",
            input_format=self.input_format,
            output_format=self.output_format,
            supports=self.supports,
            cost=self.cost,
            exec_mem=self.exec_mem,
            execute=self.execute,
        )


def default_cost(op: Op, stats: Any) -> float:
    return 1.0


def default_exec_mem(op: Op, stats: Any) -> int:
    return 0


# FIXME: This initialization of Rust operators should happen before runtime.
def _as_rusty_one_hot_encoder(estimator):
    if isinstance(estimator, RustyOneHotEncoder):
        rusty = estimator
    else:
        params = estimator.get_params(deep=False)
        rusty = RustyOneHotEncoder(**params)
    rusty._stratum_force_rust = True
    return rusty


def _supports_one_hot_encoder_op(op: Op) -> bool:
    if not isinstance(op, TransformerOp):
        return False
    supported, _ = supports_rust_one_hot_encoder(op.original_estimator)
    return supported


def _execute_one_hot_encoder(op: Op, mode: str, inputs: list[Any]) -> Any:
    if mode == "fit_transform":
        op.original_estimator = _as_rusty_one_hot_encoder(op.original_estimator)
        op.estimator = _as_rusty_one_hot_encoder(op.estimator)
    return op.process(mode, inputs)


# FIXME: This initialization of Rust operators should happen before runtime.
def _as_rusty_string_encoder(estimator):
    if isinstance(estimator, RustyStringEncoder):
        rusty = estimator
    else:
        params = estimator.get_params(deep=False)
        rusty = RustyStringEncoder(**params)
    rusty._stratum_force_rust = True
    return rusty


def _supports_string_encoder_op(op: Op) -> bool:
    if not isinstance(op, TransformerOp):
        return False
    supported, _ = supports_rust_string_encoder(op.original_estimator)
    return supported


def _execute_string_encoder(op: Op, mode: str, inputs: list[Any]) -> Any:
    if mode == "fit_transform":
        op.original_estimator = _as_rusty_string_encoder(op.original_estimator)
        op.estimator = _as_rusty_string_encoder(op.estimator)
    return op.process(mode, inputs)


RUST_KERNELS: tuple[RustKernelRegistration, ...] = (
    RustKernelRegistration(
        name="one_hot_encoder",
        logical_op_type=TransformerOp,
        input_format="frame",
        output_format="matrix",
        supports=_supports_one_hot_encoder_op,
        execute=_execute_one_hot_encoder,
        cost=default_cost,
        exec_mem=default_exec_mem,
    ),
    RustKernelRegistration(
        name="string_encoder",
        logical_op_type=TransformerOp,
        input_format="frame",
        output_format="frame",
        supports=_supports_string_encoder_op,
        execute=_execute_string_encoder,
        cost=default_cost,
        exec_mem=default_exec_mem,
    ),
)


def register_rust_physical_operators(registry: PhysicalRegistry) -> PhysicalRegistry:
    for kernel in RUST_KERNELS:
        registry.register(kernel.as_physical_impl())
    return registry
