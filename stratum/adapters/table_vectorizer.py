"""Fused TableVectorizer implementations.

This module contains two implementations for the TableVectorizer operator:
a fused implementation that avoids dataframe wrapping and bookkeeping at each ApplyToEachCol step,
and a multithreaded fused implementation that uses eligible Rust leaf encoders.

"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Lock

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from skrub import DatetimeEncoder as _DatetimeEncoder
from skrub import StringEncoder as _StringEncoder
from skrub import TableVectorizer as _SkrubTableVectorizer
from skrub import _dataframe as sbd
from skrub._apply_to_each_col import _fit_transform_column, _transform_column
from skrub._check_input import CheckInputDataFrame
from skrub._clean_categories import CleanCategories
from skrub._clean_null_strings import CleanNullStrings
from skrub._drop_uninformative import DropUninformative
from skrub._join_utils import pick_column_names
from skrub._select_cols import Drop
from skrub._table_vectorizer import (
    PassThrough as _PassThrough,
    ShortReprDict as _ShortReprDict,
    _check_transformer,
)
from skrub._to_datetime import ToDatetime
from skrub._to_float import ToFloat
from skrub._to_str import ToStr

from stratum.adapters.one_hot_encoder import (
    RustyOneHotEncoder,
    supports_rust_one_hot_encoder,
)
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    supports_rust_string_encoder,
)
from stratum.utils._utils import log_time, start_time

__all__ = ["ExactFusedTableVectorizer", "StratumFusedTableVectorizer"]

# =========== Dataclasses to hold runtime state ====================

@dataclass
class _ColumnPlan:
    """Fitted execution state for one cleaned input column."""

    input_name: str
    preprocessing: list = field(default_factory=list)
    kind: str | None = None
    main_transformer: object | None = None
    output_names: list[str] = field(default_factory=list)
    postprocessing: dict[str, object] = field(default_factory=dict)
    dropped_by_preprocessing: bool = False


@dataclass
class _FrameColumn:
    """A column currently present in the logical frame, with its owner."""

    owner: _ColumnPlan
    column: pd.Series


@dataclass(frozen=True)
class _ColumnResult:
    """Main-encoder result returned without mutating shared fitted state."""

    input_position: int
    output_columns: list
    fitted_transformer: object


class _FusedTableVectorizer(_SkrubTableVectorizer):
    """This class contains a tight implementation of a fused TableVectorizer operator.
       - Results match skrub's table vectorizer for supported parameters
       - It is intentionally conservative. Leaf encoders stay unchanged.
       - Only optimizes orchestration by avoiding dataframe wrapping & bookkeeping at each ApplyToEachCol step.
       - It uses skrub internals (_fit_transform_column, _transform_column, selectors, name collision tools)
         so semantics stay aligned with Skrub’s reference behavior.
    """

    _MAIN_KINDS = ("numeric", "datetime", "low_cardinality", "high_cardinality")

    # =========== Support and config checks ===================

    @classmethod
    def supports(cls, estimator, ctx=None) -> bool:
        """Return whether ``estimator`` belongs to the initial fused subset."""
        if hasattr(estimator, "original_estimator"):
            estimator = estimator.original_estimator
        if not isinstance(estimator, _SkrubTableVectorizer):
            return False
        if estimator.specific_transformers:
            return False
        return all(
            cls._supports_role(name, getattr(estimator, name))
            for name in cls._MAIN_KINDS
        )

    @staticmethod
    def _supports_role(name, transformer) -> bool:
        if isinstance(transformer, str):
            return transformer in ("drop", "passthrough")
        if name in ("numeric", "datetime"):
            return isinstance(transformer, (_PassThrough, _DatetimeEncoder, Drop))
        if name == "low_cardinality":
            return type(transformer) in (OneHotEncoder, _PassThrough, Drop)
        if name == "high_cardinality":
            return type(transformer) in (_StringEncoder, _PassThrough, Drop)
        return False

    def _check_supported_configuration(self):
        if self.specific_transformers:
            raise ValueError(
                "_FusedTableVectorizer does not support 'specific_transformers'."
            )
        unsupported = [
            name
            for name in self._MAIN_KINDS
            if not self._supports_role(name, getattr(self, name))
        ]
        if unsupported:
            raise ValueError(
                "_FusedTableVectorizer does not support transformer roles: "
                f"{unsupported}."
            )

    @staticmethod
    def _check_pandas_input(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Fused TableVectorizer implementations only support pandas "
                "DataFrame input."
            )

    # =========== Helper methods (thin wrappers around skrub) ===========

    @staticmethod
    def _fit_column(column, transformer, y, allow_reject):
        """Fit one leaf through Skrub's exact per-column reference helper."""
        _, output_columns, fitted = _fit_transform_column(
            column,
            y,
            [sbd.name(column)],
            transformer,
            allow_reject,
            {},
        )
        return output_columns, fitted

    @staticmethod
    def _transform_column(column, fitted):
        """Transform one leaf through Skrub's exact per-column helper."""
        return _transform_column(column, fitted, {})

    @staticmethod
    def _column_names(columns):
        return [sbd.name(column) for column in columns]

    @staticmethod
    def _rename_columns(columns, names):
        return [sbd.rename(column, name) for column, name in zip(columns, names)]

    def _preprocessors(self):
        return [
            CleanNullStrings(null_strings=self.null_strings),
            DropUninformative(
                drop_null_fraction=self.drop_null_fraction,
                drop_if_constant=self.drop_if_constant,
                drop_if_unique=self.drop_if_unique,
            ),
            ToDatetime(format=self.datetime_format),
            ToFloat(),
            CleanCategories(),
            ToStr(),
        ]

    def _main_transformers(self):
        # _check_transformer is Skrub's own normalization for drop/passthrough
        # and performs the same clone that TableVectorizer performs when its
        # pipeline is built.
        return {
            name: _check_transformer(getattr(self, name))
            for name in self._MAIN_KINDS
        }

    def _fit_main_columns(self, prepared, main_transformers, y, plans):
        """Fit main encoders in the serial runtime's deterministic kind order."""
        positions = {id(plan): position for position, plan in enumerate(plans)}
        results = []
        for kind in self._MAIN_KINDS:
            for frame_column in prepared:
                if frame_column.owner.kind != kind:
                    continue
                output_columns, fitted = self._fit_column(
                    frame_column.column,
                    main_transformers[kind],
                    y,
                    allow_reject=False,
                )
                results.append(
                    _ColumnResult(
                        input_position=positions[id(frame_column.owner)],
                        output_columns=output_columns,
                        fitted_transformer=fitted,
                    )
                )
        return results

    def _transform_main_columns(self, prepared):
        """Run fitted main encoders serially and return input-ordered results."""
        return [
            _ColumnResult(
                input_position=position,
                output_columns=self._transform_column(
                    frame_column.column,
                    frame_column.owner.main_transformer,
                ),
                fitted_transformer=frame_column.owner.main_transformer,
            )
            for position, frame_column in prepared
        ]


    # =========== Full training pipeline =========================

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the fused runtime and return one assembled dataframe."""
        self._check_pandas_input(X)
        self._check_supported_configuration()

        # The input checker is the one unavoidable full-frame preprocessing.
        # It also owns Skrub's column-name warnings and cleaning.
        t = start_time()
        input_checker = CheckInputDataFrame()
        checked = input_checker.fit_transform(X)
        log_time("fit_transform: input_checker", t)

        main_transformers = self._main_transformers()

        plans = [
            _ColumnPlan(input_name=name)
            for name in input_checker.feature_names_out_
        ]

        # ============ Preprocessing block =======================
        preprocessors = self._preprocessors()
        prepared = []
        t_preprocess = start_time()
        for plan in plans:
            column = sbd.col(checked, plan.input_name)
            for preprocessor in preprocessors:
                output_columns, fitted = self._fit_column(
                    column, preprocessor, y, allow_reject=True
                )
                if fitted is None:
                    continue
                plan.preprocessing.append(fitted)
                if not output_columns:
                    plan.dropped_by_preprocessing = True
                    break
                # The default preprocessing chain is single-column.
                if len(output_columns) != 1:
                    raise ValueError(
                        "Fused TableVectorizer preprocessing produced more than "
                        f"one output column for {plan.input_name!r}."
                    )
                column = output_columns[0]

            # Result: prepared list of _FrameColumns that survived preprocessing
            if not plan.dropped_by_preprocessing:
                prepared.append(_FrameColumn(owner=plan, column=column))
        log_time("fit_transform: preprocessing block (all columns)", t_preprocess)

        # ============ Column kind classification ===============
        for frame_column in prepared:
            column = frame_column.column
            if sbd.is_numeric(column):
                frame_column.owner.kind = "numeric"
            elif sbd.is_any_date(column):
                frame_column.owner.kind = "datetime"
            else:
                try:
                    is_low_cardinality = (
                        sbd.n_unique(column) < self.cardinality_threshold
                    )
                except Exception:
                    # Following skrub, leave unhashable object columns eligible
                    # for the high-cardinality encoder.
                    is_low_cardinality = False
                frame_column.owner.kind = (
                    "low_cardinality" if is_low_cardinality else "high_cardinality"
                )

        frame = prepared
        kind_to_columns = {name: [] for name in self._MAIN_KINDS}

        # ======== Main transformers block (avoid intermediates) ===========
        t_main = start_time()
        main_results = {
            result.input_position: result
            for result in self._fit_main_columns(
                prepared, main_transformers, y, plans
            )
        }
        plan_positions = {id(plan): position for position, plan in enumerate(plans)}
        for kind in self._MAIN_KINDS:
            forbidden_names = {sbd.name(frame_column.column) for frame_column in frame}
            updated_frame = []
            for frame_column in frame:
                plan = frame_column.owner
                if plan.kind != kind:
                    updated_frame.append(frame_column)
                    continue

                result = main_results[plan_positions[id(plan)]]
                output_columns = result.output_columns
                fitted = result.fitted_transformer
                # Resolve and rename output names
                suggested_names = self._column_names(output_columns)
                output_names = pick_column_names(
                    suggested_names,
                    forbidden_names - {sbd.name(frame_column.column)},
                )
                output_columns = self._rename_columns(output_columns, output_names)
                forbidden_names.update(output_names)

                # Store fitted transformers and outputs names in the plan
                plan.main_transformer = fitted
                plan.output_names = output_names
                kind_to_columns[kind].append(plan.input_name)
                updated_frame.extend(
                    _FrameColumn(owner=plan, column=column)
                    for column in output_columns
                )
            frame = updated_frame
        log_time("fit_transform: main transformers block (all kinds)", t_main)

        # ============ Postprocessing block ===============
        t = start_time()
        postprocessor = ToFloat()
        forbidden_names = {sbd.name(frame_column.column) for frame_column in frame}
        updated_frame = []
        for frame_column in frame:
            plan = frame_column.owner
            column = frame_column.column
            if sbd.is_categorical(column):
                updated_frame.append(frame_column)
                continue

            output_columns, fitted = self._fit_column(
                column, postprocessor, y, allow_reject=True
            )
            if fitted is None:
                updated_frame.append(frame_column)
                continue

            suggested_names = self._column_names(output_columns)
            output_names = pick_column_names(
                suggested_names,
                forbidden_names - {sbd.name(column)},
            )
            output_columns = self._rename_columns(output_columns, output_names)
            forbidden_names.update(output_names)
            for output_column, output_name in zip(output_columns, output_names):
                plan.postprocessing[output_name] = fitted
                updated_frame.append(
                    _FrameColumn(owner=plan, column=output_column)
                )
        frame = updated_frame
        log_time("fit_transform: postprocessing block", t)

        # ============ Metadata reconstruction ===============
        self._input_checker = input_checker
        self._column_plans = plans
        self.feature_names_in_ = list(input_checker.feature_names_out_)
        self.n_features_in_ = len(self.feature_names_in_)
        self.all_outputs_ = [sbd.name(frame_column.column) for frame_column in frame]

        self.input_to_outputs_ = {}
        for plan in plans:
            if plan.dropped_by_preprocessing:
                self.input_to_outputs_[plan.input_name] = [plan.input_name]
            else:
                self.input_to_outputs_[plan.input_name] = list(plan.output_names)

        self.output_to_input_ = {
            output: input_name
            for input_name, outputs in self.input_to_outputs_.items()
            for output in outputs
        }

        self.kind_to_columns_ = {
            kind: list(kind_to_columns[kind]) for kind in self._MAIN_KINDS
        }
        self.kind_to_columns_["specific"] = []
        self.column_to_kind_ = {
            input_name: kind
            for kind, input_names in self.kind_to_columns_.items()
            for input_name in input_names
        }

        self.transformers_ = {
            plan.input_name: plan.main_transformer
            for kind in self._MAIN_KINDS
            for plan in plans
            if plan.kind == kind and plan.main_transformer is not None
        }

        self.all_processing_steps_ = {}
        for plan in plans:
            steps = list(plan.preprocessing)
            if plan.main_transformer is not None:
                steps.append(plan.main_transformer)
            postprocessing = {
                output_name: plan.postprocessing[output_name]
                for output_name in plan.output_names
                if output_name in plan.postprocessing
            }
            if postprocessing:
                steps.append(_ShortReprDict(postprocessing))
            self.all_processing_steps_[plan.input_name] = steps


        result = self._assemble(frame, checked)
        return result


    # =========== Inference pipeline ============================

    def transform(self, X):
        """Transform with the fitted ordered column plan."""
        check_is_fitted(self, "transformers_")
        self._check_pandas_input(X)

        t = start_time()
        checked = self._input_checker.transform(X)
        log_time("transform: input_checker", t)

        frame = []
        t_preprocess = start_time()
        prepared = []
        for position, plan in enumerate(self._column_plans):
            column = sbd.col(checked, plan.input_name)
            dropped = False
            for fitted in plan.preprocessing:
                output_columns = self._transform_column(column, fitted)
                if not output_columns:
                    dropped = True
                    break
                if len(output_columns) != 1:
                    raise ValueError(
                        "Fused TableVectorizer preprocessing produced more than "
                        f"one output column for {plan.input_name!r}."
                    )
                column = output_columns[0]
            if dropped or plan.dropped_by_preprocessing:
                continue
            prepared.append((position, _FrameColumn(owner=plan, column=column)))

        main_results = {
            result.input_position: result
            for result in self._transform_main_columns(prepared)
        }
        for position, frame_column in prepared:
            plan = frame_column.owner
            output_columns = main_results[position].output_columns
            output_columns = self._rename_columns(
                output_columns, plan.output_names
            )

            for output_column, output_name in zip(
                output_columns, plan.output_names
            ):
                postprocessor = plan.postprocessing.get(output_name)
                if postprocessor is not None:
                    post_columns = self._transform_column(
                        output_column, postprocessor
                    )
                    post_columns = self._rename_columns(
                        post_columns, [output_name]
                    )
                    if post_columns:
                        output_column = post_columns[0]
                frame.append(
                    _FrameColumn(owner=plan, column=output_column)
                )
        log_time("transform: all columns (preprocess + main + postprocess)", t_preprocess)

        result = self._assemble(frame, checked)
        return result

    @staticmethod
    def _assemble(frame, reference):
        columns = [frame_column.column for frame_column in frame]
        result = sbd.make_dataframe_like(reference, columns)
        return sbd.copy_index(reference, result)

#========================================================================================


class ExactFusedTableVectorizer(_FusedTableVectorizer):
    """Internal exact Python validation runtime for Skrub TableVectorizer."""

#========================================================================================

class StratumFusedTableVectorizer(_FusedTableVectorizer):
    """Multithreaded fused implementation using eligible Rust leaf encoders.

    This class overrides the main column fitting and transformation to use
    one lazily created, reusable thread pool for eligible Rust leaf encoders.
    Currently, it relies on heuristic-based thread assignment:
    column-level parallelism: up to 4; heavy string-encoder jobs: up to 2.
    The remaining methods are single-threaded as the preprocessing and
    postprocessing steps are lightweight and fast.
    TODO: Cost-based thread assignment and dynamic scheduling.
    TODO: Future optimizations: 1) parallelize pre and post-processing steps,
      2) skip ToFloat post-processing if already float.
      Currently, these are not applied as pre and post-processing have negligible overheads.
    """
    _COLUMN_PARALLELISM = 4
    _HEAVY_PARALLELISM = 2
    _executor_init_lock = Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the instance pool, creating it on the first async job."""
        executor = getattr(self, "_executor", None)
        if executor is None:
            with self._executor_init_lock:
                executor = getattr(self, "_executor", None)
                if executor is None:
                    executor = ThreadPoolExecutor(
                        max_workers=self._COLUMN_PARALLELISM
                    )
                    self._executor = executor
        return executor

    def close(self) -> None:
        """Shut down the vectorizer's reusable worker pool, if it exists."""
        with self._executor_init_lock:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                self._executor = None
                executor.shutdown(wait=True)

    def __del__(self):
        # Estimators do not have a common lifecycle hook, so release workers
        # when an instance is collected as a fallback for callers that do not
        # explicitly call close().
        # TODO: Move threadpool creation and shutdown to the parallelization planner
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        # ThreadPoolExecutor and its locks are runtime state, not fitted state.
        state = super().__getstate__()
        state.pop("_executor", None)
        return state

    @staticmethod
    def _supports_role(name, transformer) -> bool:
        if name == "low_cardinality" and isinstance(
            transformer, RustyOneHotEncoder
        ):
            return True
        if name == "high_cardinality" and isinstance(
            transformer, RustyStringEncoder
        ):
            return True
        return _FusedTableVectorizer._supports_role(name, transformer)

    @classmethod
    def supports(cls, estimator, ctx=None) -> bool:
        if hasattr(estimator, "original_estimator"):
            estimator = estimator.original_estimator
        if not super().supports(estimator, ctx):
            return False
        return any(
            supported
            for supported, _ in (
                supports_rust_one_hot_encoder(estimator.low_cardinality),
                supports_rust_string_encoder(estimator.high_cardinality),
            )
        )

    def _main_transformers(self):
        transformers = super()._main_transformers()

        low_cardinality = transformers["low_cardinality"]
        if supports_rust_one_hot_encoder(low_cardinality)[0]:
            params = low_cardinality.get_params(deep=False)
            low_cardinality = RustyOneHotEncoder(
                drop=params["drop"],
                dtype=params["dtype"],
                handle_unknown=params["handle_unknown"],
                sparse_output=params["sparse_output"],
            )
            low_cardinality._stratum_force_rust = True
            transformers["low_cardinality"] = low_cardinality

        high_cardinality = transformers["high_cardinality"]
        if supports_rust_string_encoder(high_cardinality)[0]:
            high_cardinality = RustyStringEncoder(
                **high_cardinality.get_params(deep=False)
            )
            high_cardinality._stratum_force_rust = True
            transformers["high_cardinality"] = high_cardinality

        return transformers

    @staticmethod
    def _fit_job(position, column, transformer, y, heavy_slots):
        if isinstance(transformer, _StringEncoder):
            with heavy_slots:
                output_columns, fitted = _FusedTableVectorizer._fit_column(
                    column, transformer, y, allow_reject=False
                )
        else:
            output_columns, fitted = _FusedTableVectorizer._fit_column(
                column, transformer, y, allow_reject=False
            )
        return _ColumnResult(position, output_columns, fitted)

    @staticmethod
    def _transform_job(position, column, transformer, heavy_slots):
        if isinstance(transformer, _StringEncoder):
            with heavy_slots:
                output_columns = _FusedTableVectorizer._transform_column(
                    column, transformer
                )
        else:
            output_columns = _FusedTableVectorizer._transform_column(
                column, transformer
            )
        return _ColumnResult(position, output_columns, transformer)

    def _fit_main_columns(self, prepared, main_transformers, y, plans):
        # Workers own their cloned estimator state. Results are consumed in
        # input order before plans, names, or public metadata are mutated.
        positions = {id(plan): position for position, plan in enumerate(plans)}
        heavy_slots = BoundedSemaphore(self._HEAVY_PARALLELISM)
        jobs: list[_ColumnResult | Future] = []
        transformers = [
            main_transformers[frame_column.owner.kind]
            for frame_column in prepared
        ]
        has_python_string_job = any(
            isinstance(transformer, _StringEncoder)
            and not isinstance(transformer, RustyStringEncoder)
            for transformer in transformers
        )
        native_thread_limit = (
            threadpool_limits(limits=1)
            if has_python_string_job
            else nullcontext()
        )
        executor = None
        with native_thread_limit:
            for frame_column, transformer in zip(
                prepared, transformers, strict=True
            ):
                plan = frame_column.owner
                position = positions[id(plan)]
                if isinstance(transformer, (_PassThrough, Drop)):
                    output_columns, fitted = self._fit_column(
                        frame_column.column,
                        transformer,
                        y,
                        allow_reject=False,
                    )
                    jobs.append(
                        _ColumnResult(position, output_columns, fitted)
                    )
                else:
                    if executor is None:
                        # Lazily create the threadpool on first use
                        executor = self._get_executor()
                    jobs.append(
                        executor.submit(
                            self._fit_job,
                            position,
                            frame_column.column,
                            transformer,
                            y,
                            heavy_slots,
                        )
                    )
            return [
                job.result() if isinstance(job, Future) else job
                for job in jobs
            ]

    def _transform_main_columns(self, prepared):
        heavy_slots = BoundedSemaphore(self._HEAVY_PARALLELISM)
        jobs: list[_ColumnResult | Future] = []
        has_python_string_job = any(
            isinstance(frame_column.owner.main_transformer, _StringEncoder)
            and not isinstance(
                frame_column.owner.main_transformer, RustyStringEncoder
            )
            for _, frame_column in prepared
        )
        native_thread_limit = (
            threadpool_limits(limits=1)
            if has_python_string_job
            else nullcontext()
        )
        executor = None
        with native_thread_limit:
            for position, frame_column in prepared:
                transformer = frame_column.owner.main_transformer
                if isinstance(transformer, (_PassThrough, Drop)):
                    jobs.append(
                        _ColumnResult(
                            position,
                            self._transform_column(
                                frame_column.column, transformer
                            ),
                            transformer,
                        )
                    )
                else:
                    if executor is None:
                        # Lazily create the threadpool on first use
                        executor = self._get_executor()
                    jobs.append(
                        executor.submit(
                            self._transform_job,
                            position,
                            frame_column.column,
                            transformer,
                            heavy_slots,
                        )
                    )
            return [
                job.result() if isinstance(job, Future) else job
                for job in jobs
            ]
