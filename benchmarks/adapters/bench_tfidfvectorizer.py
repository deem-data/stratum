"""End-to-end benchmark of Stratum's Rust-backed TfidfVectorizer.

uniform 1-99 words per document, sampled from a fixed case-insensitive
unique vocabulary of ASCII words, with a trailing period.

Both legs use ``stratum.TfidfVectorizer`` (the patched adapter):

* ``version=rust``     → ``rust_backend=True``  (Rust kernel)
* ``version=sklearn``  → ``rust_backend=False`` (sklearn fallback)

The benchmark measures the public API end to end, including all steps of the Kernel:
pandas to list materialization, adapter validation, ASCII scanning, FFI, fitted
Python state, NumPy handoff, and SciPy CSR wrapping.

Thread limits are set in each worker's environment before its Python
interpreter starts. The Rust thread count is also set through
``stratum.set_config(num_threads=...)``.
sklearn ignores thread count and is run once per case at
``n_jobs=1``.

Timing, memory, and stage profiling use independent worker processes:

* timing reports the median of fresh-estimator repetitions;
* memory reports the peak of one unprofiled fit after resetting Linux VmHWM;
* profiling runs one separate Rust fit with ``debug_timing`` enabled.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_TIMING_REPEATS = 5
_DATASET_LENGTHS = [10_000, 100_000, 1_000_000]
_N_UNIQUE_WORDS = [1000, 10_000]
_N_JOBS = [1, 2, 4, 8]
_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "stratum-benchmark-matplotlib"


def benchmark_grid() -> tuple[list[tuple[int, int]], list[int]]:
    """Return ((dataset_length, n_unique_words), …) and n_jobs values."""
    cases = [
        (length, words)
        for length in _DATASET_LENGTHS
        for words in _N_UNIQUE_WORDS
    ]
    return cases, list(_N_JOBS)

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def _tag_slug(tag: str) -> str:
    """Filesystem-safe tag for result filenames."""
    slug = re.sub(r"[^\w.-]+", "_", tag.strip())
    return slug or "untagged"


def _csv_path(tag: str) -> Path:
    return RESULTS_DIR / f"macrobenchmark_{_tag_slug(tag)}.csv"

# Matches util::print_timing / adapters' rb.print_timing when following
# DEBUG_TIMING.md. Stage names may contain spaces.
_RUST_STAGE_RE = re.compile(
    r"^\[rust\]\s+(?P<stage>.+?):\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\s*$"
)
_PYTHON_STAGE_RE = re.compile(
    r"^\[python\]\s+(?P<stage>.+?):\s*(?P<sec>\d+(?:\.\d+)?)\s*s\s*$"
)
_STAGE_BEGIN = "STAGE_PROFILE_BEGIN"
_STAGE_END = "STAGE_PROFILE_END"

# Leaf stage order for console / stacked stage plots (non-overlapping; sums ≈ wall).
_STAGE_ORDER = (
    "tv py_materialize",
    "tv ascii_prescan",
    "tv prep",
    "tv ffi_extract",
    "tv pass_a_stats",
    "tv select_vocab",
    "tv pass_b_emit",
    "tv assemble_csr",
    "tv numpy_handoff",
    "tv csr_wrap",
)
# Parent timers; excluded from harvest — use the leaf stages above instead.
_STAGE_SKIP = frozenset({"tv fit_transform total", "tv transform total"})


# Seed and corpus shape defaults.
_CORPUS_SEED = 67
_CORPUS_MAX_WORD_LEN = 10  # token length uniform in [2, max_length)
_CORPUS_MAX_SENTENCE_LEN = 100  # words/doc uniform in [1, max_sentence_len)


def get_synthetic_data(
    n_unique_words: int,
    n_sentences: int,
    *,
    max_length: int = _CORPUS_MAX_WORD_LEN,
    max_sentence_len: int = _CORPUS_MAX_SENTENCE_LEN,
    seed: int = _CORPUS_SEED,
) -> object:
    """Build a Series of sentences with an exact normalized vocabulary."""
    import pandas as pd

    if n_unique_words < 1:
        raise ValueError("n_unique_words must be at least 1")
    if n_sentences < 1:
        raise ValueError("n_sentences must be at least 1")
    if max_length <= 2:
        raise ValueError("max_length must be greater than 2")
    if max_sentence_len <= 1:
        raise ValueError("max_sentence_len must be greater than 1")

    rng = random.Random(seed)
    words: list[str] = []
    normalized_words: set[str] = set()
    while len(words) < n_unique_words:
        word = "".join(
            rng.choices(string.ascii_letters, k=rng.choice(range(2, max_length)))
        )
        normalized = word.lower()
        if normalized in normalized_words:
            continue
        normalized_words.add(normalized)
        words.append(word)

    rows = [
        " ".join(rng.choices(words, k=rng.choice(range(1, max_sentence_len)))) + "."
        for _ in range(n_sentences)
    ]
    return pd.Series(rows, name="text")


def _read_proc_status_kb(keys: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            for key in keys:
                if line.startswith(f"{key}:"):
                    out[key] = int(line.split()[1])
    except OSError:
        pass
    return out


def _current_rss_mb() -> float:
    """Current RSS in MiB (Linux VmRSS when available)."""
    vals = _read_proc_status_kb({"VmRSS"})
    if "VmRSS" in vals:
        return vals["VmRSS"] / 1024.0

    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except ImportError as exc:
        raise RuntimeError(
            "Current RSS requires Linux /proc or the benchmark dependency psutil"
        ) from exc


def _reset_peak_rss() -> None:
    """Reset Linux VmHWM to current RSS before the measured call."""
    try:
        Path("/proc/self/clear_refs").write_text("5\n", encoding="ascii")
    except OSError as exc:
        raise RuntimeError(
            "Exact call-scoped peak RSS requires Linux /proc/self/clear_refs"
        ) from exc


def _csr_nbytes_mb(matrix) -> float:
    """Backing-store size of a scipy CSR (data + indices + indptr) in MiB."""
    return (
        int(matrix.data.nbytes)
        + int(matrix.indices.nbytes)
        + int(matrix.indptr.nbytes)
    ) / (1024.0 * 1024.0)


def _parse_stage_lines(lines: list[str]) -> dict[str, float]:
    """Parse timing lines, optionally clipped to a STAGE_PROFILE_* window."""
    begin = next((i for i, line in enumerate(lines) if _STAGE_BEGIN in line), None)
    end = next((i for i, line in enumerate(lines) if _STAGE_END in line), None)
    if begin is not None and end is not None and end > begin:
        lines = lines[begin + 1 : end]

    stages: dict[str, float] = {}
    for line in lines:
        rust = _RUST_STAGE_RE.match(line.strip())
        if rust:
            stage = rust.group("stage")
            if stage not in _STAGE_SKIP:
                stages[stage] = float(rust.group("ms"))
            continue
        py = _PYTHON_STAGE_RE.match(line.strip())
        if py:
            stage = py.group("stage")
            if stage not in _STAGE_SKIP:
                stages[stage] = float(py.group("sec")) * 1000.0
    return stages


def parse_stage_timings(stderr: str, stdout: str = "") -> dict[str, float]:
    """Parse ``[rust]`` / ``[python]`` debug_timing lines into stage → ms.

    Rust helpers write to stderr; Python helpers write to stdout. Markers are
    emitted on both streams, so each stream is clipped independently and the
    dicts are merged (later values win on duplicate stage names).
    """
    stages = _parse_stage_lines(stderr.splitlines())
    stages.update(_parse_stage_lines(stdout.splitlines()))
    return stages


def _format_stages(stages: dict[str, float]) -> str:
    if not stages:
        return ""
    ordered = [s for s in _STAGE_ORDER if s in stages]
    ordered += sorted(k for k in stages if k not in _STAGE_ORDER)
    return ", ".join(f"{k}={stages[k]:.1f}ms" for k in ordered)


def _annotate_bar_speedups(
    ax,
    x_positions,
    rust_vals,
    sklearn_vals,
    *,
    fontsize: int = 7,
) -> None:
    """Label each rust bar with speedup vs the sklearn baseline (no arrows).

    Labels are rotated 45° so adjacent n_jobs bars of similar height do not
    collide horizontally.
    """
    import numpy as np

    labeled_tops: list[float] = []
    for xi, t, st in zip(x_positions, rust_vals, sklearn_vals):
        if not (np.isfinite(t) and np.isfinite(st) and t > 0):
            continue
        ax.annotate(
            f"{st / t:.1f}×",
            xy=(xi, t),
            xytext=(0, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=fontsize,
            rotation=45,
        )
        labeled_tops.append(float(t))

    # Rotated labels need vertical headroom above the tallest labeled bar.
    if labeled_tops:
        _, y1 = ax.get_ylim()
        needed = max(labeled_tops) * 1.28
        if needed > y1:
            ax.set_ylim(top=needed)


def _set_worker_environment(n_jobs: int) -> None:
    """Install thread limits before importing scientific Python packages."""
    os.environ["SKRUB_RUST_THREADS"] = str(n_jobs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["SKRUB_RUST_DEBUG_TIMING"] = "0"
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))


def _worker_environment(n_jobs: int) -> dict[str, str]:
    """Return the environment used to start a worker interpreter."""
    env = os.environ.copy()
    env.update(
        {
            "SKRUB_RUST_THREADS": str(n_jobs),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "SKRUB_RUST_DEBUG_TIMING": "0",
        }
    )
    env.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
    return env


def _new_vectorizer(TfidfVectorizer, version: str):
    vectorizer = TfidfVectorizer()
    if version == "rust" and not vectorizer._rust_enabled():
        raise RuntimeError(
            "TfidfVectorizer Rust path is not enabled; check rust_backend/"
            "allow_patch and that the extension is built."
        )
    return vectorizer


def _assert_expected_backend(vectorizer, version: str) -> None:
    rust_model_exists = getattr(vectorizer, "_rust_tfidf_model_", None) is not None
    if version == "rust" and not rust_model_exists:
        raise RuntimeError(
            "The Rust benchmark silently fell back to sklearn; no Rust model exists"
        )
    if version == "sklearn" and rust_model_exists:
        raise RuntimeError("The sklearn benchmark unexpectedly created a Rust model")


def _warm_up(TfidfVectorizer, X, version: str) -> None:
    """Warm the selected backend without retaining fitted state or output."""
    X_small = X[: min(1000, len(X))]
    warm_vectorizer = _new_vectorizer(TfidfVectorizer, version)
    warm_result = warm_vectorizer.fit_transform(X_small)
    _assert_expected_backend(warm_vectorizer, version)
    del warm_result, warm_vectorizer, X_small
    gc.collect()


def _array_sha256(*arrays) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(int(contiguous.size).to_bytes(8, "little", signed=False))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _vocabulary_sha256(vocabulary: dict) -> str:
    digest = hashlib.sha256()
    for term, feature in sorted(vocabulary.items()):
        encoded = term.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
        digest.update(int(feature).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _output_signature(matrix, vectorizer) -> dict:
    import numpy as np

    data = np.asarray(matrix.data)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "structure_sha256": _array_sha256(matrix.indices, matrix.indptr),
        "vocabulary_sha256": _vocabulary_sha256(vectorizer.vocabulary_),
        "data_sum": float(np.sum(data, dtype=np.float64)),
        "data_l2_squared": float(np.dot(data, data)),
    }


def _assert_signatures_match(actual: dict, expected: dict, *, context: str) -> None:
    exact_fields = (
        "shape",
        "nnz",
        "vocabulary_size",
        "structure_sha256",
        "vocabulary_sha256",
    )
    differences = [
        f"{field}: {actual.get(field)!r} != {expected.get(field)!r}"
        for field in exact_fields
        if actual.get(field) != expected.get(field)
    ]
    for field in ("data_sum", "data_l2_squared"):
        actual_value = actual.get(field)
        expected_value = expected.get(field)
        if not (
            isinstance(actual_value, (int, float))
            and isinstance(expected_value, (int, float))
            and math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            differences.append(
                f"{field}: {actual_value!r} != {expected_value!r} within tolerance"
            )
    if differences:
        raise RuntimeError(
            f"Output validation failed for {context}:\n  " + "\n  ".join(differences)
        )


def _run_timing_worker(
    TfidfVectorizer,
    X,
    version: str,
    timing_repeats: int,
) -> dict:
    samples_ms: list[float] = []
    output_signature = None
    expected_shape = None
    expected_nnz = None
    expected_vocabulary_size = None
    csr_nbytes_mb = None

    for repeat in range(timing_repeats):
        gc.collect()
        vectorizer = _new_vectorizer(TfidfVectorizer, version)
        t0 = time.perf_counter()
        result = vectorizer.fit_transform(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _assert_expected_backend(vectorizer, version)

        shape = tuple(map(int, result.shape))
        nnz = int(result.nnz)
        vocabulary_size = len(vectorizer.vocabulary_)
        if expected_shape is None:
            expected_shape = shape
            expected_nnz = nnz
            expected_vocabulary_size = vocabulary_size
            csr_nbytes_mb = _csr_nbytes_mb(result)
        else:
            if (
                shape != expected_shape
                or nnz != expected_nnz
                or vocabulary_size != expected_vocabulary_size
            ):
                raise RuntimeError(
                    f"{version} timing repetition {repeat + 1} produced a "
                    "different shape, nnz, or vocabulary size"
                )
        if repeat == timing_repeats - 1:
            # Keep signature work after the final timed call so hashing and
            # floating-point reductions cannot affect later timing samples.
            output_signature = _output_signature(result, vectorizer)
        samples_ms.append(elapsed_ms)
        del result, vectorizer
        gc.collect()

    assert output_signature is not None
    assert csr_nbytes_mb is not None
    return {
        "time_ms": statistics.median(samples_ms),
        "time_samples_ms": samples_ms,
        "csr_nbytes_mb": csr_nbytes_mb,
        "output_signature": output_signature,
    }


def _run_memory_worker(TfidfVectorizer, X, version: str) -> dict:
    vectorizer = _new_vectorizer(TfidfVectorizer, version)
    # Exercise the procfs reset path before establishing the baseline so its
    # own one-time allocations cannot be attributed to fit_transform.
    _reset_peak_rss()
    gc.collect()
    baseline_rss_mb = _current_rss_mb()
    _reset_peak_rss()

    result = vectorizer.fit_transform(X)
    status = _read_proc_status_kb({"VmHWM", "VmRSS"})
    if "VmHWM" not in status or "VmRSS" not in status:
        raise RuntimeError("Call-scoped peak RSS requires Linux VmHWM and VmRSS")
    call_peak_rss_mb = status["VmHWM"] / 1024.0
    rss_after_mb = status["VmRSS"] / 1024.0
    _assert_expected_backend(vectorizer, version)

    return {
        "baseline_rss_mb": baseline_rss_mb,
        "call_peak_rss_mb": call_peak_rss_mb,
        "call_peak_increment_mb": call_peak_rss_mb - baseline_rss_mb,
        "retained_rss_delta_mb": rss_after_mb - baseline_rss_mb,
        "csr_nbytes_mb": _csr_nbytes_mb(result),
        "output_signature": _output_signature(result, vectorizer),
    }


def _run_profile_worker(TfidfVectorizer, X, version: str) -> dict:
    if version != "rust":
        raise RuntimeError("Stage profiling is only supported for the Rust backend")

    vectorizer = _new_vectorizer(TfidfVectorizer, version)
    import stratum

    stratum.set_config(debug_timing=True)
    for stream in (sys.stderr, sys.stdout):
        print(_STAGE_BEGIN, file=stream, flush=True)
    try:
        result = vectorizer.fit_transform(X)
    finally:
        for stream in (sys.stderr, sys.stdout):
            print(_STAGE_END, file=stream, flush=True)
        stratum.set_config(debug_timing=False)

    _assert_expected_backend(vectorizer, version)
    return {
        "csr_nbytes_mb": _csr_nbytes_mb(result),
        "output_signature": _output_signature(result, vectorizer),
    }


def run_worker(
    n_jobs: int,
    n_unique_words: int,
    dataset_length: int,
    version: str,
    worker_mode: str,
    timing_repeats: int,
) -> None:
    _set_worker_environment(n_jobs)

    import stratum
    from stratum import _rust_backend as rust_backend

    if version == "rust":
        if not rust_backend.HAVE_RUST:
            raise RuntimeError(
                "The Rust extension is unavailable. Build it with "
                "`maturin develop --release` before benchmarking."
            )
        stratum.set_config(
            rust_backend=True,
            allow_patch=True,
            num_threads=n_jobs,
            debug_timing=False,
        )
    else:
        stratum.set_config(
            rust_backend=False,
            allow_patch=True,
            num_threads=n_jobs,
            debug_timing=False,
        )

    # Always go through Stratum's patched adapter; the rust_backend flag
    # selects Rust vs sklearn fallback inside the same class.
    TfidfVectorizer = stratum.TfidfVectorizer

    X = get_synthetic_data(
        n_unique_words=n_unique_words,
        n_sentences=dataset_length,
    )
    _warm_up(TfidfVectorizer, X, version)

    if worker_mode == "timing":
        payload = _run_timing_worker(
            TfidfVectorizer, X, version, timing_repeats
        )
    elif worker_mode == "memory":
        payload = _run_memory_worker(TfidfVectorizer, X, version)
    elif worker_mode == "profile":
        payload = _run_profile_worker(TfidfVectorizer, X, version)
    else:
        raise RuntimeError(f"Unknown worker mode: {worker_mode!r}")

    print(f"RESULT:{json.dumps(payload, separators=(',', ':'))}")


def _run_worker_process(
    n_jobs: int,
    n_unique_words: int,
    dataset_length: int,
    version: str,
    worker_mode: str,
    timing_repeats: int,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n-jobs",
        str(n_jobs),
        "--n-unique-words",
        str(n_unique_words),
        "--dataset-length",
        str(dataset_length),
        "--version",
        version,
        "--worker-mode",
        worker_mode,
        "--timing-repeats",
        str(timing_repeats),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_worker_environment(n_jobs),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{worker_mode} worker failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            payload = json.loads(line.removeprefix("RESULT:"))
            break
    if payload is None:
        raise RuntimeError(f"Worker produced no RESULT line:\n{proc.stdout}")

    if worker_mode == "profile":
        # Rust print_timing → stderr; Python rb.print_timing → stdout.
        stages = parse_stage_timings(proc.stderr, proc.stdout)
        if not stages:
            raise RuntimeError(
                "Profile worker produced no stage timings.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        payload["stages_ms"] = stages
    return payload


def _run_case(
    n_jobs: int,
    n_unique_words: int,
    dataset_length: int,
    version: str,
    profile_stages: bool,
    timing_repeats: int,
) -> dict:
    timing = _run_worker_process(
        n_jobs,
        n_unique_words,
        dataset_length,
        version,
        "timing",
        timing_repeats,
    )
    memory = _run_worker_process(
        n_jobs,
        n_unique_words,
        dataset_length,
        version,
        "memory",
        timing_repeats,
    )
    _assert_signatures_match(
        memory["output_signature"],
        timing["output_signature"],
        context=f"{version} timing versus memory workers",
    )
    if not math.isclose(
        memory["csr_nbytes_mb"],
        timing["csr_nbytes_mb"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Timing and memory workers produced different CSR sizes")

    metrics = {
        **timing,
        "baseline_rss_mb": memory["baseline_rss_mb"],
        "call_peak_rss_mb": memory["call_peak_rss_mb"],
        "call_peak_increment_mb": memory["call_peak_increment_mb"],
        "retained_rss_delta_mb": memory["retained_rss_delta_mb"],
        "stages_ms": {},
    }
    if profile_stages and version == "rust":
        profile = _run_worker_process(
            n_jobs,
            n_unique_words,
            dataset_length,
            version,
            "profile",
            timing_repeats,
        )
        _assert_signatures_match(
            profile["output_signature"],
            timing["output_signature"],
            context="Rust timing versus profile workers",
        )
        metrics["stages_ms"] = profile["stages_ms"]
    return metrics


def _is_finite_number(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_has_measurements(row: dict) -> bool:
    numeric_fields = (
        "time_ms",
        "baseline_rss_mb",
        "call_peak_rss_mb",
        "call_peak_increment_mb",
        "retained_rss_delta_mb",
        "csr_nbytes_mb",
        "fitted_vocabulary_size",
    )
    if not all(_is_finite_number(row.get(field)) for field in numeric_fields):
        return False

    samples_raw = row.get("time_samples_ms_json")
    signature_raw = row.get("output_signature_json")
    try:
        samples = json.loads(samples_raw) if isinstance(samples_raw, str) else []
        signature = (
            json.loads(signature_raw) if isinstance(signature_raw, str) else {}
        )
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        bool(samples)
        and all(_is_finite_number(sample) for sample in samples)
        and bool(signature)
    )


def _row_has_stages(row: dict, version: str) -> bool:
    if version != "rust":
        return True
    raw = row.get("stages_ms_json", "{}")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return False
    try:
        stages = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(stages)


def run_master(
    tag: str,
    profile_stages: bool,
    force: bool,
    timing_repeats: int,
    *,
    cases: list[tuple[int, int]],
    n_jobs_vals: list[int],
) -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _csv_path(tag)

    results: list[dict] = []
    if csv_path.exists() and not force:
        try:
            loaded = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read existing results at {csv_path}"
            ) from exc

        results = loaded.to_dict("records")
        print(f"Loaded {len(results)} benchmark results from {csv_path}.")
    elif force and csv_path.exists():
        print(f"--force: ignoring existing {csv_path}")
    else:
        print(f"Writing results to {csv_path}")

    def signature_from_row(row: dict) -> dict:
        raw = row.get("output_signature_json", "{}")
        try:
            return json.loads(raw) if isinstance(raw, str) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    reference_signatures: dict[tuple[int, int], dict] = {}
    for row in results:
        if row.get("version") != "sklearn" or not _row_has_measurements(row):
            continue
        signature = signature_from_row(row)
        if signature:
            reference_signatures[
                (int(row["dataset_length"]), int(row["n_unique_words"]))
            ] = signature

    def already_done(length: int, words: int, n_jobs: int, version: str) -> bool:
        for r in results:
            if not (
                r["dataset_length"] == length
                and r["n_unique_words"] == words
                and r["n_jobs"] == n_jobs
                and r["version"] == version
                and r.get("timing_repeats") == timing_repeats
            ):
                continue
            if not _row_has_measurements(r):
                continue
            if profile_stages and not _row_has_stages(r, version):
                continue
            if version == "rust":
                reference = reference_signatures.get((length, words))
                if not reference:
                    continue
                try:
                    _assert_signatures_match(
                        signature_from_row(r),
                        reference,
                        context="cached Rust versus sklearn rows",
                    )
                except RuntimeError as exc:
                    print(f"Ignoring invalid cached row: {exc}")
                    continue
            return True
        return False

    def record(
        length: int,
        words: int,
        n_jobs: int,
        version: str,
        metrics: dict,
    ) -> None:
        # Drop incomplete prior rows for this key so re-runs replace them.
        nonlocal results
        results = [
            r
            for r in results
            if not (
                r["dataset_length"] == length
                and r["n_unique_words"] == words
                and r["n_jobs"] == n_jobs
                and r["version"] == version
            )
        ]
        stages = metrics.get("stages_ms") or {}
        signature = metrics["output_signature"]
        reference_key = (length, words)
        if version == "sklearn":
            reference_signatures[reference_key] = signature
        else:
            reference = reference_signatures.get(reference_key)
            if reference is None:
                raise RuntimeError(
                    "Cannot validate Rust output because the sklearn reference failed"
                )
            _assert_signatures_match(
                signature,
                reference,
                context=f"Rust n_jobs={n_jobs} versus sklearn",
            )

        row = {
            "timing_repeats": timing_repeats,
            "input_type": "pandas_series",
            "corpus_seed": _CORPUS_SEED,
            "n_jobs": n_jobs,
            "n_unique_words": words,
            "dataset_length": length,
            "version": version,
            "time_ms": metrics["time_ms"],
            "time_samples_ms_json": json.dumps(metrics["time_samples_ms"]),
            "baseline_rss_mb": metrics["baseline_rss_mb"],
            "call_peak_rss_mb": metrics["call_peak_rss_mb"],
            "call_peak_increment_mb": metrics["call_peak_increment_mb"],
            "retained_rss_delta_mb": metrics["retained_rss_delta_mb"],
            "csr_nbytes_mb": metrics["csr_nbytes_mb"],
            "fitted_vocabulary_size": signature["vocabulary_size"],
            "output_signature_json": json.dumps(signature, sort_keys=True),
            "stages_ms_json": json.dumps(stages, sort_keys=True),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        stage_note = f"  stages=[{_format_stages(stages)}]" if stages else ""
        samples = metrics["time_samples_ms"]
        print(
            f"Median: {metrics['time_ms']:.3f} ms "
            f"(range {min(samples):.3f}–{max(samples):.3f})  "
            f"call_peak={metrics['call_peak_rss_mb']:.1f} MiB  "
            f"peak_increment={metrics['call_peak_increment_mb']:.1f} MiB  "
            f"retained_delta={metrics['retained_rss_delta_mb']:.1f} MiB  "
            f"csr={metrics['csr_nbytes_mb']:.1f} MiB  "
            f"vocab={signature['vocabulary_size']}"
            f"{stage_note}"
        )

    # Both legs run through Stratum; only rust_backend / num_threads differ.
    # Sklearn ignores thread count — run it once per case at n_jobs=1.
    failures: list[str] = []
    for length, words in cases:
        for n_jobs in n_jobs_vals:
            for version in ("sklearn", "rust"):
                if version == "sklearn" and n_jobs != 1:
                    continue
                if already_done(length, words, n_jobs, version):
                    print(
                        f"Skipping {version} for length={length}, words={words}, "
                        f"n_jobs={n_jobs} (already exists)"
                    )
                    continue
                print(
                    f"Running {version} for length={length}, words={words}, "
                    f"n_jobs={n_jobs}..."
                )
                try:
                    metrics = _run_case(
                        n_jobs,
                        words,
                        length,
                        version,
                        profile_stages,
                        timing_repeats,
                    )
                    record(length, words, n_jobs, version, metrics)
                except RuntimeError as exc:
                    failure = (
                        f"{version} length={length}, words={words}, "
                        f"n_jobs={n_jobs}: {exc}"
                    )
                    failures.append(failure)
                    print(f"FAILED: {failure}")
                    continue

    if failures:
        detail = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(
            f"Benchmark incomplete; {len(failures)} case(s) failed:\n{detail}"
        )
    print(f"All benchmarks finished! Results in {csv_path}")
    print("Running plot generation...")
    generate_plots(tag)


def generate_plots(tag: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    csv_path = _csv_path(tag)
    if not csv_path.exists():
        print(f"No results at {csv_path}; skipping plots.")
        return

    df_all = pd.read_csv(csv_path)
    if df_all.empty:
        print("No benchmark results available; skipping plots.")
        return
    df = df_all[df_all["version"] == "rust"]
    # Baseline: stratum with rust_backend=False at matching n_jobs=1.
    df_sklearn = df_all[
        (df_all["version"] == "sklearn") & (df_all["n_jobs"] == 1)
    ]

    unit = "ms"
    n_jobs_vals = sorted(df["n_jobs"].unique())
    word_vals = sorted(df["n_unique_words"].unique())
    length_vals = sorted(df["dataset_length"].unique())

    metrics = (
        ("time_ms", f"time ({unit})", "latency"),
        ("call_peak_rss_mb", "call peak RSS (MiB)", "call peak RSS"),
        (
            "call_peak_increment_mb",
            "peak RSS over baseline (MiB)",
            "peak RSS increment",
        ),
        (
            "retained_rss_delta_mb",
            "retained RSS delta (MiB)",
            "retained RSS delta",
        ),
        ("csr_nbytes_mb", "CSR nbytes (MiB)", "CSR nbytes"),
    )

    fig, axes = plt.subplots(
        len(length_vals),
        len(metrics),
        figsize=(4.2 * len(metrics), 3.4 * len(length_vals)),
        squeeze=False,
    )

    njobs_colors = {
        1: "#C4C4C4",
        2: "#525252",
        4: "#f47e7e",
        8: "#620000",
        24: "#c1272d",
    }
    sklearn_color = "#444444"

    n_bars = len(n_jobs_vals) + 1
    total_width = 0.7
    bar_width = total_width / n_bars
    offsets = np.linspace(
        -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
    )

    for i, length in enumerate(length_vals):
        for col, (metric, ylabel, title_metric) in enumerate(metrics):
            if metric not in df_all.columns:
                axes[i][col].set_visible(False)
                continue
            ax = axes[i][col]
            sub = df[df["dataset_length"] == length]
            sub_sklearn = df_sklearn[df_sklearn["dataset_length"] == length]
            x = np.arange(len(word_vals))

            sklearn_vals = []
            for w in word_vals:
                row = sub_sklearn[sub_sklearn.n_unique_words == w]
                sklearn_vals.append(
                    row[metric].iloc[0] if not row.empty else float("nan")
                )
            ax.bar(x + offsets[0], sklearn_vals, bar_width, color=sklearn_color)

            for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
                vals = []
                for w in word_vals:
                    row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
                    vals.append(row[metric].iloc[0] if not row.empty else float("nan"))
                ax.bar(
                    x + offset, vals, bar_width, color=njobs_colors.get(n_jobs, "#888")
                )

                if metric == "time_ms":
                    _annotate_bar_speedups(
                        ax, x + offset, vals, sklearn_vals
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(word_vals)
            ax.set_title(f"n_rows = {length:,} ({title_metric})")
            ax.set_xlabel("n_unique_words")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", ls=":", alpha=0.5)

    legend_handles = [
        Patch(color=sklearn_color, label="stratum (rust_backend=False)")
    ]
    legend_handles += [
        Patch(color=njobs_colors.get(j, "#888"), label=str(j)) for j in n_jobs_vals
    ]

    fig.legend(
        handles=legend_handles,
        title="rust_backend=True (n_jobs)",
        ncol=len(n_jobs_vals) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    slug = _tag_slug(tag)
    out = RESULTS_DIR / f"macrobenchmark_njobs_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")

    _generate_speedup_plot(df_all, tag)
    _generate_stage_plot(df_all, tag)


def _generate_speedup_plot(df_all, tag: str) -> None:
    """Standalone latency plot: time (ms) with sklearn baseline + rust n_jobs."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    rust = df_all[df_all["version"] == "rust"]
    sklearn1 = df_all[
        (df_all["version"] == "sklearn") & (df_all["n_jobs"] == 1)
    ]
    if rust.empty or sklearn1.empty:
        print("Missing rust/sklearn rows; skipping time plot.")
        return

    n_jobs_vals = sorted(rust["n_jobs"].unique())
    word_vals = sorted(rust["n_unique_words"].unique())
    length_vals = sorted(rust["dataset_length"].unique())

    njobs_colors = {
        1: "#C4C4C4",
        2: "#525252",
        4: "#f47e7e",
        8: "#620000",
        24: "#c1272d",
    }
    sklearn_color = "#444444"

    fig, axes = plt.subplots(
        len(length_vals),
        1,
        figsize=(8.5, 3.4 * len(length_vals)),
        squeeze=False,
    )

    n_bars = len(n_jobs_vals) + 1
    total_width = 0.7
    bar_width = total_width / n_bars
    offsets = np.linspace(
        -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
    )

    for i, length in enumerate(length_vals):
        ax = axes[i][0]
        sub = rust[rust["dataset_length"] == length]
        sub_sklearn = sklearn1[sklearn1["dataset_length"] == length]
        x = np.arange(len(word_vals))

        sklearn_vals = []
        for w in word_vals:
            row = sub_sklearn[sub_sklearn.n_unique_words == w]
            sklearn_vals.append(
                float(row["time_ms"].iloc[0]) if not row.empty else float("nan")
            )
        ax.bar(x + offsets[0], sklearn_vals, bar_width, color=sklearn_color)

        for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
            vals = []
            for w in word_vals:
                row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
                vals.append(
                    float(row["time_ms"].iloc[0]) if not row.empty else float("nan")
                )
            ax.bar(
                x + offset, vals, bar_width, color=njobs_colors.get(n_jobs, "#888")
            )
            _annotate_bar_speedups(ax, x + offset, vals, sklearn_vals)

        ax.set_xticks(x)
        ax.set_xticklabels(word_vals)
        ax.set_xlabel("n_unique_words")
        ax.set_ylabel("time (ms)")
        ax.set_title(f"n_rows = {length:,} (latency)")
        ax.grid(True, axis="y", ls=":", alpha=0.5)

    legend_handles = [
        Patch(color=sklearn_color, label="stratum (rust_backend=False)")
    ]
    legend_handles += [
        Patch(color=njobs_colors.get(j, "#888"), label=str(j)) for j in n_jobs_vals
    ]
    fig.legend(
        handles=legend_handles,
        title="rust_backend=True (n_jobs)",
        ncol=len(n_jobs_vals) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    slug = _tag_slug(tag)
    out = RESULTS_DIR / f"macrobenchmark_speedup_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved time plot to {out}")


def _generate_stage_plot(df_all, tag: str) -> None:
    """Stacked bars of rust stage timings when ``stages_ms_json`` is populated."""
    import matplotlib.pyplot as plt
    import numpy as np

    if "stages_ms_json" not in df_all.columns:
        return

    rust = df_all[df_all["version"] == "rust"].copy()
    parsed = []
    for _, row in rust.iterrows():
        raw = row.get("stages_ms_json", "{}")
        try:
            stages = json.loads(raw) if isinstance(raw, str) else {}
        except (TypeError, json.JSONDecodeError):
            stages = {}
        if not stages:
            continue
        parsed.append((row, stages))
    if not parsed:
        print("No stage timings found; skipping stage plot.")
        return

    stage_names: list[str] = []
    for s in _STAGE_ORDER:
        if any(s in stages for _, stages in parsed):
            stage_names.append(s)
    extras = sorted(
        {
            name
            for _, stages in parsed
            for name in stages
            if name not in stage_names
        }
    )
    stage_names.extend(extras)

    length_vals = sorted({int(row["dataset_length"]) for row, _ in parsed})
    word_vals = sorted({int(row["n_unique_words"]) for row, _ in parsed})
    n_jobs_vals = sorted({int(row["n_jobs"]) for row, _ in parsed})

    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(stage_names)}

    fig, axes = plt.subplots(
        len(length_vals),
        len(word_vals),
        figsize=(4.5 * len(word_vals), 3.2 * len(length_vals)),
        squeeze=False,
        sharey="row",
    )

    for i, length in enumerate(length_vals):
        for j, words in enumerate(word_vals):
            ax = axes[i][j]
            x = np.arange(len(n_jobs_vals))
            bottoms = np.zeros(len(n_jobs_vals))
            for stage in stage_names:
                vals = []
                for n_jobs in n_jobs_vals:
                    ms = 0.0
                    for row, stages in parsed:
                        if (
                            int(row["dataset_length"]) == length
                            and int(row["n_unique_words"]) == words
                            and int(row["n_jobs"]) == n_jobs
                        ):
                            ms = float(stages.get(stage, 0.0))
                            break
                    vals.append(ms)
                ax.bar(
                    x,
                    vals,
                    bottom=bottoms,
                    color=colors[stage],
                    label=stage if i == 0 and j == 0 else None,
                    width=0.65,
                )
                bottoms = bottoms + np.asarray(vals, dtype=float)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in n_jobs_vals])
            ax.set_xlabel("n_jobs")
            if j == 0:
                ax.set_ylabel("stage time (ms)")
            ax.set_title(f"n_rows={length:,}, words={words}")
            ax.grid(True, axis="y", ls=":", alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="stage",
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    slug = _tag_slug(tag)
    out = RESULTS_DIR / f"macrobenchmark_stages_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stage plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-unique-words", type=int, default=1000)
    parser.add_argument("--dataset-length", type=int, default=10000)
    parser.add_argument("--version", type=str, choices=["rust", "sklearn"])
    parser.add_argument(
        "--worker-mode",
        choices=["timing", "memory", "profile"],
        default="timing",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help="Fresh-estimator timing repetitions per case (default: %(default)s).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="tfidf_rust_backend",
        help="Label for this run; results are written to "
        "results/macrobenchmark_<tag>.csv and matching PNG plots.",
    )
    parser.add_argument(
        "--profile-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a separate Rust debug-timing worker and harvest stage lines "
        "(default: on). Use --no-profile-stages to skip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing CSV rows and re-run the selected grid.",
    )
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")

    if args.worker:
        if args.version is None:
            parser.error("--version is required with --worker")
        run_worker(
            args.n_jobs,
            args.n_unique_words,
            args.dataset_length,
            args.version,
            args.worker_mode,
            args.timing_repeats,
        )
    else:
        cases, n_jobs_vals = benchmark_grid()
        case_str = ", ".join(f"{length:,}/{words:,}" for length, words in cases)
        print(
            f"Benchmark grid (full): cases=[{case_str}], "
            f"n_jobs={n_jobs_vals}, repeats={args.timing_repeats}"
        )
        run_master(
            args.tag,
            args.profile_stages,
            args.force,
            args.timing_repeats,
            cases=cases,
            n_jobs_vals=n_jobs_vals,
        )
