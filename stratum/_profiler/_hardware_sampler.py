"""Standalone sampler executable. Deliberately does not import the engine.

Only run-level startup/shutdown messages cross the process boundary. Sampling
owns no Manager process and never receives per-operator RPCs.
"""
from __future__ import annotations

from collections import deque
import json
import os
import sys
import threading
from time import perf_counter, process_time


def sample_tree(root, psutil, previous, excluded_pid):
    """Accumulate observed worker CPU deltas; never count the sampler."""
    processes = [root, *root.children(recursive=True)]
    engine_rss = tree_rss = 0
    current = {}
    delta = 0.0
    for proc in processes:
        if proc.pid == excluded_pid:
            continue
        try:
            with proc.oneshot():
                rss = proc.memory_info().rss
                cpu = proc.cpu_times()
                identity = (proc.pid, proc.create_time())
            seconds = cpu.user + cpu.system
            current[identity] = seconds
            delta += max(0.0, seconds - previous.get(identity, 0.0))
            tree_rss += rss
            if proc.pid == root.pid:
                engine_rss = rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    previous.clear()
    previous.update(current)
    return engine_rss, tree_rss, delta


def main():
    # Optional dependency is imported only in this small, separate interpreter.
    import psutil

    pid, interval, capacity = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    root = psutil.Process(pid)
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.buffer.readline(), stop.set()), daemon=True).start()
    samples = deque(maxlen=capacity)
    previous = {}
    sampling_seconds = 0.0
    failed_samples = total_samples = 0
    peak_rss = rss_sum = engine_rss_sum = peak_engine_rss = 0
    first_time = last_time = last_cpu = first_cpu = None
    service_peak_rss = 0
    cpu = 0.0
    ready_sent = False
    myself = psutil.Process()
    psutil.cpu_percent(interval=None)  # Prime the host's CPU counters.

    while True:
        start = perf_counter()
        try:
            if not root.is_running() or root.status() == psutil.STATUS_ZOMBIE:
                break
            engine_rss, tree_rss, delta = sample_tree(root, psutil, previous, os.getpid())
            cpu += delta
            now = perf_counter()
            dt = now - last_time if last_time is not None else 0.0
            cpu_percent = max(0.0, (cpu - last_cpu) / dt * 100) if dt else None
            host_cpu = psutil.cpu_percent(interval=None) if dt else None
            host_memory = psutil.virtual_memory().used
            samples.append({
                "timestamp": now, "collection_start": start, "interval_seconds": dt,
                "engine_rss_bytes": engine_rss, "tree_rss_bytes": tree_rss,
                "tree_cpu_percent": cpu_percent, "host_cpu_percent": host_cpu,
                "host_memory_used_bytes": host_memory,
            })
            total_samples += 1
            rss_sum += tree_rss
            engine_rss_sum += engine_rss
            peak_rss = max(peak_rss, tree_rss)
            peak_engine_rss = max(peak_engine_rss, engine_rss)
            if first_time is None:
                first_time, first_cpu = now, cpu
            last_time, last_cpu = now, cpu
            service_peak_rss = max(service_peak_rss, myself.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            failed_samples += 1
            if not root.is_running():
                break
        finally:
            sampling_seconds += perf_counter() - start

        if not ready_sent and total_samples:
            print(json.dumps({"ready": True}), flush=True)
            ready_sent = True
        if stop.is_set() or stop.wait(max(0.0, interval - (perf_counter() - start))):
            break

    report = {
        "samples": list(samples), "sample_count": total_samples,
        "dropped_samples": max(0, total_samples - len(samples)),
        "failed_samples": failed_samples, "sampling_seconds": sampling_seconds,
        "service_cpu_seconds": process_time(),
        "service_peak_rss_bytes": service_peak_rss,
        "summary": {
            "tree_rss_mean_bytes": rss_sum / total_samples if total_samples else None,
            "tree_rss_peak_bytes": peak_rss if total_samples else None,
            "engine_rss_mean_bytes": engine_rss_sum / total_samples if total_samples else None,
            "engine_rss_peak_bytes": peak_engine_rss if total_samples else None,
            "tree_cpu_seconds_observed": max(0.0, last_cpu - first_cpu) if total_samples else 0.0,
            "sample_window_seconds": last_time - first_time if total_samples else 0.0,
        },
    }
    json.dump(report, sys.stdout, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
