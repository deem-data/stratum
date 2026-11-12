"""Resident state ownership and concurrent accounting."""
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Thread
import weakref

import pytest

from stratum.runtime._state_store import StateStore


class Payload:
    pass


def test_generic_values_and_explicit_ownership():
    store = StateStore()
    key, value = object(), Payload()
    ref = weakref.ref(value)
    store.put(key, value, size=40)
    assert key in store
    del value
    borrowed = store.get(key)
    assert ref() is borrowed
    assert store.remove(key)
    assert key not in store
    assert store.memory_usage == 0
    assert ref() is borrowed
    del borrowed
    gc.collect()
    assert ref() is None
    assert not store.remove(key)
    with pytest.raises(KeyError):
        store.get(key)
    store.put((1, "none"), None, size=0)
    assert store.get((1, "none")) is None


def test_insertion_and_replacement_update_accounting():
    store = StateStore()
    old = Payload()
    store.put("a", old, size=60)
    store.put("b", None, size=40)
    assert store.snapshot() == {"entries": 2, "memory_usage": 100}
    assert store.get("a") is old
    replacement = Payload()
    store.put("a", replacement, size=160)
    assert store.get("a") is replacement
    assert store.snapshot() == {"entries": 2, "memory_usage": 200}
    store.put("a", None, size=10)
    store.put("c", None, size=50)
    assert store.snapshot() == {"entries": 3, "memory_usage": 100}
    store.clear()
    assert store.snapshot() == {"entries": 0, "memory_usage": 0}


@pytest.mark.parametrize("bad", [-1, 1.5, "10"])
def test_invalid_byte_counts_do_not_mutate_store(bad):
    store = StateStore()
    old = Payload()
    store.put("a", old, size=50)
    before = store.snapshot()
    with pytest.raises((TypeError, ValueError)):
        store.put("a", None, size=bad)
    assert store.snapshot() == before
    assert store.get("a") is old


def test_concurrent_insertions_preserve_accounting():
    store = StateStore()
    barrier = Barrier(12)

    def insert(k):
        barrier.wait(timeout=5)
        store.put(k, Payload(), size=10)

    with ThreadPoolExecutor(max_workers=12) as workers:
        list(workers.map(insert, range(12)))
    assert store.snapshot() == {"entries": 12, "memory_usage": 120}


def test_concurrent_replace_remove_clear_and_snapshot():
    store = StateStore()
    barrier = Barrier(6)

    def exercise(worker):
        barrier.wait(timeout=5)
        for i in range(300):
            key = i % 4
            if worker == 0:
                store.clear()
            elif worker == 1:
                store.remove(key)
            else:
                store.put(key, (worker, i), size=10)
                try:
                    assert len(store.get(key)) == 2
                except KeyError:  # another thread removed the entry
                    pass
            snapshot = store.snapshot()
            assert snapshot["memory_usage"] == 10 * snapshot["entries"]

    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(exercise, range(6)))
    store.clear()
    assert store.memory_usage == len(store) == 0


@pytest.mark.parametrize("operation", ["replace", "remove", "clear"])
def test_value_destructors_run_outside_lock(operation):
    store = StateStore()
    observed = []

    class InspectOnDelete:
        def __del__(self):
            observed.append(store.snapshot()["memory_usage"])

    store.put("a", InspectOnDelete(), size=10)

    def release():
        if operation == "replace":
            store.put("a", None, size=20)
        elif operation == "remove":
            store.remove("a")
        else:
            store.clear()
        gc.collect()

    thread = Thread(target=release, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "destructor deadlocked while inspecting the store"
    assert observed == [20 if operation == "replace" else 0]
