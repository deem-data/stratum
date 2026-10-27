"""Binding a recorded pandas call to pandas' own signature.

The pipeline is assumed to be written against pandas, so the optimizer binds
each call it records against the pandas function or method called, then parses
the bound arguments into an op's fields.
"""
from __future__ import annotations

import inspect


def bind_pandas_function(func, args, kwargs) -> dict | None:
    """``func(*args, **kwargs)``'s arguments by parameter name, or ``None`` when
    ``func`` would reject them. Only the arguments actually passed are returned."""
    try:
        bound = inspect.signature(func).bind(*(args or ()), **(kwargs or {}))
    except TypeError:
        return None
    return dict(bound.arguments)


def bind_pandas_method(owner, method_name: str, args, kwargs) -> dict | None:
    """The arguments of a call to ``owner``'s method ``method_name``, without the
    receiver, by parameter name; ``None`` when pandas would reject them (unknown
    keyword, too many positionals, ...). Pandas defaults stay implicit."""
    arguments = bind_pandas_function(getattr(owner, method_name), (None, *(args or ())), kwargs)
    if arguments is not None:
        arguments.pop("self")
    return arguments
