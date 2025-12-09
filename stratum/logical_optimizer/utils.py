from typing import Iterable
from sklearn.base import BaseEstimator
from skrub._data_ops import DataOp
from skrub._data_ops._choosing import Choice
from skrub._data_ops._data_ops import Call, GetItem, CallMethod, GetAttr, Apply, Value, BinOp
from skrub.selectors._base import All


def equals_data_op(op1: DataOp, op2: DataOp):
    """
    Check whether two Skrub DataOp nodes are functionally equivalent.

    This function compares two DataOp instances (`op1` and `op2`) by inspecting
    their underlying `_skrub_impl` operators. It determines whether they represent
    semantically identical computation steps in the DataOps graph — i.e., whether
    they produce the same result given the same inputs.

    Parameters
    ----------
    op1 : DataOp
        First Skrub DataOp node to compare.
    op2 : DataOp
        Second Skrub DataOp node to compare.

    Returns
    -------
    bool
        True if both DataOps are equivalent in structure and semantics,
        False otherwise.
    """
    impl1 = op1._skrub_impl
    impl2 = op2._skrub_impl
    if type(impl1) == type(impl2):
        if isinstance(impl1, GetItem):
            # op1 = data["col1"], op2 = data["col1"]
            return (impl1.container is impl2.container and
                    (isinstance(impl1.key, str) and impl1.key == impl2.key or
                     _stable_id(impl1.key) == _stable_id(impl2.key)))
        if isinstance(impl1, GetAttr):
            # op1 = data.attribute1, op2 = data.attribute1
            return impl1.source_object is impl2.source_object and impl1.attr_name == impl2.attr_name
        elif isinstance(impl1, Call):
            # op1 = col.skb.apply_func(my_func, arg1, arg2) , op2 = col.skb.apply_func(my_func, arg1, arg2)
            if impl1.func == impl2.func and len(impl1.args) == len(impl2.args):
                inputs_ids1 = _stable_id(impl1.args)
                inputs_ids2 = _stable_id(impl2.args)
                return inputs_ids1 == inputs_ids2
        elif isinstance(impl1, CallMethod):
            # op1 = col.apply(my_func, arg1, arg2) , op2 = col.apply(my_func, arg1, arg2)
            if impl1.obj is impl2.obj and impl1.method_name == impl2.method_name:
                inputs_ids1 = _stable_id(impl1.args)
                inputs_ids2 = _stable_id(impl2.args)
                named_inputs_ids1 = _stable_id(impl1.kwargs)
                named_inputs_ids2 = _stable_id(impl2.kwargs)
                return inputs_ids1 == inputs_ids2 and named_inputs_ids1 == named_inputs_ids2
        elif isinstance(impl1, Apply):
            # enc1 = StandardScaler(arg1)
            # enc2 = StandardScaler(arg1)
            # op1 = data.skb.apply(enc1), op2 = data.skb.apply(enc2)
            est1 = impl1.estimator
            est2 = impl2.estimator
            if impl1.X is impl2.X and type(est1) == type(est2):
                # Check if columns are the same:
                cols1 = {"all"} if isinstance(impl1.cols, All) else set(impl1.cols)
                cols2 = {"all"} if isinstance(impl2.cols, All) else set(impl2.cols)
                # TODO also match All with set(cols) if cols contains all columns of the input frame
                if set(cols1) == set(cols2):
                    return estimator_equality_check(est1, est2)
        elif isinstance(impl1, BinOp):
            # op1 = col1 / col2
            # op2 = col1 / col2
            if impl1.op == impl2.op:
                return _stable_id(impl1.left) == _stable_id(impl2.left) and _stable_id(impl1.right) == _stable_id(
                    impl2.right)

    return False


def estimator_equality_check(est1: DataOp, est2: DataOp) -> bool:
    """"
    Check if two estimators are semantically equal.
    """
    params1 = est1.get_params()
    params2 = est2.get_params()
    for key, value in params1.items():
        value2 = params2.get(key)
        if value2 != value and (
                type(value) != type(value2)
                or not isinstance(value, BaseEstimator)
                or not estimator_equality_check(value, value2)):
            return False
    return True


def hash_data_op(op: DataOp) -> int:
    """
    Compute a hash value for a Skrub DataOp node, consistent with equals_data_op().

    This function produces a stable, structure-aware hash used for caching and
    deduplication of computation graph nodes. Two DataOps that are equal
    according to `equals_data_op` will always produce the same hash value.

    Parameters
    ----------
    op : DataOp
        The Skrub DataOp instance to hash.

    Returns
    -------
    int
        A hash value uniquely identifying this DataOp's structure and semantics.
    """
    impl = op._skrub_impl
    t = type(impl)

    if isinstance(impl, GetItem):
        return hash((t, id(impl.container), _stable_id(impl.key)))

    elif isinstance(impl, GetAttr):
        # op = data.attribute1
        return hash((t, id(impl.source_object), impl.attr_name))

    elif isinstance(impl, Call):
        # op = col.skb.apply_func(my_func, arg1, arg2)
        arg_ids = frozenset(id(arg) for arg in impl.args)
        return hash((t, impl.func, arg_ids))

    elif isinstance(impl, CallMethod):
        # op = col.apply(my_func, arg1, arg2)
        arg_ids = frozenset(_stable_id(arg) for arg in impl.args)
        kwarg_ids = frozenset(id(kwarg) for kwarg in impl.kwargs.values())
        return hash((t, id(impl.obj), impl.method_name, arg_ids, kwarg_ids))

    elif isinstance(impl, Apply):
        # op = data.skb.apply(estimator)
        est = impl.estimator
        if isinstance(impl.cols, All):
            # All columns -> only estimator type + param structure
            est_type = type(est)
            est_params = hash_estimator(est)
            return hash((t, id(impl.X), est_type, est_params))
        else:
            # Specific columns
            col_ids = frozenset(id(c) for c in impl.cols)
            est_type = type(est)
            est_params = frozenset(est.get_params().items())
            return hash((t, id(impl.X), col_ids, est_type, est_params))
    elif isinstance(impl, BinOp):
        return hash((t, impl.op, _stable_id(impl.left), _stable_id(impl.right)))

    else:
        # Fallback for unknown DataOp types
        return hash((t, id(impl)))


def hash_estimator(est: BaseEstimator) -> int:
    """
    Hash an estimator.
    """
    param_hashes = []
    for key, value in est.get_params().items():
        if isinstance(value, BaseEstimator):
            param_hashes.append((key, hash_estimator(value))) 
        else:
            param_hashes.append(((key, _stable_id(value))))
    return hash(tuple(param_hashes))


def _stable_id(obj):
    """
    Returns a deterministic, structure-aware hashable surrogate for id(obj),
    such that lists/sets/tuples with the same unordered contents produce
    the same hash value, independent of their identity.
    """
    if isinstance(obj, (list, set, tuple)):
        # unordered, element-wise stable ids
        return frozenset(_stable_id(x) for x in obj)
    elif isinstance(obj, dict):
        return frozenset((k, _stable_id(v)) for k, v in obj.items())
    elif hasattr(obj, "__hash__") and not isinstance(obj, DataOp):
        # hashable primitive or object
        return hash(obj)
    else:
        # fallback to identity for unhashable/unrecognized
        return id(obj)


def update_data_op(op: DataOp, old_input: DataOp, new_input: DataOp):
    """
    Update a DataOp node by replacing references to an old subexpression
    (`old_input`) with a new one (`new_input`).

    Performs in-place updates when possible to minimize object creation.
    Raises if `old_input` is not found among the op's dependencies.
    """
    impl = op._skrub_impl

    if isinstance(impl, GetItem):
        if impl.container is old_input:
            impl.container = new_input
            return

    elif isinstance(impl, GetAttr):
        if impl.source_object is old_input:
            impl.source_object = new_input
            return

    elif isinstance(impl, Call):
        args = impl.args
        found, args = replace_data_op_in_iterable(args, new_input, old_input)
        if found:
            impl.args = args
            return

    elif isinstance(impl, CallMethod):
        if impl.obj is old_input:
            impl.obj = new_input
            return

        args = impl.args
        found, args = replace_data_op_in_iterable(args, new_input, old_input)
        if found:
            impl.args = args
            return

        kwargs = impl.kwargs
        found, kwargs = replace_data_op_in_iterable(kwargs, new_input, old_input)
        if found:
            impl.kwargs = kwargs
            return

    elif isinstance(impl, Apply):
        if impl.X is old_input:
            impl.X = new_input
            return

        if impl.y is old_input:
            impl.y = new_input
            return

    elif isinstance(impl, Value):
        if isinstance(impl.value, Choice):
            outcomes = impl.value.outcomes
            for i, outcome in enumerate(outcomes):
                if outcome is old_input:
                    outcomes[i] = new_input
                    return
    elif isinstance(impl, BinOp):
        if impl.left is old_input:
            impl.left = new_input
            return
        elif impl.right is old_input:
            impl.right = new_input
            return
    raise Exception(f"Could not find old DataOp {old_input} during input update for {op}")



def replace_data_op_in_iterable(iterable: Iterable, new_input: DataOp,
                                old_input: DataOp) -> tuple[bool, Iterable]:
    """
    Helper Method to replace a DataOp node in an iterable with a new one.
    """
    found = False
    if isinstance(iterable, tuple):
        new_args = []

        for a in iterable:
            if a is old_input:
                new_args.append(new_input)
                found = True
            else:
                new_args.append(a)
        return found, tuple(new_args)
    elif isinstance(iterable, dict):
        for k, v in iterable.items():
            if v is old_input:
                iterable[k] = new_input
                found = True
        return found, iterable
    else:
        raise NotImplementedError("Non-tuple arguments of method call are not supported yet.")
