from sklearn.base import BaseEstimator
from skrub._data_ops import DataOp
from skrub._data_ops._choosing import BaseChoice, Choice
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
            return (id(impl1.container) == id(impl2.container) and
                    (isinstance(impl1.key, str) and impl1.key == impl2.key or
                     _stable_id(impl1.key) == _stable_id(impl2.key)))
        if isinstance(impl1, GetAttr):
            # op1 = data.attribute1, op2 = data.attribute1
            return id(impl1.source_object) == id(impl2.source_object) and impl1.attr_name == impl2.attr_name
        elif isinstance(impl1, Call):
            # op1 = col.skb.apply_func(my_func, arg1, arg2) , op2 = col.skb.apply_func(my_func, arg1, arg2)
            if impl1.func == impl2.func and len(impl1.args) == len(impl2.args):
                inputs_ids1 = _stable_id(impl1.args)
                inputs_ids2 = _stable_id(impl2.args)
                return inputs_ids1 == inputs_ids2
        elif isinstance(impl1, CallMethod):
            # op1 = col.apply(my_func, arg1, arg2) , op2 = col.apply(my_func, arg1, arg2)
            if id(impl1.obj) == id(impl2.obj) and impl1.method_name == impl2.method_name:
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
            if id(impl1.X) == id(impl2.X) and type(est1) == type(est2) :
                # Check if columns are the same:
                if isinstance(impl1.cols, All) and isinstance(impl2.cols, All) or set(impl1.cols) == set(impl2.cols):
                    return estimator_equality_check(est1, est2)
        elif isinstance(impl1, BinOp):
            # op1 = col1 / col2
            # op2 = col1 / col2
            if impl1.op == impl2.op:
                return _stable_id(impl1.left) == _stable_id(impl2.left) and _stable_id(impl1.right) == _stable_id(impl2.right)

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
    for key, value in est.get_params().items():
        if isinstance(value, BaseEstimator):
            return hash_estimator(value)
        else:
            return hash((key, _stable_id(value)))


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
    found = False

    if isinstance(impl, GetItem):
        if id(impl.container) == id(old_input):
            impl.container = new_input
            found = True

    elif isinstance(impl, GetAttr):
        if id(impl.source_object) == id(old_input):
            impl.source_object = new_input
            found = True

    elif isinstance(impl, Call):
        args = impl.args
        if isinstance(args, list):
            for i, arg in enumerate(args):
                if id(arg) == id(old_input):
                    args[i] = new_input
                    found = True
        elif isinstance(args, tuple):
            impl.args = tuple(new_input if id(a) == id(old_input) else a for a in args)
            if any(id(a) == id(old_input) for a in args):
                found = True

    elif isinstance(impl, CallMethod):
        # Object
        if id(impl.obj) == id(old_input):
            impl.obj = new_input
            found = True

        # Args (list or tuple)
        args = impl.args
        if isinstance(args, list):
            for i, arg in enumerate(args):
                if id(arg) == id(old_input):
                    args[i] = new_input
                    found = True
        elif isinstance(args, tuple):
            impl.args = tuple(new_input if id(a) == id(old_input) else a for a in args)
            if any(id(a) == id(old_input) for a in args):
                found = True

        # Kwargs (dict)
        kwargs = impl.kwargs
        if isinstance(kwargs, dict):
            for k, v in kwargs.items():
                if id(v) == id(old_input):
                    kwargs[k] = new_input
                    found = True
        else:
            # Immutable or proxy object — rebuild
            new_kwargs = {
                k: (new_input if id(v) == id(old_input) else v)
                for k, v in kwargs.items()
            }
            if any(id(v) == id(old_input) for v in kwargs.values()):
                found = True
            impl.kwargs = new_kwargs

    elif isinstance(impl, Apply):
        if id(impl.X) == id(old_input):
            impl.X = new_input
            found = True

        if id(impl.y) == id(old_input):
            impl.y = new_input
            found = True

        # Columns (can be list or symbolic)
        if isinstance(impl.cols, list):
            for i, c in enumerate(impl.cols):
                if id(c) == id(old_input):
                    impl.cols[i] = new_input
                    found = True
        elif isinstance(impl.cols, tuple):
            impl.cols = tuple(new_input if id(c) == id(old_input) else c for c in impl.cols)
            if any(id(c) == id(old_input) for c in impl.cols):
                found = True
    elif isinstance(impl, Value):
        if isinstance(impl.value, BaseChoice):
            if isinstance(impl.value, Choice):
                outcomes = impl.value.outcomes
                for i, outcome in enumerate(outcomes):
                    if id(outcome) == id(old_input):
                        outcomes[i] = new_input
                        found = True
    elif isinstance(impl, BinOp):
        if id(impl.left) == id(old_input):
            impl.left = new_input
            found = True
        elif id(impl.right) == id(old_input):
            impl.right = new_input
            found = True
    if not found:
        raise Exception(f"Could not find old DataOp {old_input} during input update for {op}")