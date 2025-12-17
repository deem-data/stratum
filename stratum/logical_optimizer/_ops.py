from __future__ import annotations

from sklearn import clone
from skrub._data_ops._data_ops import DataOp
from skrub._data_ops._choosing import Choice
from skrub._data_ops._data_ops import Value
from ._op_comparison import equals_skrub_impl
from pandas import DataFrame


class Op():
    def __init__(self, skrub_impl= None, children=None, parents=None, name=None, is_X=False, is_y=False):
        self.name = name
        self.skrub_impl = skrub_impl
        self.is_frame = False
        self.children = children
        self.parents = parents
        self.intermediate = None
        self.is_X = is_X
        self.is_y = is_y
        self.was_cloned = False

    def __str__(self):
        return f"Op(name={self.name})"
    
    def __repr__(self):
        return f"Op(name={self.name}, cloned={self.was_cloned}, id={id(self)})"

    def eq_shallow(self, other):
        """ 
        Check if two Op objects are shallowly equal, i.e. if their skrub_impls are equal, we do not check the children and parents.
        """
        return equals_skrub_impl(self.skrub_impl, other.skrub_impl)
    
    def clone(self, children: list[Op] = None, parents: list[Op] = None):
        if self.skrub_impl is not None:
            attributes = {}
            for att in self.skrub_impl._fields:
                if att == "estimator":
                    estm = self.skrub_impl.estimator
                    params = estm.get_params()
                    estm_new = clone(estm)
                    estm_new.set_params(**params)
                    attributes[att] = clone(estm_new)
                else:
                    attributes[att] = getattr(self.skrub_impl, att)
            new_impl = self.skrub_impl.__class__(**attributes)
        else:
            new_impl = None

        new_op = Op(new_impl, children=children, parents=parents)
        new_op.name = self.name
        new_op.is_frame = self.is_frame
        new_op.is_X = self.is_X
        new_op.is_y = self.is_y
        new_op.was_cloned = True
        return new_op

    def update_name(self):
        pass

    def has_children(self) -> bool:
        return self.children is not None and len(self.children) > 0

    def is_choice(self) -> bool:
        return isinstance(self, ChoiceOp)

    def add_child(self, child: Op):
        self.children.append(child)

    def add_parent(self, parent: Op):
        self.parents.append(parent)

class ChoiceOp(Op):
    def __init__(self, outcome_names: list[str] = None, n_outcomes: int = None, choice_name: str=None, append_choice_name = True, parents = None):
        if outcome_names is None:
            outcome_names = [[(choice_name, f"Opt{i}")] for i in range(n_outcomes)]
        elif append_choice_name:
            outcome_names = [[(choice_name, name)] for name in outcome_names]
        else:
            outcome_names = outcome_names
        super().__init__(skrub_impl=None, parents=parents)
        self.outcome_names = outcome_names
        self.update_name()

    def make_outcome_names(self):
        # TODO find a better way for naming the unnamed choices
        return [", ".join(
                f"Choice{len(combi) - i - 1}:{value}" if choice_name is None else f"{choice_name}:{value}"
                for i, (choice_name, value) in enumerate(combi)
            ) for combi in self.outcome_names]

    def update_name(self):
        self.name = "  |  ".join(self.make_outcome_names())


class ValueOp(Op):
    def __init__(self, value, is_X=False, is_y=False):
        super().__init__()
        self.name = f"Value: {"DataFrame" if isinstance(value, DataFrame) else value}" 
        self.value = value
        self.is_X = is_X
        self.is_y = is_y


class SearchEvalOp(Op):
    def __init__(self, outcome_names: list[str], parent: Op = None):
        super().__init__(skrub_impl=None)
        self.name = "evaluate gridsearch" 
        self.outcome_names = outcome_names
        self.parents = [] if parent is None else [parent]
        self.children = []



def as_op(data_op: DataOp):
    impl = data_op._skrub_impl
    if isinstance(impl, Value):
        if isinstance(impl.value, Choice):
            choice = impl.value
            parents = [0]*len(choice.outcomes)
            for i, outcome in enumerate(choice.outcomes):
                if not isinstance(outcome, DataOp):
                    parents[i] = ValueOp(outcome)
            return ChoiceOp(choice.outcome_names, len(choice.outcomes), choice.name, parents=parents)
        else:
            return ValueOp(impl.value, is_X=impl.is_X, is_y=impl.is_y)
    else:
        return Op(
            skrub_impl=impl, 
            is_X=impl.is_X, 
            is_y=impl.is_y, 
            name=data_op.__skrub_short_repr__())
