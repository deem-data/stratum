from ._ops import Op

class Rewrite():
    def rewrite(self, op: Op):
        raise NotImplementedError("Subclasses must implement this method")
