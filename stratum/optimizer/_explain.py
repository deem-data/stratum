from __future__ import annotations

from stratum.optimizer.logical._ops import Op


def explain_linear_plan(
    name: str,
    linearized_ops: list[Op],
    split_pos: int | None = None,
) -> None:
    """Print a human-readable linearized execution plan.

    For plans with a splitting/CV operation (split_pos is not None), this prints
    the plan structure divided into pre-split, split, fit, and predict phases,
    remapping the operations in the predict phase to their predicted indices
    to make dependencies clear.

    Parameters
    ----------
    name : str
        The name of the execution plan.
    linearized_ops : list[Op]
        List of operations in topological order.
    split_pos : int, optional
        The position of the split/CV operation in the linearized list.
    """
    op_idx = {op: i for i, op in enumerate(linearized_ops)}
    post_split = linearized_ops[split_pos + 1:] if split_pos is not None else []
    idx_width = len(str(len(linearized_ops) - 1 + len(post_split)))

    fmt_idx = lambda i: f"[{i:0{idx_width}d}]"

    # Compute layout widths for nice arrow alignment
    pre_split_len = split_pos + 1 if split_pos is not None else None
    widths = [
        5 + idx_width + len(str(op))
        for op in linearized_ops[:pre_split_len]
    ]
    if split_pos is not None:
        widths += [7 + idx_width + len(str(op)) for op in post_split]
    arrow_col = max(widths, default=20) + 2

    def fmt(
        display_i: int,
        op: Op,
        indent: str = "  ",
        remap: dict[int, int] | None = None,
    ) -> str:
        r = remap or {}
        inputs = [
            fmt_idx(r.get(op_idx[inp], op_idx[inp])) if inp in op_idx else f"?[{inp}]"
            for inp in op.inputs
        ]
        prefix = f"{indent}{fmt_idx(display_i)} {op}"
        return f"{prefix:<{arrow_col}}<- ({', '.join(inputs)})"

    lines = [f"\n=== Plan: {name} ===", ""]
    if split_pos is None:
        lines.extend(fmt(i, op) for i, op in enumerate(linearized_ops))
    else:
        lines.extend(fmt(i, linearized_ops[i]) for i in range(split_pos))
        lines.extend([
            "",
            "CV Loop:",
            fmt(split_pos, linearized_ops[split_pos]),
            "",
            "  Fit Phase:",
        ])
        
        fit_start = split_pos + 1
        predict_start = fit_start + len(post_split)
        remap = {fit_start + j: predict_start + j for j in range(len(post_split))}

        lines.extend(
            fmt(fit_start + j, op, indent="    ")
            for j, op in enumerate(post_split)
        )
        lines.extend(["", "  Transform / Predict:"])
        lines.extend(
            fmt(predict_start + j, op, indent="    ", remap=remap)
            for j, op in enumerate(post_split)
        )
        lines.extend([
            "",
            f"Total: {split_pos} pre-split  |  1 split  |  {len(post_split)} fit  |  {len(post_split)} predict\n",
        ])

    print("\n".join(lines))
