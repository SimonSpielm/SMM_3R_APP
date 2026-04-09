# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.19.11",
#   "numpy>=1.26",
#   "plotly>=5.24",
# ]
# [tool.marimo.display]
# theme = "dark"
# cell_output = "below"
# ///

from __future__ import annotations

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

BRANCH = +1


@app.cell
def _():
    import marimo as mo
    import smm_core

    return mo, smm_core


@app.cell(hide_code=True)
def _(mo):
    mo.md("# 3R Self-Motion Manifold")
    return


@app.cell
def _(mo):
    l1 = mo.ui.slider(
        start=0.20,
        stop=1.20,
        step=0.01,
        value=0.50,
        debounce=True,
        show_value=True,
        include_input=True,
        label="l1",
        full_width=True,
    )
    l2 = mo.ui.slider(
        start=0.20,
        stop=1.20,
        step=0.01,
        value=0.50,
        debounce=True,
        show_value=True,
        include_input=True,
        label="l2",
        full_width=True,
    )
    l3 = mo.ui.slider(
        start=0.20,
        stop=1.20,
        step=0.01,
        value=0.50,
        debounce=True,
        show_value=True,
        include_input=True,
        label="l3",
        full_width=True,
    )
    target_x = mo.ui.slider(
        start=-1.40,
        stop=1.40,
        step=0.01,
        value=1.00,
        debounce=True,
        show_value=True,
        include_input=True,
        label="target x",
        full_width=True,
    )
    target_y = mo.ui.slider(
        start=-1.40,
        stop=1.40,
        step=0.01,
        value=0.50,
        debounce=True,
        show_value=True,
        include_input=True,
        label="target y",
        full_width=True,
    )
    ds = mo.ui.slider(
        start=0.01,
        stop=0.08,
        step=0.005,
        value=0.03,
        debounce=True,
        show_value=True,
        include_input=True,
        label="ds",
        full_width=True,
    )
    progress_pct = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=0,
        show_value=True,
        include_input=True,
        label="position [%]",
        full_width=True,
    )

    controls = mo.vstack(
        [
            mo.hstack([l1, l2, l3], widths="equal"),
            mo.hstack([target_x, target_y, ds], widths="equal"),
            progress_pct,
        ],
        gap=1,
    )
    controls
    return ds, l1, l2, l3, progress_pct, target_x, target_y


@app.cell
def _(ds, l1, l2, l3, progress_pct, smm_core, target_x, target_y):
    lengths = smm_core.validate_lengths((l1.value, l2.value, l3.value))
    target_xy = (float(target_x.value), float(target_y.value))
    progress_01 = float(progress_pct.value) / 100.0

    result = {
        "ok": False,
        "message": "",
        "branch": 1,
        "lengths": lengths,
        "target_xy": target_xy,
        "progress_01": progress_01,
        "phi_seed": None,
        "qs": None,
        "emb": None,
        "current_idx": 0,
        "current_q": None,
        "curve_points": 0,
        "curve_length": 0.0,
    }

    try:
        qs, phi_seed = smm_core.get_branch_curve(
            lengths=lengths,
            target_xy=target_xy,
            branch=1,
            ds=float(ds.value),
        )
        emb = smm_core.periodic_embedding_3d(qs)
        current_idx = smm_core.progress_to_index(qs, progress_01)
        curve_length = float(smm_core.cumulative_torus_arclength(qs)[-1])

        result.update(
            {
                "ok": True,
                "phi_seed": float(phi_seed),
                "qs": qs,
                "emb": emb,
                "current_idx": current_idx,
                "current_q": qs[current_idx],
                "curve_points": int(len(qs)),
                "curve_length": curve_length,
            }
        )
    except Exception as exc:
        result["message"] = str(exc)

    return result


@app.cell(hide_code=True)
def _(mo, result):
    if result["ok"]:
        q1, q2, q3 = [float(v) for v in result["current_q"]]
        status = mo.md(
            f"`branch {result['branch']:+d}`  |  `seed {result['phi_seed']:.3f}`  |  `points {result['curve_points']}`  |  `length {result['curve_length']:.3f}`  |  `q ({q1:.3f}, {q2:.3f}, {q3:.3f})`"
        )
    else:
        status = mo.md(f"**Error:** `{result['message']}`").callout(kind="danger")
    status
    return status


@app.cell
def _(mo, result, smm_core):
    if not result["ok"]:
        manifold_plot = mo.md("No 3D plot.")
        robot_plot = mo.md("No robot plot.")
        joint_plot = mo.md("No joint plot.")
    else:
        manifold_plot = mo.ui.plotly(
            smm_core.make_manifold_figure(
                result["emb"], result["current_idx"], result["branch"]
            )
        )
        robot_plot = mo.ui.plotly(
            smm_core.make_robot_figure(
                result["current_q"], result["lengths"], result["target_xy"]
            )
        )
        joint_plot = mo.ui.plotly(
            smm_core.make_joint_angle_figure(result["qs"], result["current_idx"])
        )
    return joint_plot, manifold_plot, robot_plot


@app.cell(hide_code=True)
def _(joint_plot, manifold_plot, mo, robot_plot, status):
    mo.vstack(
        [
            status,
            mo.hstack([manifold_plot, robot_plot], widths="equal"),
            joint_plot,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
