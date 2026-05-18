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


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    from types import SimpleNamespace

    # ── Kinematics ────────────────────────────────────────────────────────────

    def wrap(a):
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    def torus_dist(a, b):
        return float(np.linalg.norm(wrap(np.array(a) - np.array(b))))

    def fk_xy(q, L):
        q1, q2, q3 = q
        l1, l2, l3 = L
        return np.array([
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3),
            l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3),
        ])

    def fk_pts(q, L):
        q1, q2, q3 = q
        l1, l2, l3 = L
        p0 = np.zeros(2)
        p1 = p0 + l1 * np.array([np.cos(q1), np.sin(q1)])
        p2 = p1 + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        p3 = p2 + l3 * np.array([np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)])
        return np.vstack([p0, p1, p2, p3])

    def jac_v(q, L):
        q1, q2, q3 = q
        l1, l2, l3 = L
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)
        return np.array([
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [ l1 * c1 + l2 * c12 + l3 * c123,  l2 * c12 + l3 * c123,  l3 * c123],
        ])

    def jac_full(q, L):
        # Full planar pose Jacobian: x, y, phi.
        return np.vstack([jac_v(q, L), np.ones((1, 3))])

    # ── Self-motion manifold parametrisation ───────────────────────────────────
    #
    # For a fixed end-effector target (tx, ty), the 3R planar robot keeps one
    # internal degree of freedom:
    #
    #     dim(q) - dim(task) = 3 - 2 = 1
    #
    # We parametrise this internal motion by the wrist/end-effector orientation:
    #
    #     phi = q1 + q2 + q3
    #
    # For every phi, the wrist point
    #
    #     w = target - l3 * (cos(phi), sin(phi))
    #
    # must be reachable by the 2-link chain (l1, l2). Reachability requires:
    #
    #     |l1 - l2| <= ||w|| <= l1 + l2
    #
    # If this condition holds, the 2-link chain has two IK branches:
    # elbow-up and elbow-down. These branches coincide only at singular
    # configurations, where the arm is fully stretched or folded and det(J)=0.

    def _ik_config(L, tx, ty, phi, elbow):
        l1, l2, l3 = L
        wx = tx - l3 * np.cos(phi)
        wy = ty - l3 * np.sin(phi)
        d2 = wx**2 + wy**2
        c2 = (d2 - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(c2) > 1:
            return None
        s2 = elbow * np.sqrt(max(0.0, 1.0 - c2**2))
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(wy, wx) - np.arctan2(l2 * s2, l1 + l2 * c2)
        q3 = phi - q1 - q2
        return wrap(np.array([q1, q2, q3]))

    def build_loop(L, tx, ty, phi_vals):
        """Build one closed SMM loop from a valid phi arc."""
        up, dn = [], []
        for phi in phi_vals:
            qu = _ik_config(L, tx, ty, phi, 1)
            qd = _ik_config(L, tx, ty, phi, -1)
            if qu is not None:
                up.append(qu)
            if qd is not None:
                dn.append(qd)
        if len(up) < 3 or len(dn) < 3:
            return None
        return np.array(up + list(reversed(dn)) + [up[0]])

    def find_loops(L, tx, ty, n_phi=500):
        """Find valid phi arcs and build closed self-motion loops."""
        l1, l2, l3 = L
        phis = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)

        valid = []
        for phi in phis:
            wx = tx - l3 * np.cos(phi)
            wy = ty - l3 * np.sin(phi)
            d = np.sqrt(wx**2 + wy**2)
            valid.append(abs(l1 - l2) <= d <= l1 + l2)
        valid = np.array(valid)

        if not np.any(valid):
            return []

        # Find connected valid arcs on the periodic phi circle.
        start = 0
        for i in range(n_phi):
            if valid[i] and not valid[(i - 1) % n_phi]:
                start = i
                break

        arcs = []
        current = []
        for i in range(n_phi):
            idx = (start + i) % n_phi
            if valid[idx]:
                current.append(phis[idx])
            else:
                if len(current) >= 5:
                    arcs.append(current)
                current = []
        if len(current) >= 5:
            arcs.append(current)

        loops = []
        for k, arc in enumerate(arcs):
            loop = build_loop(L, tx, ty, arc)
            if loop is not None:
                phi_start = arc[0]
                phi_end = arc[-1]
                qs_start = _ik_config(L, tx, ty, phi_start, 1)
                qs_end = _ik_config(L, tx, ty, phi_end, 1)
                loops.append({
                    "qs": loop,
                    "phi_arc": arc,
                    "label": f"Branch {k + 1}",
                    "sing_start": qs_start,
                    "sing_end": qs_end,
                })
        return loops

    def norm_arc(qs):
        s = np.zeros(len(qs))
        for i in range(1, len(qs)):
            s[i] = s[i - 1] + torus_dist(qs[i], qs[i - 1])
        return s / s[-1] if s[-1] > 0 else s

    def embed3d_joint(loops):
        """Shared 3D embedding so loops and singular points use one coordinate system."""
        sing_qs = []
        for lp in loops:
            for key in ("sing_start", "sing_end"):
                if lp.get(key) is not None:
                    sing_qs.append(lp[key])

        all_qs = np.vstack(
            [lp["qs"] for lp in loops] + ([np.vstack(sing_qs)] if sing_qs else [])
        )
        e = np.column_stack([
            np.cos(all_qs[:, 0]), np.sin(all_qs[:, 0]),
            np.cos(all_qs[:, 1]), np.sin(all_qs[:, 1]),
            np.cos(all_qs[:, 2]), np.sin(all_qs[:, 2]),
        ])
        c = e - e.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ vt[:3].T

        offset = 0
        embs = []
        for lp in loops:
            n = len(lp["qs"])
            embs.append(proj[offset:offset + n])
            offset += n

        sing_embs = []
        for lp in loops:
            d = {}
            for key in ("sing_start", "sing_end"):
                if lp.get(key) is not None:
                    d[key] = proj[offset]
                    offset += 1
            sing_embs.append(d)
        return embs, sing_embs

    def det_along(qs, L):
        return np.array([np.linalg.det(jac_full(q, L)) for q in qs])

    def gripper(tip, theta, scale=0.12):
        t = np.array([np.cos(theta), np.sin(theta)])
        n = np.array([-np.sin(theta), np.cos(theta)])
        pc = tip + 0.03 * scale * t
        fb1 = pc + 0.35 * scale * n
        fb2 = pc - 0.35 * scale * n
        return (fb1, fb2), (fb1, fb1 + 0.60 * scale * t), (fb2, fb2 + 0.60 * scale * t)

    COLORS = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]

    # ── Figure factories ──────────────────────────────────────────────────────

    def fig_manifold(loops, embs, sing_embs, idxs):
        fig = go.Figure()
        for k, (lp, emb, semb, idx) in enumerate(zip(loops, embs, sing_embs, idxs)):
            col = COLORS[k % len(COLORS)]
            idx = int(np.clip(idx, 0, len(emb) - 1))
            fig.add_trace(go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="lines", name=lp["label"], line=dict(color=col, width=4),
            ))
            z = emb[idx]
            fig.add_trace(go.Scatter3d(
                x=[z[0]], y=[z[1]], z=[z[2]],
                mode="markers", showlegend=False,
                marker=dict(color=col, size=9, symbol="circle", line=dict(color="white", width=2)),
            ))

            for skey, slabel in [("sing_start", "Singularity (arc start)"), ("sing_end", "Singularity (arc end)")]:
                if skey in semb:
                    sz = semb[skey]
                    fig.add_trace(go.Scatter3d(
                        x=[sz[0]], y=[sz[1]], z=[sz[2]],
                        mode="markers",
                        name="Singularity" if (k == 0 and skey == "sing_start") else None,
                        showlegend=(k == 0 and skey == "sing_start"),
                        marker=dict(color="#F59E0B", size=11, symbol="diamond", line=dict(color="white", width=2)),
                        hovertemplate=f"{lp['label']} {slabel}<extra></extra>",
                    ))

        all_spts = []
        for k, (lp, semb) in enumerate(zip(loops, sing_embs)):
            for skey in ("sing_start", "sing_end"):
                if skey in semb:
                    all_spts.append((semb[skey], k, skey))

        shown_conn = False
        for i in range(len(all_spts)):
            for j in range(i + 1, len(all_spts)):
                pi, ki, key_i = all_spts[i]
                pj, kj, key_j = all_spts[j]
                if ki == kj:
                    continue
                qi = loops[ki][key_i]
                qj = loops[kj][key_j]
                if qi is None or qj is None:
                    continue
                dist = float(np.linalg.norm(wrap(qi - qj)))
                if dist < 0.5:
                    fig.add_trace(go.Scatter3d(
                        x=[pi[0], pj[0]], y=[pi[1], pj[1]], z=[pi[2], pj[2]],
                        mode="lines",
                        name="Singular connection" if not shown_conn else None,
                        showlegend=not shown_conn,
                        line=dict(color="#F59E0B", width=3, dash="dot"),
                        hoverinfo="skip",
                    ))
                    shown_conn = True

        fig.update_layout(
            title="Self-motion manifold — torus embedding (orange = singularities)",
            scene=dict(xaxis_title="z₁", yaxis_title="z₂", zaxis_title="z₃", aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    def fig_robot(loops, idxs, L, tgt):
        fig = go.Figure()
        total = sum(L)
        pad = 0.15
        dashes = ["solid", "dash", "dot", "dashdot"]

        for k, (lp, idx) in enumerate(zip(loops, idxs)):
            col = COLORS[k % len(COLORS)]
            q = lp["qs"][int(np.clip(idx, 0, len(lp["qs"]) - 1))]
            pts = fk_pts(q, L)
            theta = float(np.sum(q))
            (a, b), (c, d), (e, f) = gripper(pts[-1], theta, scale=0.35 * L[-1])
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name=lp["label"],
                line=dict(color=col, width=2.5, dash=dashes[k % len(dashes)]),
                marker=dict(color=col, size=7),
            ))
            fig.add_trace(go.Scatter(
                x=[a[0], b[0], None, c[0], d[0], None, e[0], f[0]],
                y=[a[1], b[1], None, c[1], d[1], None, e[1], f[1]],
                mode="lines", showlegend=False,
                line=dict(color=col, width=2, dash=dashes[k % len(dashes)]),
            ))

        fig.add_trace(go.Scatter(
            x=[tgt[0]], y=[tgt[1]], mode="markers", name="Target",
            marker=dict(symbol="x", size=14, color="#F59E0B", line_width=2),
        ))
        fig.update_layout(
            title="Robot configurations",
            xaxis=dict(title="x [m]", range=[-total - pad, total + pad]),
            yaxis=dict(title="y [m]", range=[-total - pad, total + pad], scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    def fig_det(loops, idxs, L):
        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="#6B7280", line_width=1)
        for k, (lp, idx) in enumerate(zip(loops, idxs)):
            col = COLORS[k % len(COLORS)]
            s = norm_arc(lp["qs"])
            d = det_along(lp["qs"], L)
            fig.add_trace(go.Scatter(x=s, y=d, mode="lines", name=lp["label"], line=dict(color=col, width=2)))
            xi = float(s[int(np.clip(idx, 0, len(s) - 1))])
            fig.add_vline(x=xi, line_dash="dot", line_color=col, line_width=1.5)
        fig.update_layout(
            title="det(J) along each branch — zeros mark singularities",
            xaxis=dict(title="Normalised arc length s"),
            yaxis=dict(title="det(J)"),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    def fig_joints(loops, idxs):
        fig = go.Figure()
        for k, (lp, idx) in enumerate(zip(loops, idxs)):
            col = COLORS[k % len(COLORS)]
            s = norm_arc(lp["qs"])
            qu = np.unwrap(lp["qs"], axis=0)
            opacities = [1.0, 0.6, 0.35]
            for i, op in enumerate(opacities):
                r = int(col[1:3], 16)
                g = int(col[3:5], 16)
                b = int(col[5:7], 16)
                fig.add_trace(go.Scatter(
                    x=s, y=qu[:, i], mode="lines", name=f"q{i + 1} {lp['label']}",
                    line=dict(color=f"rgba({r},{g},{b},{op})", width=1.8),
                ))
            xi = float(s[int(np.clip(idx, 0, len(s) - 1))])
            fig.add_vline(x=xi, line_dash="dot", line_color=col, line_width=1.5)
        fig.update_layout(
            title="Joint angles along each branch",
            xaxis=dict(title="Normalised arc length s"),
            yaxis=dict(title="Unwrapped angle [rad]"),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    core = SimpleNamespace(
        find_loops=find_loops,
        embed3d_joint=embed3d_joint,
        norm_arc=norm_arc,
        fig_manifold=fig_manifold,
        fig_robot=fig_robot,
        fig_det=fig_det,
        fig_joints=fig_joints,
    )
    return COLORS, core, go, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# 3R Planar Robot — Self-Motion Manifolds and Singular Branch Transitions

This notebook visualises the topology of self-motion manifolds using a planar 3R redundant robot.

For a fixed end-effector position, the robot still has one internal degree of freedom:

\[
3 - 2 = 1
\]

Therefore, the inverse kinematics does not yield a single configuration, but a continuous family of configurations called a **self-motion manifold**.

## Elbow-Up and Elbow-Down Branches

The manifold is parametrised by the wrist orientation:

\[
\phi = q_1 + q_2 + q_3
\]

For each valid wrist position, the 2-link subsystem admits two IK solutions:

- elbow-up
- elbow-down

These are different branches of the self-motion manifold.

## Singularities

The branches meet only at singular configurations, where the arm becomes fully stretched or folded:

\[
\det(J)=0
\]

At these points, the Jacobian loses rank and the IK branches collapse to the same configuration. Therefore, transitions between branches require passing through a singularity.

The orange markers denote singular transition points.

## Disconnected Manifolds

For certain link lengths and target positions, the valid range of \(\phi\) splits into multiple disconnected intervals. Each interval generates a separate closed self-motion loop.

> Default example: \(l_1=0.50,\ l_2=0.20,\ l_3=0.30,\ target=(0,0.50)\), producing two disconnected self-motion branches.
""")


@app.cell
def _(mo):
    l1 = mo.ui.slider(0.20, 0.70, 0.01, value=0.50, label="l₁", show_value=True, include_input=True, full_width=True, debounce=True)
    l2 = mo.ui.slider(0.10, 0.50, 0.01, value=0.20, label="l₂", show_value=True, include_input=True, full_width=True, debounce=True)
    l3 = mo.ui.slider(0.10, 0.50, 0.01, value=0.30, label="l₃", show_value=True, include_input=True, full_width=True, debounce=True)
    tx = mo.ui.slider(-0.60, 0.60, 0.01, value=0.00, label="target x", show_value=True, include_input=True, full_width=True, debounce=True)
    ty = mo.ui.slider(0.10, 1.00, 0.01, value=0.50, label="target y", show_value=True, include_input=True, full_width=True, debounce=True)
    p1 = mo.ui.slider(0, 100, 1, value=25, label="Branch 1 — position [%]", show_value=True, include_input=True, full_width=True)
    p2 = mo.ui.slider(0, 100, 1, value=75, label="Branch 2 — position [%]", show_value=True, include_input=True, full_width=True)

    mo.vstack([
        mo.hstack([l1, l2, l3], widths="equal"),
        mo.hstack([tx, ty], widths="equal"),
        mo.hstack([p1, p2], widths="equal"),
    ], gap=1)
    return l1, l2, l3, p1, p2, tx, ty


@app.cell
def _(core, l1, l2, l3, np, p1, p2, tx, ty):
    _L = (float(l1.value), float(l2.value), float(l3.value))
    _T = (float(tx.value), float(ty.value))

    _loops = core.find_loops(_L, _T[0], _T[1])

    _sing_embs = []
    if len(_loops) >= 1:
        _embs, _sing_embs = core.embed3d_joint(_loops)
        for _lp, _emb in zip(_loops, _embs):
            _lp["emb"] = _emb

    _pcts = [p1.value, p2.value]
    _idxs = []
    for _k, _lp in enumerate(_loops):
        _pct = _pcts[_k] if _k < len(_pcts) else 50
        _n = len(_lp["qs"])
        _idxs.append(int(np.clip(round(_pct / 100 * (_n - 1)), 0, _n - 1)))

    result = {"L": _L, "T": _T, "loops": _loops, "idxs": _idxs, "sing_embs": _sing_embs}
    return (result,)


@app.cell(hide_code=True)
def _(mo, result, np):
    _loops = result["loops"]
    _L = result["L"]
    _T = result["T"]
    _dist = float(np.sqrt(_T[0]**2 + _T[1]**2))
    _l1, _l2, _l3 = _L

    if len(_loops) == 0:
        _info = mo.callout(mo.md(
            f"**No valid branch found.** Target distance={_dist:.3f}. "
            f"Try moving the target or adjusting link lengths."
        ), kind="danger")
    elif len(_loops) == 1:
        _info = mo.callout(mo.md(
            f"**Only one branch found** ({len(_loops[0]['qs'])} points). "
            f"To get two separate branches, try **l₁=0.50, l₂=0.20, l₃=0.30, target=(0, 0.50)**."
        ), kind="warn")
    else:
        _parts = " &nbsp;|&nbsp; ".join(f"**{lp['label']}**: {len(lp['qs'])} points" for lp in _loops)
        _info = mo.md(f"✓ **{len(_loops)} disconnected branches found** &nbsp;|&nbsp; {_parts}")
    _info


@app.cell
def _(core, mo, result):
    _loops = result["loops"]
    _idxs = result["idxs"]
    _L = result["L"]
    _T = result["T"]

    if len(_loops) == 0:
        _layout = mo.md("Adjust parameters to find manifold branches.")
    else:
        _embs = [lp["emb"] for lp in _loops]
        _sing_embs = result.get("sing_embs", [{} for _ in _loops])
        _fm = core.fig_manifold(_loops, _embs, _sing_embs, _idxs)
        _fr = core.fig_robot(_loops, _idxs, _L, _T)
        _fd = core.fig_det(_loops, _idxs, _L)
        _fj = core.fig_joints(_loops, _idxs)
        _layout = mo.vstack([
            mo.hstack([mo.ui.plotly(_fm), mo.ui.plotly(_fr)], widths="equal"),
            mo.hstack([mo.ui.plotly(_fj), mo.ui.plotly(_fd)], widths="equal"),
        ], gap=1)
    _layout


if __name__ == "__main__":
    app.run()
