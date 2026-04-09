from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import plotly.graph_objects as go


# -----------------------------
# Geometry + continuation core
# -----------------------------


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def fk_xy(q, lengths):
    q1, q2, q3 = q
    l1, l2, l3 = lengths
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
    return np.array([x, y], dtype=float)


def fk_points(q, lengths):
    q1, q2, q3 = q
    l1, l2, l3 = lengths

    p0 = np.array([0.0, 0.0])
    p1 = p0 + np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    p2 = p1 + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    p3 = p2 + np.array([l3 * np.cos(q1 + q2 + q3), l3 * np.sin(q1 + q2 + q3)])
    return np.vstack([p0, p1, p2, p3])


def jacobian_xy(q, lengths):
    q1, q2, q3 = q
    l1, l2, l3 = lengths

    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)

    return np.array(
        [
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
        ],
        dtype=float,
    )


def project_to_xy_constraint(
    q_init,
    target_xy,
    lengths,
    *,
    max_iters=30,
    tol=1e-11,
    damping=1e-10,
):
    q = np.asarray(q_init, dtype=float).copy()
    target_xy = np.asarray(target_xy, dtype=float)

    for _ in range(max_iters):
        err = fk_xy(q, lengths) - target_xy
        if np.linalg.norm(err) < tol:
            return q, True

        J = jacobian_xy(q, lengths)
        A = J @ J.T + damping * np.eye(2)
        dq = J.T @ np.linalg.solve(A, err)
        q = q - dq

    ok = np.linalg.norm(fk_xy(q, lengths) - target_xy) < 1e-8
    return q, ok


def tangent_nullspace(q, lengths, previous_tangent=None):
    _, _, vh = np.linalg.svd(jacobian_xy(q, lengths))
    t = vh[-1]
    t = t / np.linalg.norm(t)

    if previous_tangent is not None and np.dot(t, previous_tangent) < 0.0:
        t = -t

    return t


def torus_distance(q_a, q_b):
    diff = wrap_to_pi(np.asarray(q_a) - np.asarray(q_b))
    return float(np.linalg.norm(diff))


def wrist_c2(lengths, target_xy, phi_seed):
    l1, l2, l3 = lengths
    x, y = target_xy
    wx = x - l3 * np.cos(phi_seed)
    wy = y - l3 * np.sin(phi_seed)
    return (wx**2 + wy**2 - l1**2 - l2**2) / (2.0 * l1 * l2)


def seed_configuration_from_phi(lengths, target_xy, *, phi_seed=0.2, elbow=+1):
    """
    Use a 2R wrist construction to obtain one valid start seed.
    The manifold itself is then traced by nullspace continuation.
    """
    l1, l2, l3 = lengths
    x, y = target_xy

    wx = x - l3 * np.cos(phi_seed)
    wy = y - l3 * np.sin(phi_seed)

    c2 = (wx**2 + wy**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if c2 < -1.0 or c2 > 1.0:
        raise RuntimeError(
            "No valid seed found. Change the target or link lengths."
        )

    c2 = np.clip(c2, -1.0, 1.0)
    s2 = (1.0 if elbow >= 0 else -1.0) * np.sqrt(max(0.0, 1.0 - c2**2))
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(wy, wx) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q3 = phi_seed - q1 - q2
    return np.array([q1, q2, q3], dtype=float)


def candidate_phi_sequence(preferred_phi=0.0, n=361):
    preferred_phi = float(preferred_phi)
    deltas = np.linspace(0.0, np.pi, max(3, n // 2))
    out = [wrap_to_pi(preferred_phi)]
    for d in deltas[1:]:
        out.append(wrap_to_pi(preferred_phi + d))
        out.append(wrap_to_pi(preferred_phi - d))
    # preserve order while removing duplicates
    dedup = []
    seen = set()
    for phi in out:
        key = round(float(phi), 10)
        if key not in seen:
            seen.add(key)
            dedup.append(float(phi))
    return dedup


def find_valid_phi_seed(lengths, target_xy, branch=+1, preferred_phi=0.0):
    for phi in candidate_phi_sequence(preferred_phi=preferred_phi):
        c2 = wrist_c2(lengths, target_xy, phi)
        if -1.0 <= c2 <= 1.0:
            try:
                seed_configuration_from_phi(
                    lengths, target_xy, phi_seed=phi, elbow=branch
                )
                return phi
            except RuntimeError:
                continue
    raise RuntimeError(
        "No valid seed found for these parameters. The target is likely outside the reachable set for this branch."
    )


def trace_self_motion_manifold(
    q0,
    target_xy,
    lengths,
    *,
    ds=0.03,
    max_steps=4000,
    min_steps_before_closure=120,
    closure_tol=0.07,
):
    q, ok = project_to_xy_constraint(q0, target_xy, lengths)
    if not ok:
        raise RuntimeError(
            "Could not project the start configuration onto the position constraint."
        )

    q_start = q.copy()
    t_start = tangent_nullspace(q, lengths)
    t = t_start.copy()
    qs = [q.copy()]

    for k in range(max_steps):
        step = ds
        success = False

        for _ in range(10):
            q_pred = q + step * t
            q_new, ok = project_to_xy_constraint(q_pred, target_xy, lengths)
            moved = torus_distance(q_new, q) > 1e-5

            if ok and moved:
                success = True
                break
            step *= 0.5

        if not success:
            raise RuntimeError(
                "Continuation failed. Reduce ds or change the parameters."
            )

        t_new = tangent_nullspace(q_new, lengths, previous_tangent=t)
        qs.append(q_new.copy())
        q = q_new
        t = t_new

        if (
            k > min_steps_before_closure
            and torus_distance(q, q_start) < closure_tol
            and np.dot(t, t_start) > 0.7
        ):
            qs.append(q_start.copy())
            return np.array(qs)

    raise RuntimeError(
        "No closed loop found. Increase max_steps or reduce ds."
    )


def get_branch_curve(lengths, target_xy, *, branch=+1, ds=0.03, preferred_phi=None):
    if preferred_phi is None:
        preferred_phi = 0.30 if branch >= 0 else -0.30
    phi_seed = find_valid_phi_seed(
        lengths=lengths,
        target_xy=target_xy,
        branch=branch,
        preferred_phi=preferred_phi,
    )
    q0 = seed_configuration_from_phi(
        lengths,
        target_xy,
        phi_seed=phi_seed,
        elbow=branch,
    )
    qs = trace_self_motion_manifold(q0, target_xy, lengths, ds=ds)
    return qs, phi_seed


# -----------------------------
# Analysis helpers
# -----------------------------


def periodic_embedding_3d(qs):
    emb6 = np.column_stack(
        [
            np.cos(qs[:, 0]),
            np.sin(qs[:, 0]),
            np.cos(qs[:, 1]),
            np.sin(qs[:, 1]),
            np.cos(qs[:, 2]),
            np.sin(qs[:, 2]),
        ]
    )
    mean6 = np.mean(emb6, axis=0, keepdims=True)
    centered = emb6 - mean6
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    emb3 = centered @ vt[:3].T
    return emb3


def cumulative_torus_arclength(qs):
    s = np.zeros(len(qs))
    for i in range(1, len(qs)):
        s[i] = s[i - 1] + torus_distance(qs[i], qs[i - 1])
    return s


def progress_to_index(qs, progress_01):
    progress_01 = float(np.clip(progress_01, 0.0, 1.0))
    if len(qs) <= 1:
        return 0
    return int(round(progress_01 * (len(qs) - 1)))


def normalized_curve_progress(qs):
    s = cumulative_torus_arclength(qs)
    if s[-1] <= 1e-12:
        return np.zeros_like(s)
    return s / s[-1]


def unwrap_curve(qs):
    return np.unwrap(qs, axis=0)


def validate_lengths(lengths):
    lengths = tuple(float(v) for v in lengths)
    if any(v <= 0 for v in lengths):
        raise ValueError("All link lengths must be positive.")
    return lengths


# -----------------------------
# Plotly figure factories
# -----------------------------


def make_manifold_figure(emb, current_idx, branch):
    current_idx = int(np.clip(current_idx, 0, len(emb) - 1))
    z = emb[current_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=emb[:, 0],
            y=emb[:, 1],
            z=emb[:, 2],
            mode="lines",
            name=f"Branch {branch:+d}",
            hovertemplate="z₁=%{x:.3f}<br>z₂=%{y:.3f}<br>z₃=%{z:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[z[0]],
            y=[z[1]],
            z=[z[2]],
            mode="markers",
            name="Current configuration",
            marker=dict(size=6),
            hovertemplate="current point<extra></extra>",
        )
    )
    fig.update_layout(
        title="Self-motion manifold (torus-aware embedding)",
        scene=dict(
            xaxis_title="z₁",
            yaxis_title="z₂",
            zaxis_title="z₃",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def gripper_segments(tip, theta, scale=0.14):
    t = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])

    palm_offset = 0.03 * scale
    jaw_half_width = 0.35 * scale
    finger_length = 0.60 * scale

    palm_center = tip + palm_offset * t
    fb1 = palm_center + jaw_half_width * n
    fb2 = palm_center - jaw_half_width * n
    ft1 = fb1 + finger_length * t
    ft2 = fb2 + finger_length * t
    return (fb1, fb2), (fb1, ft1), (fb2, ft2)


def make_robot_figure(q, lengths, target_xy):
    pts = fk_points(q, lengths)
    theta = float(np.sum(q))
    (a, b), (c, d), (e, f) = gripper_segments(pts[-1], theta, scale=0.35 * lengths[-1])
    total_length = float(sum(lengths))
    pad = 0.15

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="lines+markers",
            name="Robot arm",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[a[0], b[0], None, c[0], d[0], None, e[0], f[0]],
            y=[a[1], b[1], None, c[1], d[1], None, e[1], f[1]],
            mode="lines",
            name="Gripper",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target_xy[0]],
            y=[target_xy[1]],
            mode="markers",
            name="Target",
            marker=dict(symbol="x", size=12),
            hovertemplate="Target: (%{x:.3f}, %{y:.3f})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Current robot configuration",
        xaxis=dict(title="x₁", range=[-total_length - pad, total_length + pad]),
        yaxis=dict(
            title="x₂",
            range=[-total_length - pad, total_length + pad],
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def make_joint_angle_figure(qs, current_idx):
    current_idx = int(np.clip(current_idx, 0, len(qs) - 1))
    s = normalized_curve_progress(qs)
    qu = unwrap_curve(qs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=qu[:, 0], mode="lines", name="q₁"))
    fig.add_trace(go.Scatter(x=s, y=qu[:, 1], mode="lines", name="q₂"))
    fig.add_trace(go.Scatter(x=s, y=qu[:, 2], mode="lines", name="q₃"))

    x_cur = float(s[current_idx])
    y_min = float(np.min(qu))
    y_max = float(np.max(qu))
    fig.add_trace(
        go.Scatter(
            x=[x_cur, x_cur],
            y=[y_min, y_max],
            mode="lines",
            name="Current position",
            line=dict(dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Joint angles along the closed curve",
        xaxis=dict(title="Normalized curve position", range=[0.0, 1.0]),
        yaxis=dict(title="Unwrapped angle [rad]"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def format_q(q):
    return tuple(float(v) for v in q)
