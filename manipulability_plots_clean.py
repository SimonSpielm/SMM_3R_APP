"""
Clean manipulability ellipsoid plots – planar 3R robot
======================================================

Creates two thesis-style figures:

  Fig. 1: Full task-space manipulability ellipsoid and strong translational
          ellipse in the omega = 0 plane.
  Fig. 2: Planar robot configuration with weak and strong translational
          manipulability ellipses at the end-effector.

Run:
    python manipulability_plots_clean.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ─── Plot style ──────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors chosen to keep the figures readable in print.
COLOR_WEAK = "#6B7280"      # neutral grey
COLOR_STRONG = "#1F77B4"    # blue
COLOR_ROBOT = "#111111"
COLOR_GRID = "#E5E7EB"


# ─── Link lengths and joint configuration ───────────────────────────────────

L = (0.4, 0.4, 0.4)

# IK: find q such that fk_xy(q, L) = (0, 1)
_phi = np.pi / 4
_l1, _l2, _l3 = L
_wx = 0.0 - _l3 * np.cos(_phi)
_wy = 1.0 - _l3 * np.sin(_phi)
_c2 = (_wx**2 + _wy**2 - _l1**2 - _l2**2) / (2 * _l1 * _l2)
_c2 = float(np.clip(_c2, -1.0, 1.0))
_s2 = -np.sqrt(1.0 - _c2**2)
_q2 = np.arctan2(_s2, _c2)
_q1 = np.arctan2(_wy, _wx) - np.arctan2(_l2 * _s2, _l1 + _l2 * _c2)
_q3 = _phi - _q1 - _q2
q = np.array([_q1, _q2, _q3])


# ─── Kinematics ──────────────────────────────────────────────────────────────

def fk_points(q, L):
    q1, q2, q3 = q
    l1, l2, l3 = L
    p0 = np.zeros(2)
    p1 = p0 + l1 * np.array([np.cos(q1), np.sin(q1)])
    p2 = p1 + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    p3 = p2 + l3 * np.array([np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)])
    return np.vstack([p0, p1, p2, p3])


def J_full(q, L):
    """3 x 3 Jacobian with rows [xdot, ydot, omega]."""
    q1, q2, q3 = q
    l1, l2, l3 = L

    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)

    Jv = np.array([
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [ l1 * c1 + l2 * c12 + l3 * c123,  l2 * c12 + l3 * c123,  l3 * c123],
    ])
    Jo = np.ones((1, 3))
    return np.vstack([Jv, Jo])


def J_v(q, L):
    return J_full(q, L)[:2, :]


def J_o(q, L):
    return J_full(q, L)[2:, :]


def strong_translational_jacobian(q, L):
    Jv = J_v(q, L)
    Jo = J_o(q, L)
    N_omega = np.eye(Jo.shape[1]) - np.linalg.pinv(Jo) @ Jo
    return Jv @ N_omega


# ─── Ellipsoid parametrization ───────────────────────────────────────────────

def ellipsoid_points(M3, n_theta=40, n_phi=80):
    """Parametrize x^T M^{-1} x = 1 for a 3D SPD matrix M."""
    vals, vecs = np.linalg.eigh(M3)
    vals = np.maximum(vals, 1e-14)

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    sphere = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH),
    ], axis=-1)

    transform = vecs @ np.diag(np.sqrt(vals))
    points = sphere @ transform.T
    return points[..., 0], points[..., 1], points[..., 2]


def ellipse2d_points(M2, n=400):
    """Parametrize x^T M^{-1} x = 1 for a 2D SPD matrix M."""
    vals, vecs = np.linalg.eigh(M2)
    vals = np.maximum(vals, 1e-14)

    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    transform = vecs @ np.diag(np.sqrt(vals))
    points = transform @ circle
    return points[0], points[1]


def principal_axes_2d(M2):
    vals, vecs = np.linalg.eigh(M2)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs


# ─── Helpers ─────────────────────────────────────────────────────────────────

def clean_2d_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8)


def clean_3d_axes(ax):
    ax.grid(True, linewidth=0.35)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#D1D5DB")
    ax.yaxis.pane.set_edgecolor("#D1D5DB")
    ax.zaxis.pane.set_edgecolor("#D1D5DB")
    ax.tick_params(pad=2, labelsize=8)


def set_equal_3d_limits(ax, X, Y, Z, margin=1.05):
    lim = max(np.abs(X).max(), np.abs(Y).max(), np.abs(Z).max()) * margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 0.8))


def save_figure(fig, name):
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ─── Figure 1: 3D ellipsoids ────────────────────────────────────────────────

def fig1_3d_ellipsoids():
    # Scale J_omega by L_char so the ω axis has comparable magnitude to
    # the translational axes (avoids a needle-shaped ellipsoid).
    L_char = float(np.mean(L))
    Jv  = J_v(q, L)
    Jo  = L_char * np.ones((1, 3))
    Jf_scaled = np.vstack([Jv, Jo])
    Jvs = strong_translational_jacobian(q, L)

    M_full   = Jf_scaled @ Jf_scaled.T
    M_strong = Jvs @ Jvs.T

    X, Y, Z = ellipsoid_points(M_full, n_theta=50, n_phi=100)
    ex, ey  = ellipse2d_points(M_strong)
    ez = np.zeros_like(ex)

    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    # Transparent surface plus sparse wireframe gives a cleaner figure than
    # many individually drawn curves.
    ax.plot_surface(
        X, Y, Z,
        color=COLOR_WEAK,
        alpha=0.06,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.plot_wireframe(
        X, Y, Z,
        rstride=2,
        cstride=2,
        color=COLOR_WEAK,
        linewidth=0.35,
        alpha=0.50,
    )

    # Strong translational ellipse in the omega = 0 plane.
    ax.plot(
        ex, ey, ez,
        color=COLOR_STRONG,
        linewidth=2.0,
        #label=r"Strong translational $(\omega=0)$",
    )

    # Subtle zero plane reference.
    lim_xy = max(np.abs(X).max(), np.abs(Y).max())
    ax.plot([-lim_xy, lim_xy], [0, 0], [0, 0], color="#9CA3AF", linewidth=0.8, alpha=0.7)
    ax.plot([0, 0], [-lim_xy, lim_xy], [0, 0], color="#9CA3AF", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"$\dot{x}\;[\mathrm{m/s}]$", labelpad=2)
    ax.set_ylabel(r"$\dot{y}\;[\mathrm{m/s}]$", labelpad=2)
    ax.set_zlabel(r"$\omega\;[\mathrm{rad/s}]$", labelpad=2)
    # z-label: keep default rotation (avoids overlap with tick labels)
    ax.view_init(elev=22, azim=-55)

    set_equal_3d_limits(ax, X, Y, Z)
    clean_3d_axes(ax)
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "manipulability_fig1_3d_clean")


# ─── Figure 2: workspace view ────────────────────────────────────────────────

def _ellipse_mesh(ax, M, cx, cy, scale, color,
                  n_rings=12, n_radial=16, lw=0.55, alpha=0.75):
    """Draw a mesh of concentric ellipses + radial lines (reference style)."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, 1e-14)
    # Transform: maps unit circle → ellipse scaled by `scale`
    A = vecs @ np.diag(np.sqrt(vals)) * scale   # 2×2

    # Concentric rings
    t = np.linspace(0, 2 * np.pi, 360)
    circle = np.array([np.cos(t), np.sin(t)])
    for frac in np.linspace(1.0 / n_rings, 1.0, n_rings):
        pts = frac * A @ circle
        ax.plot(cx + pts[0], cy + pts[1], color=color, lw=lw, alpha=alpha)

    # Radial lines (full diameters, evenly spaced in angle)
    angles = np.linspace(0, np.pi, n_radial, endpoint=False)
    for angle in angles:
        d = A @ np.array([np.cos(angle), np.sin(angle)])
        ax.plot([cx - d[0], cx + d[0]],
                [cy - d[1], cy + d[1]],
                color=color, lw=lw, alpha=alpha)


def fig2_workspace():
    Jv  = J_v(q, L)
    Jvs = strong_translational_jacobian(q, L)

    M_weak   = Jv  @ Jv.T
    M_strong = Jvs @ Jvs.T

    pts = fk_points(q, L)
    ee  = pts[-1]

    scale = 0.55   # visual scale: velocity [m/s] → spatial [m] for display

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    clean_2d_axes(ax)
    ax.set_facecolor("white")

    # Weak ellipse mesh (grey, behind everything)
    _ellipse_mesh(ax, M_weak, ee[0], ee[1], scale,
                  color=COLOR_WEAK, n_rings=12, n_radial=16, lw=0.55, alpha=0.80)

    # Strong ellipse mesh (blue, on top)
    _ellipse_mesh(ax, M_strong, ee[0], ee[1], scale,
                  color=COLOR_STRONG, n_rings=8, n_radial=12, lw=0.70, alpha=0.90)

    # Outer boundary lines (slightly thicker)
    for M, color, lw_outer in [(M_weak, COLOR_WEAK, 1.0), (M_strong, COLOR_STRONG, 1.6)]:
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, 1e-14)
        A = vecs @ np.diag(np.sqrt(vals)) * scale
        t = np.linspace(0, 2 * np.pi, 360)
        bnd = A @ np.array([np.cos(t), np.sin(t)])
        ax.plot(ee[0] + bnd[0], ee[1] + bnd[1], color=color, lw=lw_outer, alpha=1.0)

    # Robot arm on top
    ax.plot(pts[:, 0], pts[:, 1], "-",
            color=COLOR_ROBOT, linewidth=2.6,
            solid_capstyle="round", zorder=10)
    ax.scatter(pts[0, 0], pts[0, 1],
               s=50, marker="s", color=COLOR_ROBOT, zorder=11)
    ax.scatter(pts[1:-1, 0], pts[1:-1, 1],
               s=44, facecolor="white", edgecolor=COLOR_ROBOT,
               linewidth=1.2, zorder=12)
    ax.scatter(ee[0], ee[1], s=36, color=COLOR_ROBOT, zorder=13)

    # Axis limits: tight around all drawn content
    vals_w, vecs_w = np.linalg.eigh(M_weak)
    A_w = vecs_w @ np.diag(np.sqrt(np.maximum(vals_w, 0))) * scale
    t   = np.linspace(0, 2 * np.pi, 360)
    bx  = ee[0] + (A_w @ np.array([np.cos(t), np.sin(t)]))[0]
    by  = ee[1] + (A_w @ np.array([np.cos(t), np.sin(t)]))[1]
    all_x = np.concatenate([pts[:, 0], bx])
    all_y = np.concatenate([pts[:, 1], by])
    px = 0.10 * (all_x.max() - all_x.min())
    py = 0.10 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - px, all_x.max() + px)
    ax.set_ylim(all_y.min() - py, all_y.max() + py)

    ax.set_xlabel(r"$x\;[\mathrm{m}],\;\dot{x}\;[\mathrm{m/s}]$")
    ax.set_ylabel(r"$y\;[\mathrm{m}],\;\dot{y}\;[\mathrm{m/s}]$")

    fig.tight_layout()
    save_figure(fig, "manipulability_fig2_workspace_clean")



# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig1_3d_ellipsoids()
    fig2_workspace()
    print("Done.")
