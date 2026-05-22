"""
Clean velocity polytope plots – planar 3R robot
===============================================

Creates two thesis-style figures:

  Fig 1: 3D joint space velocity constraints cube and omega = 0 slice.
  Fig 2: Planar robot configuration with task-space velocity polytopes 
         (full and strong/constrained) at the end-effector.

Run:
    python polytope_plots_clean.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import itertools

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

# Colors chosen to match the reference images and existing plot style
COLOR_WEAK = "#737373"      # dark grey (edges)
FILL_WEAK = "#F3F4F6"       # light grey (fill)
COLOR_STRONG = "#1F77B4"    # blue
FILL_STRONG = "#D8EAF6"     # light blue (fill)
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


def J_v(q, L):
    q1, q2, q3 = q
    l1, l2, l3 = L

    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)

    return np.array([
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [ l1 * c1 + l2 * c12 + l3 * c123,  l2 * c12 + l3 * c123,  l3 * c123],
    ])


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


def set_equal_3d_limits(ax, limit, margin=1.05):
    lim = limit * margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def save_figure(fig, name):
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ─── Figure 1: 3D Joint Space Polytopes ─────────────────────────────────────

def fig1_3d_polytopes():
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Plot the cube ([-1, 1]^3) representing joint velocity limits
    cube_vertices = np.array(list(itertools.product([-1, 1], repeat=3)))
    
    # Draw cube edges
    for i in range(8):
        for j in range(i+1, 8):
            if np.sum(np.abs(cube_vertices[i] - cube_vertices[j])) == 2:
                ax.plot(
                    [cube_vertices[i, 0], cube_vertices[j, 0]],
                    [cube_vertices[i, 1], cube_vertices[j, 1]],
                    [cube_vertices[i, 2], cube_vertices[j, 2]],
                    color=COLOR_WEAK, linewidth=1.5, zorder=1
                )
    
    # Draw cube vertices
    ax.scatter(cube_vertices[:, 0], cube_vertices[:, 1], cube_vertices[:, 2], 
               color="black", s=30, zorder=4)

    # 2. Plot the omega=0 slice (hexagon)
    # The plane is q1 + q2 + q3 = 0, intersection with [-1, 1]^3
    hex_verts = np.array([
        [1, 0, -1],
        [1, -1, 0],
        [0, -1, 1],
        [-1, 0, 1],
        [-1, 1, 0],
        [0, 1, -1]
    ])
    
    # Add polygon fill
    poly = Poly3DCollection([hex_verts], alpha=0.3, facecolor=FILL_STRONG, zorder=2)
    ax.add_collection3d(poly)
    
    # Draw polygon edges
    hex_closed = np.vstack((hex_verts, hex_verts[0]))
    ax.plot(hex_closed[:, 0], hex_closed[:, 1], hex_closed[:, 2], 
            color=COLOR_STRONG, linewidth=2.0, zorder=3)
    
    # Draw polygon vertices
    ax.scatter(hex_verts[:, 0], hex_verts[:, 1], hex_verts[:, 2], 
               color=COLOR_STRONG, s=40, zorder=5)

    ax.set_xlabel(r"$\dot{q}_1\;[\mathrm{rad/s}]$", labelpad=5)
    ax.set_ylabel(r"$\dot{q}_2\;[\mathrm{rad/s}]$", labelpad=5)
    ax.set_zlabel(r"$\dot{q}_3\;[\mathrm{rad/s}]$", labelpad=5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    
    ax.view_init(elev=20, azim=25)
    set_equal_3d_limits(ax, 1.0)
    clean_3d_axes(ax)

    save_figure(fig, "polytope_fig1_3d_clean")


# ─── Figure 2: Workspace Polytopes ──────────────────────────────────────────

def draw_2d_polytope(ax, vertices, edge_color, fill_color, cx, cy, scale, alpha_fill=1.0, zorder=1):
    """Computes the convex hull and draws the resulting 2D polytope."""
    hull = ConvexHull(vertices)
    
    # Extract and close ordered vertices
    ordered_verts = vertices[hull.vertices]
    ordered_verts = np.vstack((ordered_verts, ordered_verts[0]))
    
    # Transform to workspace
    ws_verts = ordered_verts * scale
    ws_verts[:, 0] += cx
    ws_verts[:, 1] += cy
    
    # Plot fill
    ax.fill(ws_verts[:, 0], ws_verts[:, 1], color=fill_color, alpha=alpha_fill, zorder=zorder)
    
    # Plot edges
    ax.plot(ws_verts[:, 0], ws_verts[:, 1], color=edge_color, linewidth=1.5, zorder=zorder+1)
    
    # Plot vertices
    vert_color = "black" if edge_color == COLOR_WEAK else edge_color
    ax.scatter(ws_verts[:-1, 0], ws_verts[:-1, 1], color=vert_color, s=20, zorder=zorder+2)
    
    return ws_verts


def fig2_workspace():
    Jv = J_v(q, L)
    
    # 1. Joint space vertices
    cube_vertices = np.array(list(itertools.product([-1, 1], repeat=3)))
    hex_verts = np.array([
        [1, 0, -1], [1, -1, 0], [0, -1, 1],
        [-1, 0, 1], [-1, 1, 0], [0, 1, -1]
    ])
    
    # 2. Map directly to task space velocities
    ws_cube_verts = (Jv @ cube_vertices.T).T
    ws_hex_verts = (Jv @ hex_verts.T).T
    
    # Kinematics to find the end-effector position
    pts = fk_points(q, L)
    ee = pts[-1]
    
    # Visual scale to overlap velocity on workspace
    scale = 0.55
    
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    clean_2d_axes(ax)
    ax.set_facecolor("white")
    
    # Draw full velocity polytope (weak constraint, allows any omega)
    bnd_weak = draw_2d_polytope(ax, ws_cube_verts, COLOR_WEAK, FILL_WEAK, 
                                ee[0], ee[1], scale, zorder=2)
    
    # Draw constrained velocity polytope (strong constraint, omega=0)
    draw_2d_polytope(ax, ws_hex_verts, COLOR_STRONG, FILL_STRONG, 
                     ee[0], ee[1], scale, alpha_fill=0.8, zorder=5)
    
    # Robot arm on top
    ax.plot(pts[:, 0], pts[:, 1], "-",
            color=COLOR_ROBOT, linewidth=3.0,
            solid_capstyle="round", zorder=10)
    ax.scatter(pts[0, 0], pts[0, 1],
               s=50, marker="s", color="white", edgecolor=COLOR_ROBOT, linewidth=2, zorder=11)
    ax.scatter(pts[1:-1, 0], pts[1:-1, 1],
               s=44, facecolor="white", edgecolor=COLOR_ROBOT,
               linewidth=2.0, zorder=12)
    ax.scatter(ee[0], ee[1], s=36, facecolor="white", edgecolor=COLOR_ROBOT, linewidth=2.0, zorder=13)
    
    # Auto-scale axis limits
    all_x = np.concatenate([pts[:, 0], bnd_weak[:, 0]])
    all_y = np.concatenate([pts[:, 1], bnd_weak[:, 1]])
    px = 0.10 * (all_x.max() - all_x.min())
    py = 0.10 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - px, all_x.max() + px)
    ax.set_ylim(all_y.min() - py, all_y.max() + py)
    
    ax.set_xlabel(r"$x\;[\mathrm{m}],\;\dot{x}\;[\mathrm{m/s}]$")
    ax.set_ylabel(r"$y\;[\mathrm{m}],\;\dot{y}\;[\mathrm{m/s}]$")
    
    fig.tight_layout()
    save_figure(fig, "polytope_fig2_workspace_clean")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig1_3d_polytopes()
    fig2_workspace()
    print("Done.")
