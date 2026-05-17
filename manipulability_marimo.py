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

    def wrap_to_pi(a):
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    def fk_xy(q, L):
        q1,q2,q3 = q; l1,l2,l3 = L
        x = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
        y = l1*np.sin(q1) + l2*np.sin(q1+q2) + l3*np.sin(q1+q2+q3)
        return np.array([x, y])

    def fk_points(q, L):
        q1,q2,q3 = q; l1,l2,l3 = L
        p0 = np.zeros(2)
        p1 = p0 + l1*np.array([np.cos(q1), np.sin(q1)])
        p2 = p1 + l2*np.array([np.cos(q1+q2), np.sin(q1+q2)])
        p3 = p2 + l3*np.array([np.cos(q1+q2+q3), np.sin(q1+q2+q3)])
        return np.vstack([p0,p1,p2,p3])

    def jac_v(q, L):
        """2×3 translational Jacobian."""
        q1,q2,q3 = q; l1,l2,l3 = L
        s1,c1 = np.sin(q1),np.cos(q1)
        s12,c12 = np.sin(q1+q2),np.cos(q1+q2)
        s123,c123 = np.sin(q1+q2+q3),np.cos(q1+q2+q3)
        return np.array([
            [-l1*s1-l2*s12-l3*s123, -l2*s12-l3*s123, -l3*s123],
            [ l1*c1+l2*c12+l3*c123,  l2*c12+l3*c123,  l3*c123],
        ])

    def jac_full(q, L):
        """3×3 full Jacobian [J_v; J_ω] for planar 3R."""
        return np.vstack([jac_v(q, L), np.ones((1, 3))])

    def nullspace_projector(Jo):
        """3×3 nullspace projector of J_ω (1×3)."""
        Jo_pinv = Jo.T @ np.linalg.inv(Jo @ Jo.T)
        return np.eye(3) - Jo_pinv @ Jo

    def manip_measures(q, L):
        Jv = jac_v(q, L)
        Jo = np.ones((1, 3))
        N  = nullspace_projector(Jo)
        Jvs = Jv @ N
        M_full   = jac_full(q,L) @ jac_full(q,L).T   # 3×3
        M_weak   = Jv @ Jv.T                           # 2×2
        M_strong = Jvs @ Jvs.T                         # 2×2
        w_y = float(np.sqrt(max(np.linalg.det(M_weak), 0)))
        w_s = float(np.sqrt(max(np.linalg.det(M_strong), 0)))
        return M_full, M_weak, M_strong, w_y, w_s

    # ── SMM tracing (same as main app) ───────────────────────────────────────

    def project_q(q, target, L, max_iters=30, tol=1e-11, damp=1e-10):
        q = np.array(q, dtype=float)
        for _ in range(max_iters):
            err = fk_xy(q, L) - target
            if np.linalg.norm(err) < tol:
                return q, True
            J = jac_v(q, L)
            A = J @ J.T + damp*np.eye(2)
            q -= J.T @ np.linalg.solve(A, err)
        return q, np.linalg.norm(fk_xy(q,L)-target) < 1e-8

    def nullspace_tangent(q, L, prev=None):
        _, _, vh = np.linalg.svd(jac_v(q, L))
        t = vh[-1] / np.linalg.norm(vh[-1])
        if prev is not None and np.dot(t, prev) < 0:
            t = -t
        return t

    def torus_dist(a, b):
        return float(np.linalg.norm(wrap_to_pi(np.array(a)-np.array(b))))

    def seed_from_phi(L, target, phi, elbow=1):
        l1,l2,l3 = L; x,y = target
        wx = x - l3*np.cos(phi); wy = y - l3*np.sin(phi)
        c2 = (wx**2+wy**2-l1**2-l2**2)/(2*l1*l2)
        if abs(c2) > 1: raise RuntimeError("No seed")
        s2 = (1 if elbow>=0 else -1)*np.sqrt(max(0,1-c2**2))
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(wy,wx) - np.arctan2(l2*s2, l1+l2*c2)
        return np.array([q1, q2, phi-q1-q2])

    def find_seed(L, target, branch=1):
        for phi in np.linspace(-np.pi, np.pi, 361):
            l1,l2,l3 = L; x,y = target
            wx=x-l3*np.cos(phi); wy=y-l3*np.sin(phi)
            c2=(wx**2+wy**2-l1**2-l2**2)/(2*l1*l2)
            if -1<=c2<=1:
                try:
                    return seed_from_phi(L, target, phi, elbow=branch), phi
                except: continue
        raise RuntimeError("No valid seed")

    def trace_smm(L, target, branch=1, ds=0.03,
                  max_steps=4000, min_steps=120, closure_tol=0.07):
        q0, _ = find_seed(L, target, branch)
        q, ok = project_q(q0, target, L)
        if not ok: raise RuntimeError("Projection failed")
        q_start = q.copy()
        t_start = nullspace_tangent(q, L)
        t = t_start.copy()
        qs = [q.copy()]
        for k in range(max_steps):
            step = ds
            for _ in range(10):
                qp, ok = project_q(q+step*t, target, L)
                if ok and torus_dist(qp,q) > 1e-5: break
                step *= 0.5
            if not ok: raise RuntimeError("Continuation failed")
            t = nullspace_tangent(qp, L, prev=t)
            qs.append(qp.copy()); q = qp
            if k > min_steps and torus_dist(q,q_start)<closure_tol and np.dot(t,t_start)>0.7:
                qs.append(q_start.copy()); break
        return np.array(qs)

    def arc_length(qs):
        s = np.zeros(len(qs))
        for i in range(1,len(qs)):
            s[i] = s[i-1]+torus_dist(qs[i],qs[i-1])
        return s

    # ── Ellipse helpers (Plotly) ──────────────────────────────────────────────

    def ellipse_xy(M2, cx, cy, scale=1.0, n=200):
        """(x,y) arrays for the ellipse x^T M^{-1} x = 1 centred at (cx,cy)."""
        vals, vecs = np.linalg.eigh(M2)
        vals = np.maximum(vals, 1e-14)
        t = np.linspace(0, 2*np.pi, n)
        pts = vecs @ np.diag(np.sqrt(vals)*scale) @ np.array([np.cos(t), np.sin(t)])
        return cx+pts[0], cy+pts[1]

    def ellipsoid_surface(M3, n_u=30, n_v=60):
        """(X,Y,Z) mesh for the 3D ellipsoid x^T M^{-1} x = 1."""
        vals, vecs = np.linalg.eigh(M3)
        vals = np.maximum(vals, 1e-14)
        u = np.linspace(0, np.pi, n_u)
        v = np.linspace(0, 2*np.pi, n_v)
        U, V = np.meshgrid(u, v, indexing='ij')
        sphere = np.stack([np.sin(U)*np.cos(V), np.sin(U)*np.sin(V), np.cos(U)], -1)
        A = vecs @ np.diag(np.sqrt(vals))
        pts = sphere @ A.T
        return pts[...,0], pts[...,1], pts[...,2]

    core = SimpleNamespace(
        fk_xy=fk_xy, fk_points=fk_points,
        jac_v=jac_v, jac_full=jac_full,
        manip_measures=manip_measures,
        trace_smm=trace_smm, arc_length=arc_length,
        ellipse_xy=ellipse_xy, ellipsoid_surface=ellipsoid_surface,
        torus_dist=torus_dist,
    )
    return core, go, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("# 3R Manipulability along the Self-Motion Manifold")


@app.cell
def _(mo):
    l1 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₁",show_value=True,include_input=True,full_width=True,debounce=True)
    l2 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₂",show_value=True,include_input=True,full_width=True,debounce=True)
    l3 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₃",show_value=True,include_input=True,full_width=True,debounce=True)
    tx = mo.ui.slider(-1.40,1.40,0.01,value=0.00,label="target x",show_value=True,include_input=True,full_width=True,debounce=True)
    ty = mo.ui.slider(-1.40,1.40,0.01,value=1.00,label="target y",show_value=True,include_input=True,full_width=True,debounce=True)
    ds = mo.ui.slider(0.01,0.08,0.005,value=0.03,label="ds",show_value=True,include_input=True,full_width=True,debounce=True)
    progress  = mo.ui.slider(0,100,1,value=0,label="Position on SMM [%]",show_value=True,include_input=True,full_width=True)
    ell_scale = mo.ui.slider(0.5,3.0,0.05,value=1.0,label="Ellipsoid Darstellungsgröße (×)",show_value=True,include_input=True,full_width=True)

    controls = mo.vstack([
        mo.hstack([l1,l2,l3], widths="equal"),
        mo.hstack([tx,ty,ds], widths="equal"),
        mo.hstack([progress, ell_scale], widths="equal"),
    ], gap=1)
    controls
    return ds, ell_scale, l1, l2, l3, progress, tx, ty


@app.cell
def _(core, ds, l1, l2, l3, np, tx, ty):
    _L = (float(l1.value), float(l2.value), float(l3.value))
    _target = (float(tx.value), float(ty.value))

    _result = {"ok": False, "msg": "", "qs": None, "s": None, "L": _L, "target": _target}
    try:
        _qs = core.trace_smm(_L, _target, ds=float(ds.value))
        _s  = core.arc_length(_qs)
        _s  = _s / _s[-1]  # normalise
        _result.update({"ok": True, "qs": _qs, "s": _s})
    except Exception as e:
        _result["msg"] = str(e)

    smm = _result
    return (smm,)


@app.cell
def _(core, np, progress, smm):
    if smm["ok"]:
        _qs = smm["qs"]
        _idx = int(round(float(progress.value)/100 * (len(_qs)-1)))
        _idx = int(np.clip(_idx, 0, len(_qs)-1))
        _q   = _qs[_idx]
        _M_full, _M_weak, _M_strong, _w_y, _w_s = core.manip_measures(_q, smm["L"])

        # compute along full manifold
        _w_y_all, _w_s_all = [], []
        for _qi in _qs:
            _, _, _, _wy_i, _ws_i = core.manip_measures(_qi, smm["L"])
            _w_y_all.append(_wy_i); _w_s_all.append(_ws_i)

        manip = {
            "ok": True,
            "idx": _idx, "q": _q,
            "M_full": _M_full, "M_weak": _M_weak, "M_strong": _M_strong,
            "w_y": _w_y, "w_s": _w_s,
            "w_y_all": np.array(_w_y_all),
            "w_s_all": np.array(_w_s_all),
        }
    else:
        manip = {"ok": False}
    return (manip,)


@app.cell
def _(core, ell_scale, go, manip, np, smm):
    if not smm["ok"] or not manip["ok"]:
        _fig_robot = go.Figure().update_layout(title="Error: " + smm.get("msg",""))
    else:
        _q   = manip["q"]
        _L   = smm["L"]
        _pts = core.fk_points(_q, _L)
        _ee  = _pts[-1]
        _total = sum(_L); _pad = 0.2

        # Normalize to 20% of arm length × user scale multiplier
        _vals_w, _ = np.linalg.eigh(manip["M_weak"])
        _max_ax_w  = float(np.sqrt(max(_vals_w.max(), 1e-14)))
        _sc = 0.20 * _total / _max_ax_w * float(ell_scale.value) if _max_ax_w > 1e-12 else 0.01

        # weak ellipse (grey)
        _fig_robot = go.Figure()
        _fig_robot.add_trace(go.Scatter(
            x=_pts[:,0], y=_pts[:,1], mode="lines+markers",
            line=dict(color="#1E293B", width=3),
            marker=dict(color="#1E293B", size=8),
            showlegend=False,
        ))
        _wx, _wy = core.ellipse_xy(manip["M_weak"], _ee[0], _ee[1], scale=_sc)
        _fig_robot.add_trace(go.Scatter(
            x=_wx, y=_wy, mode="lines", name="Weak translational",
            line=dict(color="#6B7280", width=2),
            fill="toself", fillcolor="rgba(107,114,128,0.10)",
        ))

        # strong ellipse (blue) – same scale as weak, so relative sizes match
        _sx, _sy = core.ellipse_xy(manip["M_strong"], _ee[0], _ee[1], scale=_sc)
        _fig_robot.add_trace(go.Scatter(
            x=_sx, y=_sy, mode="lines", name="Strong translational",
            line=dict(color="#1D6FD8", width=2),
            fill="toself", fillcolor="rgba(29,111,216,0.12)",
        ))

        # target
        _fig_robot.add_trace(go.Scatter(
            x=[smm["target"][0]], y=[smm["target"][1]],
            mode="markers", name="Target",
            marker=dict(symbol="x", size=14, color="#F59E0B", line_width=2),
        ))

        _fig_robot.update_layout(
            xaxis=dict(title="x [m]", range=[-_total-_pad, _total+_pad], scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y [m]", range=[-_total-_pad, _total+_pad]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        )

    fig_robot = _fig_robot
    return (fig_robot,)


@app.cell
def _(go, manip, smm):
    if not smm["ok"] or not manip["ok"]:
        _fig_w = go.Figure()
    else:
        _s = smm["s"]
        _fig_w = go.Figure()
        _fig_w.add_trace(go.Scatter(
            x=_s, y=manip["w_y_all"], mode="lines", name="w – Yoshikawa",
            line=dict(color="#DC2626", width=2), yaxis="y1",
        ))
        _fig_w.add_trace(go.Scatter(
            x=_s, y=manip["w_s_all"], mode="lines", name="w_s – strong",
            line=dict(color="#1D6FD8", width=2), yaxis="y2",
        ))
        _xi = _s[manip["idx"]]
        _fig_w.add_vline(x=float(_xi), line_dash="dash", line_color="#F59E0B", line_width=2)
        _fig_w.add_annotation(
            x=float(_xi), y=float(max(manip["w_y_all"])), yref="y1",
            text=f"w={manip['w_y']:.3f}<br>w_s={manip['w_s']:.3f}",
            showarrow=True, arrowhead=2, bgcolor="white", bordercolor="#F59E0B",
        )
        _fig_w.update_layout(
            xaxis=dict(title="Normierter SMM-Parameter s"),
            yaxis =dict(title="w (Yoshikawa)",  title_font=dict(color="#DC2626"), tickfont=dict(color="#DC2626")),
            yaxis2=dict(title="w_s (stark translational)", title_font=dict(color="#1D6FD8"), tickfont=dict(color="#1D6FD8"),
                        overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0,r=80,t=30,b=0),
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        )

    fig_w = _fig_w
    return (fig_w,)


@app.cell
def _(core, go, manip, smm):
    if not smm["ok"] or not manip["ok"]:
        _fig3d = go.Figure()
    else:
        # Scale ω row by L_char (mean link length) for dimensional consistency
        # so the 3D ellipsoid isn't dominated by the unit-less rotational DOF
        _L_char = float(np.mean(smm["L"]))
        _Jv   = core.jac_v(manip["q"], smm["L"])
        _Jo_s = _L_char * np.ones((1, 3))          # dimensionally scaled
        _Jf_s = np.vstack([_Jv, _Jo_s])
        _M3   = _Jf_s @ _Jf_s.T
        _X, _Y, _Z = core.ellipsoid_surface(_M3, n_u=30, n_v=60)

        # wireframe: meridians + parallels
        _traces = []
        for _i in range(0, _X.shape[0], 2):
            _traces.append(go.Scatter3d(
                x=_X[_i], y=_Y[_i], z=_Z[_i], mode="lines",
                line=dict(color="#555555", width=1), showlegend=False,
                hoverinfo="skip",
            ))
        for _j in range(0, _X.shape[1], 3):
            _traces.append(go.Scatter3d(
                x=_X[:,_j], y=_Y[:,_j], z=_Z[:,_j], mode="lines",
                line=dict(color="#555555", width=1), showlegend=False,
                hoverinfo="skip",
            ))

        # strong ellipse in ω=0 plane
        _ex, _ey = core.ellipse_xy(manip["M_strong"], 0, 0)
        import numpy as _np
        _traces.append(go.Scatter3d(
            x=_ex, y=_ey, z=_np.zeros_like(_ex), mode="lines",
            line=dict(color="#1D6FD8", width=3), name="Strong (ω=0)",
        ))

        _fig3d = go.Figure(data=_traces)
        _fig3d.update_layout(
            scene=dict(
                xaxis_title="ẋ [m/s]",
                yaxis_title="ẏ [m/s]",
                zaxis_title="ω [rad/s]",
                aspectmode="data",
                xaxis=dict(backgroundcolor="white", title_font=dict(size=12)),
                yaxis=dict(backgroundcolor="white", title_font=dict(size=12)),
                zaxis=dict(backgroundcolor="white", title_font=dict(size=12)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="white",
        )

    fig_3d = _fig3d
    return (fig_3d,)


@app.cell(hide_code=True)
def _(fig_3d, fig_robot, fig_w, mo, manip, smm):
    if not smm["ok"]:
        _layout = mo.callout(mo.md(f"**Error:** `{smm['msg']}`"), kind="danger")
    else:
        _q = manip["q"]
        _status = mo.md(
            f"`q = ({_q[0]:.3f}, {_q[1]:.3f}, {_q[2]:.3f})` &nbsp;|&nbsp; "
            f"`w = {manip['w_y']:.4f}` &nbsp;|&nbsp; "
            f"`w_s = {manip['w_s']:.4f}` &nbsp;|&nbsp; "
            f"`pts = {len(smm['qs'])}`"
        )
        _layout = mo.vstack([
            _status,
            mo.hstack([
                mo.ui.plotly(fig_robot),
                mo.ui.plotly(fig_3d),
            ], widths="equal"),
            mo.ui.plotly(fig_w),
        ], gap=1)
    _layout


if __name__ == "__main__":
    app.run()
