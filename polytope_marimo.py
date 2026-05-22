# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.19.11",
#   "numpy>=1.26",
#   "plotly>=5.24",
#   "scipy>=1.11.0",
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
    from scipy.spatial import ConvexHull
    import itertools
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
        q1,q2,q3 = q; l1,l2,l3 = L
        s1,c1 = np.sin(q1),np.cos(q1)
        s12,c12 = np.sin(q1+q2),np.cos(q1+q2)
        s123,c123 = np.sin(q1+q2+q3),np.cos(q1+q2+q3)
        return np.array([
            [-l1*s1-l2*s12-l3*s123, -l2*s12-l3*s123, -l3*s123],
            [ l1*c1+l2*c12+l3*c123,  l2*c12+l3*c123,  l3*c123],
        ])

    def jac_full(q, L):
        return np.vstack([jac_v(q, L), np.ones((1, 3))])

    # ── Polytope Helpers ──────────────────────────────────────────────────────
    
    def slice_polygon(v_bounds):
        v1, v2, v3 = v_bounds
        verts = []
        for y in [-v2, v2]:
            for z in [-v3, v3]:
                x = -(y + z)
                if -v1 <= x <= v1:
                    verts.append([x, y, z])
        for x in [-v1, v1]:
            for z in [-v3, v3]:
                y = -(x + z)
                if -v2 <= y <= v2:
                    verts.append([x, y, z])
        for x in [-v1, v1]:
            for y in [-v2, v2]:
                z = -(x + y)
                if -v3 <= z <= v3:
                    verts.append([x, y, z])
        if len(verts) == 0:
            return np.zeros((0, 3))
        verts = np.unique(np.round(verts, 6), axis=0)
        # Sort circularly
        cx, cy = np.mean(verts[:, 0]), np.mean(verts[:, 1])
        angles = np.arctan2(verts[:, 1] - cy, verts[:, 0] - cx)
        return verts[np.argsort(angles)]

    def polytope_measures(q, L, v_bounds):
        Jv = jac_v(q, L)
        v1, v2, v3 = v_bounds
        box_verts = np.array(list(itertools.product([-v1, v1], [-v2, v2], [-v3, v3])))
        ws_box = (Jv @ box_verts.T).T
        
        try:
            hull_weak = ConvexHull(ws_box)
            area_weak = hull_weak.volume
            bnd_weak = ws_box[hull_weak.vertices]
        except:
            area_weak = 0.0
            bnd_weak = np.zeros((0, 2))
            
        slice_verts = slice_polygon(v_bounds)
        if len(slice_verts) >= 3:
            ws_slice = (Jv @ slice_verts.T).T
            try:
                hull_strong = ConvexHull(ws_slice)
                area_strong = hull_strong.volume
                bnd_strong = ws_slice[hull_strong.vertices]
            except:
                area_strong = 0.0
                bnd_strong = np.zeros((0, 2))
        else:
            area_strong = 0.0
            bnd_strong = np.zeros((0, 2))
            
        return area_weak, area_strong, bnd_weak, bnd_strong, box_verts, slice_verts

    # ── SMM tracing (same as manipulability) ─────────────────────────────────

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

    core = SimpleNamespace(
        fk_xy=fk_xy, fk_points=fk_points,
        jac_v=jac_v, jac_full=jac_full,
        polytope_measures=polytope_measures, slice_polygon=slice_polygon,
        trace_smm=trace_smm, arc_length=arc_length,
        torus_dist=torus_dist,
    )
    return ConvexHull, core, go, itertools, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("# 3R Velocity Polytopes along the Self-Motion Manifold")
    return


@app.cell
def _(mo):
    l1 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₁",show_value=True,include_input=True,full_width=True,debounce=True)
    l2 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₂",show_value=True,include_input=True,full_width=True,debounce=True)
    l3 = mo.ui.slider(0.20,1.20,0.01,value=0.40,label="l₃",show_value=True,include_input=True,full_width=True,debounce=True)
    
    v1 = mo.ui.slider(0.10,3.00,0.10,value=1.00,label="v_max₁ [rad/s]",show_value=True,include_input=True,full_width=True,debounce=True)
    v2 = mo.ui.slider(0.10,3.00,0.10,value=1.00,label="v_max₂ [rad/s]",show_value=True,include_input=True,full_width=True,debounce=True)
    v3 = mo.ui.slider(0.10,3.00,0.10,value=1.00,label="v_max₃ [rad/s]",show_value=True,include_input=True,full_width=True,debounce=True)
    
    tx = mo.ui.slider(-1.40,1.40,0.01,value=0.00,label="target x",show_value=True,include_input=True,full_width=True,debounce=True)
    ty = mo.ui.slider(-1.40,1.40,0.01,value=1.00,label="target y",show_value=True,include_input=True,full_width=True,debounce=True)
    ds = mo.ui.slider(0.01,0.08,0.005,value=0.03,label="ds",show_value=True,include_input=True,full_width=True,debounce=True)
    
    progress  = mo.ui.slider(0,100,1,value=0,label="Position on SMM [%]",show_value=True,include_input=True,full_width=True)
    poly_scale = mo.ui.slider(0.1,2.0,0.05,value=0.55,label="Polytope Darstellungsgröße (×)",show_value=True,include_input=True,full_width=True)

    controls = mo.vstack([
        mo.hstack([l1,l2,l3], widths="equal"),
        mo.hstack([v1,v2,v3], widths="equal"),
        mo.hstack([tx,ty,ds], widths="equal"),
        mo.hstack([progress, poly_scale], widths="equal"),
    ], gap=1)
    
    return controls, ds, poly_scale, l1, l2, l3, v1, v2, v3, progress, tx, ty


@app.cell
def _(core, ds, l1, l2, l3, tx, ty):
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
def _(core, np, progress, smm, v1, v2, v3):
    _v_bounds = (float(v1.value), float(v2.value), float(v3.value))
    if smm["ok"]:
        _qs = smm["qs"]
        _idx = int(round(float(progress.value)/100 * (len(_qs)-1)))
        _idx = int(np.clip(_idx, 0, len(_qs)-1))
        _q   = _qs[_idx]
        
        _area_weak, _area_strong, _bnd_weak, _bnd_strong, _box, _slice = core.polytope_measures(_q, smm["L"], _v_bounds)

        # compute along full manifold
        _aw_all, _as_all = [], []
        for _qi in _qs:
            _aw, _as, _, _, _, _ = core.polytope_measures(_qi, smm["L"], _v_bounds)
            _aw_all.append(_aw); _as_all.append(_as)

        manip = {
            "ok": True,
            "idx": _idx, "q": _q, "v_bounds": _v_bounds,
            "area_weak": _area_weak, "area_strong": _area_strong,
            "bnd_weak": _bnd_weak, "bnd_strong": _bnd_strong,
            "box": _box, "slice": _slice,
            "aw_all": np.array(_aw_all),
            "as_all": np.array(_as_all),
        }
    else:
        manip = {"ok": False}
    return (manip,)


@app.cell
def _(poly_scale, go, manip, np, smm, core):
    if not smm["ok"] or not manip["ok"]:
        _fig_robot = go.Figure().update_layout(title="Error: " + smm.get("msg",""))
    else:
        _q   = manip["q"]
        _L   = smm["L"]
        _pts = core.fk_points(_q, _L)
        _ee  = _pts[-1]
        _total = sum(_L); _pad = 0.2

        _sc = float(poly_scale.value)

        _fig_robot = go.Figure()
        
        # Robot arm
        _fig_robot.add_trace(go.Scatter(
            x=_pts[:,0], y=_pts[:,1], mode="lines+markers",
            line=dict(color="#1E293B", width=3),
            marker=dict(color="#1E293B", size=8),
            showlegend=False,
        ))
        
        # Weak polytope (grey)
        if len(manip["bnd_weak"]) > 0:
            _bnd = np.vstack((manip["bnd_weak"], manip["bnd_weak"][0])) * _sc + _ee
            _fig_robot.add_trace(go.Scatter(
                x=_bnd[:,0], y=_bnd[:,1], mode="lines", name="Full Polytope",
                line=dict(color="#6B7280", width=2),
                fill="toself", fillcolor="rgba(107,114,128,0.10)",
            ))

        # Strong polytope (blue)
        if len(manip["bnd_strong"]) > 0:
            _bnd_s = np.vstack((manip["bnd_strong"], manip["bnd_strong"][0])) * _sc + _ee
            _fig_robot.add_trace(go.Scatter(
                x=_bnd_s[:,0], y=_bnd_s[:,1], mode="lines", name="Constrained (ω=0)",
                line=dict(color="#1D6FD8", width=2),
                fill="toself", fillcolor="rgba(29,111,216,0.20)",
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
            x=_s, y=manip["aw_all"], mode="lines", name="Area (Full)",
            line=dict(color="#DC2626", width=2), yaxis="y1",
        ))
        _fig_w.add_trace(go.Scatter(
            x=_s, y=manip["as_all"], mode="lines", name="Area (Constrained)",
            line=dict(color="#1D6FD8", width=2), yaxis="y2",
        ))
        _xi = _s[manip["idx"]]
        _fig_w.add_vline(x=float(_xi), line_dash="dash", line_color="#F59E0B", line_width=2)
        _fig_w.add_annotation(
            x=float(_xi), y=float(max(manip["aw_all"])), yref="y1",
            text=f"A={manip['area_weak']:.3f}<br>A_s={manip['area_strong']:.3f}",
            showarrow=True, arrowhead=2, bgcolor="white", bordercolor="#F59E0B",
        )
        _fig_w.update_layout(
            xaxis=dict(title="Normierter SMM-Parameter s"),
            yaxis =dict(title="Area (Full)",  title_font=dict(color="#DC2626"), tickfont=dict(color="#DC2626")),
            yaxis2=dict(title="Area (Constrained)", title_font=dict(color="#1D6FD8"), tickfont=dict(color="#1D6FD8"),
                        overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0,r=80,t=30,b=0),
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        )

    fig_w = _fig_w
    return (fig_w,)


@app.cell
def _(go, manip, np, smm):
    if not smm["ok"] or not manip["ok"]:
        _fig3d = go.Figure()
    else:
        _v = manip["v_bounds"]
        _box = manip["box"]
        _slice = manip["slice"]

        _traces = []
        
        # Draw box edges
        for _i in range(8):
            for _j in range(_i+1, 8):
                if np.sum(np.abs(_box[_i] - _box[_j]) > 1e-6) == 1:
                    _traces.append(go.Scatter3d(
                        x=[_box[_i,0], _box[_j,0]], y=[_box[_i,1], _box[_j,1]], z=[_box[_i,2], _box[_j,2]],
                        mode="lines", line=dict(color="#555555", width=2), showlegend=False, hoverinfo="skip"
                    ))
                    
        # Draw slice
        if len(_slice) >= 3:
            _sl = np.vstack((_slice, _slice[0]))
            _traces.append(go.Scatter3d(
                x=_sl[:,0], y=_sl[:,1], z=_sl[:,2],
                mode="lines", line=dict(color="#1D6FD8", width=4), name="Constrained (ω=0)",
            ))
            _i, _j, _k = [], [], []
            for _idx in range(1, len(_slice)-1):
                _i.append(0)
                _j.append(_idx)
                _k.append(_idx+1)
            _traces.append(go.Mesh3d(
                x=_slice[:,0], y=_slice[:,1], z=_slice[:,2],
                i=_i, j=_j, k=_k,
                color="#1D6FD8", opacity=0.20, showlegend=False, hoverinfo="skip"
            ))

        _fig3d = go.Figure(data=_traces)
        _fig3d.update_layout(
            scene=dict(
                xaxis_title="q̇₁ [rad/s]",
                yaxis_title="q̇₂ [rad/s]",
                zaxis_title="q̇₃ [rad/s]",
                xaxis=dict(backgroundcolor="white", title_font=dict(size=12), range=[-_v[0]*1.2, _v[0]*1.2]),
                yaxis=dict(backgroundcolor="white", title_font=dict(size=12), range=[-_v[1]*1.2, _v[1]*1.2]),
                zaxis=dict(backgroundcolor="white", title_font=dict(size=12), range=[-_v[2]*1.2, _v[2]*1.2]),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="white",
        )

    fig_3d = _fig3d
    return (fig_3d,)


@app.cell(hide_code=True)
def _(fig_3d, fig_robot, fig_w, mo, manip, smm, controls):
    if not smm["ok"]:
        _layout = mo.callout(mo.md(f"**Error:** `{smm['msg']}`"), kind="danger")
    else:
        _q = manip["q"]
        _status = mo.md(
            f"`q = ({_q[0]:.3f}, {_q[1]:.3f}, {_q[2]:.3f})` &nbsp;|&nbsp; "
            f"`Area = {manip['area_weak']:.4f}` &nbsp;|&nbsp; "
            f"`Area_s = {manip['area_strong']:.4f}` &nbsp;|&nbsp; "
            f"`pts = {len(smm['qs'])}`"
        )
        _layout = mo.vstack([
            controls,
            _status,
            mo.hstack([
                mo.ui.plotly(fig_robot),
                mo.ui.plotly(fig_3d),
            ], widths="equal"),
            mo.ui.plotly(fig_w),
        ], gap=1)
    _layout
    return


if __name__ == "__main__":
    app.run()
