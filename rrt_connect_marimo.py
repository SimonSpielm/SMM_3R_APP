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

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from types import SimpleNamespace

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def make_scene():
        return {
            "bounds": (0.0, 10.0, 0.0, 7.0),
            "start": np.array([0.75, 0.75], dtype=float),
            "goal": np.array([9.25, 6.15], dtype=float),
            "rectangles": [
                (2.10, 1.00, 2.85, 4.15),
                (4.85, 2.70, 5.70, 6.25),
                (7.10, 0.70, 7.90, 3.35),
            ],
            "circles": [
                (3.75, 5.55, 0.52),
                (6.45, 1.45, 0.48),
            ],
        }

    def point_is_free(point, scene, *, clearance=0.06):
        x, y = np.asarray(point, dtype=float)
        xmin, xmax, ymin, ymax = scene["bounds"]
        if (
            x < xmin + clearance
            or x > xmax - clearance
            or y < ymin + clearance
            or y > ymax - clearance
        ):
            return False

        for rx0, ry0, rx1, ry1 in scene["rectangles"]:
            inside_x = rx0 - clearance <= x <= rx1 + clearance
            inside_y = ry0 - clearance <= y <= ry1 + clearance
            if inside_x and inside_y:
                return False

        for cx, cy, radius in scene["circles"]:
            if np.hypot(x - cx, y - cy) <= radius + clearance:
                return False

        return True

    def segment_is_free(a, b, scene, *, clearance=0.06, resolution=0.035):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        distance = float(np.linalg.norm(b - a))
        steps = max(2, int(np.ceil(distance / resolution)) + 1)

        for alpha in np.linspace(0.0, 1.0, steps):
            point = (1.0 - alpha) * a + alpha * b
            if not point_is_free(point, scene, clearance=clearance):
                return False

        return True

    def sample_free(rng, scene):
        xmin, xmax, ymin, ymax = scene["bounds"]
        for _ in range(300):
            sample = np.array(
                [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)],
                dtype=float,
            )
            if point_is_free(sample, scene):
                return sample
        raise RuntimeError("Could not sample a collision-free point.")

    def nearest_index(nodes, point):
        node_array = np.vstack(nodes)
        distances = np.linalg.norm(node_array - point, axis=1)
        return int(np.argmin(distances))

    def steer(q_from, q_to, step_size):
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        delta = q_to - q_from
        distance = float(np.linalg.norm(delta))

        if distance <= 1e-12:
            return q_from.copy()
        if distance <= step_size:
            return q_to.copy()

        return q_from + step_size * delta / distance

    def extend_tree(tree, target, scene, *, step_size, clearance, resolution):
        nearest = nearest_index(tree["nodes"], target)
        q_near = tree["nodes"][nearest]

        if np.linalg.norm(q_near - target) <= 1e-9:
            return "reached", nearest

        q_new = steer(q_near, target, step_size)
        if np.linalg.norm(q_new - q_near) <= 1e-12:
            return "trapped", None

        if not segment_is_free(
            q_near,
            q_new,
            scene,
            clearance=clearance,
            resolution=resolution,
        ):
            return "trapped", None

        tree["nodes"].append(q_new)
        tree["parents"].append(nearest)
        status = "reached" if np.linalg.norm(q_new - target) <= 1e-9 else "advanced"
        return status, len(tree["nodes"]) - 1

    def trace_path(tree, node_index):
        path = []
        while node_index >= 0:
            path.append(tree["nodes"][node_index])
            node_index = tree["parents"][node_index]
        return np.array(path[::-1])

    def combine_path(grow_tree, grow_idx, other_tree, other_idx):
        if grow_tree["label"] == "start":
            start_path = trace_path(grow_tree, grow_idx)
            goal_path = trace_path(other_tree, other_idx)
        else:
            start_path = trace_path(other_tree, other_idx)
            goal_path = trace_path(grow_tree, grow_idx)

        if len(goal_path) <= 1:
            return start_path
        return np.vstack([start_path, goal_path[-2::-1]])

    def copy_tree(tree):
        return {
            "label": tree["label"],
            "nodes": [node.copy() for node in tree["nodes"]],
            "parents": list(tree["parents"]),
        }

    def min_tree_distance(tree_a, tree_b):
        a = np.vstack(tree_a["nodes"])
        b = np.vstack(tree_b["nodes"])
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return float(np.min(distances))

    def make_snapshot(iteration, start_tree, goal_tree, sample=None, event=""):
        return {
            "iteration": int(iteration),
            "start_tree": copy_tree(start_tree),
            "goal_tree": copy_tree(goal_tree),
            "sample": None if sample is None else np.asarray(sample, dtype=float).copy(),
            "event": event,
        }

    def run_rrt_connect(
        *,
        seed=7,
        max_iterations=180,
        step_size=0.45,
        goal_bias=0.05,
        clearance=0.06,
        resolution=0.035,
    ):
        scene = make_scene()
        rng = np.random.default_rng(int(seed))
        start_tree = {"label": "start", "nodes": [scene["start"].copy()], "parents": [-1]}
        goal_tree = {"label": "goal", "nodes": [scene["goal"].copy()], "parents": [-1]}

        metrics = {
            "iteration": [0],
            "start_nodes": [1],
            "goal_nodes": [1],
            "min_distance": [min_tree_distance(start_tree, goal_tree)],
        }
        snapshots = [make_snapshot(0, start_tree, goal_tree, event="initial")]
        snapshot_targets = {
            1,
            max(2, max_iterations // 10),
            max(3, max_iterations // 4),
            max(4, max_iterations // 2),
            max_iterations,
        }

        path = None
        connected_iteration = None
        event = "not connected"

        for iteration in range(1, int(max_iterations) + 1):
            grow_tree = start_tree if iteration % 2 else goal_tree
            other_tree = goal_tree if grow_tree is start_tree else start_tree

            if rng.random() < goal_bias:
                sample = scene["goal"] if grow_tree is start_tree else scene["start"]
            else:
                sample = sample_free(rng, scene)

            status, grow_idx = extend_tree(
                grow_tree,
                sample,
                scene,
                step_size=step_size,
                clearance=clearance,
                resolution=resolution,
            )

            if status == "trapped":
                event = f"{grow_tree['label']} trapped"
            else:
                q_new = grow_tree["nodes"][grow_idx]
                connect_steps = 0

                while True:
                    other_status, other_idx = extend_tree(
                        other_tree,
                        q_new,
                        scene,
                        step_size=step_size,
                        clearance=clearance,
                        resolution=resolution,
                    )

                    if other_status == "trapped":
                        event = f"{grow_tree['label']} advanced"
                        break

                    connect_steps += 1
                    if other_status == "reached":
                        path = combine_path(grow_tree, grow_idx, other_tree, other_idx)
                        connected_iteration = iteration
                        event = f"connected after {connect_steps} connect steps"
                        break

                if path is not None:
                    metrics["iteration"].append(iteration)
                    metrics["start_nodes"].append(len(start_tree["nodes"]))
                    metrics["goal_nodes"].append(len(goal_tree["nodes"]))
                    metrics["min_distance"].append(min_tree_distance(start_tree, goal_tree))
                    snapshots.append(make_snapshot(iteration, start_tree, goal_tree, sample, event))
                    break

            metrics["iteration"].append(iteration)
            metrics["start_nodes"].append(len(start_tree["nodes"]))
            metrics["goal_nodes"].append(len(goal_tree["nodes"]))
            metrics["min_distance"].append(min_tree_distance(start_tree, goal_tree))

            if iteration in snapshot_targets:
                snapshots.append(make_snapshot(iteration, start_tree, goal_tree, sample, event))

        if snapshots[-1]["iteration"] != metrics["iteration"][-1]:
            snapshots.append(make_snapshot(metrics["iteration"][-1], start_tree, goal_tree, event=event))

        path_length = 0.0
        if path is not None and len(path) > 1:
            path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

        return {
            "scene": scene,
            "start_tree": start_tree,
            "goal_tree": goal_tree,
            "metrics": metrics,
            "snapshots": snapshots,
            "path": path,
            "path_length": path_length,
            "connected": path is not None,
            "connected_iteration": connected_iteration,
            "final_iteration": int(metrics["iteration"][-1]),
        }

    def add_trace(fig, trace, row=None, col=None):
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    def obstacle_traces(scene, *, showlegend=False):
        traces = []

        for i, (x0, y0, x1, y1) in enumerate(scene["rectangles"]):
            traces.append(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(150, 150, 160, 0.42)",
                    line=dict(color="rgba(220, 220, 230, 0.85)", width=1),
                    name="Obstacle",
                    legendgroup="obstacles",
                    showlegend=showlegend and i == 0,
                    hoverinfo="skip",
                )
            )

        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        for i, (cx, cy, radius) in enumerate(scene["circles"]):
            traces.append(
                go.Scatter(
                    x=cx + radius * np.cos(theta),
                    y=cy + radius * np.sin(theta),
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(150, 150, 160, 0.42)",
                    line=dict(color="rgba(220, 220, 230, 0.85)", width=1),
                    name="Obstacle",
                    legendgroup="obstacles",
                    showlegend=showlegend and not scene["rectangles"] and i == 0,
                    hoverinfo="skip",
                )
            )

        return traces

    def tree_edge_trace(tree, *, color, name, showlegend=True):
        xs = []
        ys = []
        for child_idx, parent_idx in enumerate(tree["parents"]):
            if parent_idx < 0:
                continue
            parent = tree["nodes"][parent_idx]
            child = tree["nodes"][child_idx]
            xs.extend([parent[0], child[0], None])
            ys.extend([parent[1], child[1], None])

        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=1.25),
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            hoverinfo="skip",
        )

    def tree_node_trace(tree, *, color, name):
        nodes = np.vstack(tree["nodes"])
        return go.Scatter(
            x=nodes[:, 0],
            y=nodes[:, 1],
            mode="markers",
            marker=dict(color=color, size=3.5),
            name=name,
            legendgroup=name,
            showlegend=False,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )

    def add_scene(fig, scene, *, row=None, col=None, showlegend=False):
        for trace in obstacle_traces(scene, showlegend=showlegend):
            add_trace(fig, trace, row, col)

        xmin, xmax, ymin, ymax = scene["bounds"]
        add_trace(
            fig,
            go.Scatter(
                x=[xmin, xmax, xmax, xmin, xmin],
                y=[ymin, ymin, ymax, ymax, ymin],
                mode="lines",
                line=dict(color="rgba(245, 245, 245, 0.55)", width=1),
                name="World boundary",
                showlegend=False,
                hoverinfo="skip",
            ),
            row,
            col,
        )

    def add_trees(fig, start_tree, goal_tree, *, row=None, col=None, showlegend=True):
        add_trace(
            fig,
            tree_edge_trace(start_tree, color="rgba(76, 201, 240, 0.44)", name="Start tree", showlegend=showlegend),
            row,
            col,
        )
        add_trace(
            fig,
            tree_edge_trace(goal_tree, color="rgba(247, 37, 133, 0.44)", name="Goal tree", showlegend=showlegend),
            row,
            col,
        )
        add_trace(fig, tree_node_trace(start_tree, color="rgba(76, 201, 240, 0.78)", name="Start tree"), row, col)
        add_trace(fig, tree_node_trace(goal_tree, color="rgba(247, 37, 133, 0.78)", name="Goal tree"), row, col)

    def add_start_goal(fig, scene, *, row=None, col=None, showlegend=True):
        add_trace(
            fig,
            go.Scatter(
                x=[scene["start"][0]],
                y=[scene["start"][1]],
                mode="markers",
                marker=dict(symbol="circle", size=12, color="#4CC9F0", line=dict(width=1.5, color="white")),
                name="Start",
                showlegend=showlegend,
                hovertemplate="Start<extra></extra>",
            ),
            row,
            col,
        )
        add_trace(
            fig,
            go.Scatter(
                x=[scene["goal"][0]],
                y=[scene["goal"][1]],
                mode="markers",
                marker=dict(symbol="star", size=14, color="#F72585", line=dict(width=1.5, color="white")),
                name="Goal",
                showlegend=showlegend,
                hovertemplate="Goal<extra></extra>",
            ),
            row,
            col,
        )

    def apply_world_axes(fig, scene, *, rows=1, cols=1):
        xmin, xmax, ymin, ymax = scene["bounds"]
        fig.update_xaxes(range=[xmin - 0.15, xmax + 0.15], title="x")
        fig.update_yaxes(range=[ymin - 0.15, ymax + 0.15], title="y")
        if rows == 1 and cols == 1:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        )
        return fig

    def make_final_figure(result):
        scene = result["scene"]
        fig = go.Figure()
        add_scene(fig, scene, showlegend=True)
        add_trees(fig, result["start_tree"], result["goal_tree"], showlegend=True)

        if result["path"] is not None:
            path = result["path"]
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines+markers",
                    line=dict(color="#FFD166", width=4),
                    marker=dict(color="#FFD166", size=5),
                    name="Found path",
                    hovertemplate="path point<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                )
            )

        add_start_goal(fig, scene, showlegend=True)
        if result["connected"]:
            title = f"RRT-Connect final tree - connected at iteration {result['connected_iteration']}"
        else:
            title = f"RRT-Connect final tree - no connection after {result['final_iteration']} iterations"
        fig.update_layout(title=title)
        return apply_world_axes(fig, scene)

    def select_snapshots(snapshots, count=4):
        if len(snapshots) <= count:
            return list(snapshots)

        indices = np.linspace(0, len(snapshots) - 1, count).round().astype(int)
        selected = []
        seen = set()
        for idx in indices:
            snap = snapshots[int(idx)]
            if snap["iteration"] not in seen:
                selected.append(snap)
                seen.add(snap["iteration"])

        if selected[-1]["iteration"] != snapshots[-1]["iteration"]:
            selected[-1] = snapshots[-1]
        return selected

    def make_timeline_figure(result):
        scene = result["scene"]
        snapshots = select_snapshots(result["snapshots"], count=4)
        while len(snapshots) < 4:
            snapshots.append(snapshots[-1])

        titles = [f"iter {snap['iteration']}: {snap['event'] or 'growth'}" for snap in snapshots[:4]]
        fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

        for i, snap in enumerate(snapshots[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            add_scene(fig, scene, row=row, col=col, showlegend=i == 0)
            add_trees(fig, snap["start_tree"], snap["goal_tree"], row=row, col=col, showlegend=i == 0)

            if snap["sample"] is not None:
                add_trace(
                    fig,
                    go.Scatter(
                        x=[snap["sample"][0]],
                        y=[snap["sample"][1]],
                        mode="markers",
                        marker=dict(symbol="x", size=9, color="#F4D35E"),
                        name="Last sample",
                        showlegend=i == 0,
                        hovertemplate="sample<extra></extra>",
                    ),
                    row,
                    col,
                )
            add_start_goal(fig, scene, row=row, col=col, showlegend=i == 0)

        fig.update_layout(title="RRT-Connect growth snapshots", height=700)
        return apply_world_axes(fig, scene, rows=2, cols=2)

    def make_metrics_figure(result):
        metrics = result["metrics"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=metrics["iteration"],
                y=metrics["start_nodes"],
                mode="lines+markers",
                name="Start tree nodes",
                line=dict(color="#4CC9F0", width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["iteration"],
                y=metrics["goal_nodes"],
                mode="lines+markers",
                name="Goal tree nodes",
                line=dict(color="#F72585", width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["iteration"],
                y=metrics["min_distance"],
                mode="lines",
                name="Closest tree distance",
                line=dict(color="#FFD166", width=3),
            ),
            secondary_y=True,
        )
        fig.update_xaxes(title="Iteration")
        fig.update_yaxes(title="Tree nodes", secondary_y=False)
        fig.update_yaxes(title="Closest distance", secondary_y=True)
        fig.update_layout(
            title="How the two trees approach each other",
            template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        )
        return fig

    rrt = SimpleNamespace(
        run_rrt_connect=run_rrt_connect,
        make_final_figure=make_final_figure,
        make_timeline_figure=make_timeline_figure,
        make_metrics_figure=make_metrics_figure,
    )
    return (rrt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # RRT-Connect
    """)
    return


@app.cell
def _(mo):
    rrt_seed = mo.ui.slider(
        start=1,
        stop=99,
        step=1,
        value=7,
        show_value=True,
        include_input=True,
        label="random seed",
        full_width=True,
    )
    rrt_iterations = mo.ui.slider(
        start=20,
        stop=400,
        step=10,
        value=180,
        debounce=True,
        show_value=True,
        include_input=True,
        label="max iterations",
        full_width=True,
    )
    rrt_step_size = mo.ui.slider(
        start=0.20,
        stop=0.90,
        step=0.05,
        value=0.45,
        debounce=True,
        show_value=True,
        include_input=True,
        label="step size",
        full_width=True,
    )
    rrt_goal_bias = mo.ui.slider(
        start=0.00,
        stop=0.20,
        step=0.01,
        value=0.05,
        debounce=True,
        show_value=True,
        include_input=True,
        label="goal bias",
        full_width=True,
    )

    controls = mo.vstack(
        [
            mo.hstack([rrt_seed, rrt_iterations], widths="equal"),
            mo.hstack([rrt_step_size, rrt_goal_bias], widths="equal"),
        ],
        gap=1,
    )
    controls
    return rrt_goal_bias, rrt_iterations, rrt_seed, rrt_step_size


@app.cell
def _(rrt, rrt_goal_bias, rrt_iterations, rrt_seed, rrt_step_size):
    rrt_result = rrt.run_rrt_connect(
        seed=int(rrt_seed.value),
        max_iterations=int(rrt_iterations.value),
        step_size=float(rrt_step_size.value),
        goal_bias=float(rrt_goal_bias.value),
    )
    return (rrt_result,)


@app.cell(hide_code=True)
def _(mo, rrt_result):
    if rrt_result["connected"]:
        rrt_status = mo.md(
            f"`connected`  |  `iteration {rrt_result['connected_iteration']}`  |  `nodes {len(rrt_result['start_tree']['nodes']) + len(rrt_result['goal_tree']['nodes'])}`  |  `path length {rrt_result['path_length']:.2f}`"
        )
    else:
        rrt_status = mo.md(
            f"`not connected`  |  `iterations {rrt_result['final_iteration']}`  |  `nodes {len(rrt_result['start_tree']['nodes']) + len(rrt_result['goal_tree']['nodes'])}`"
        ).callout(kind="warn")

    rrt_status
    return (rrt_status,)


@app.cell
def _(mo, rrt, rrt_result):
    rrt_final_plot = mo.ui.plotly(rrt.make_final_figure(rrt_result))
    rrt_timeline_plot = mo.ui.plotly(rrt.make_timeline_figure(rrt_result))
    rrt_metrics_plot = mo.ui.plotly(rrt.make_metrics_figure(rrt_result))
    return rrt_final_plot, rrt_metrics_plot, rrt_timeline_plot


@app.cell(hide_code=True)
def _(mo, rrt_final_plot, rrt_metrics_plot, rrt_status, rrt_timeline_plot):
    mo.vstack(
        [
            rrt_status,
            rrt_final_plot,
            rrt_timeline_plot,
            rrt_metrics_plot,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
