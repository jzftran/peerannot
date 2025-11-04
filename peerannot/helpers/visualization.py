from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_global_params(model) -> dict[str, Any]:
    """Safely fetches global parameters (`rho`, `pi`, `T`) from the model.

    Args:
        model: An object containing the parameters `rho`, `pi`, and `T`
                as attributes.
               If an attribute is missing, it will be set to `None`
               in the output.

    Returns:
        dict[str, Any]: A dictionary with keys `rho`, `pi`, and `T`, and their
                        corresponding values from the model (or `None` if
                        the attribute is missing).

    Example:
        >>> class MockModel:
        ...     rho = 0.5
        ...     pi = 0.3
        >>> get_global_params(MockModel())
        {`rho`: 0.5, `pi`: 0.3, `T`: None}
    """

    return {key: getattr(model, key, None) for key in ("rho", "pi", "T")}


def to_dense(mat):
    """Converts a sparse matrix or array-like object to a dense NumPy array.
    If the matrix is already dense returns the original matrix.
    Args:
        mat: Input matrix or array-like object. If the input is a sparse matrix
             (e.g., SciPy sparse matrix with a `todense` method), it will be converted
             to a dense NumPy array. Otherwise, the input is returned as-is.

    Returns:
        np.ndarray: A dense NumPy array representation of the input.


    """
    if hasattr(mat, "todense"):
        return mat.todense()
    return mat


def remap_param_to_global(local_param, local_map, global_coll) -> np.ndarray:
    """Remaps a local parameter (scalar, vector, or matrix) to a global order
    based on a mapping.

    This function supports:
    - Scalar: A single numeric value, which is broadcasted to all global
        indices.
    - Vector: A value per local index, remapped to global indices.
    - Matrix/Tensor: A matrix or tensor per ID (e.g., confusion matrices),
        remapped to global indices.

    Args:
        local_param: The local parameter to remap. Can be a scalar, vector, or
            dictionary of matrices.
        local_map: A dictionary mapping local IDs to their local indices.
        global_coll: A collection or database interface to fetch global indices
            for each local ID.

    Returns:
        np.ndarray: A dense NumPy array of the remapped parameter in global
            order.
    """

    n = global_coll.count_documents({})

    # Handle scalar case early
    if np.isscalar(local_param) or np.ndim(local_param) == 0:
        return np.full((n,), float(local_param))

    # Determine shape of one entry
    if isinstance(local_param, dict):
        try:
            first_value = next(iter(local_param.values()))
        except StopIteration:
            first_value = 0.0
    else:
        # handle numpy arrays, lists, etc.
        first_value = (
            local_param[0] if np.ndim(local_param) > 0 else local_param
        )

    shape_suffix = np.shape(first_value)
    global_shape = (n,) + shape_suffix
    global_param = np.zeros(global_shape, dtype=float)

    #  Fill according to mapping
    for k, v in local_map.items():
        doc = global_coll.find_one({"_id": k})
        if doc is None:
            continue

        if isinstance(local_param, dict):
            value = local_param.get(k, np.zeros(shape_suffix))
        else:
            if v >= len(local_param):
                continue
            value = local_param[v]

        global_param[doc["index"]] = to_dense(value)

    return global_param


def remap_T_to_global(batch_T, task_mapping, class_mapping, model):
    """Remaps a local task-class matrix (`batch_T`) to a global task-class
    matrix.

    This function takes a local matrix `batch_T` (with dimensions defined by
    local task and class indices) and remaps it to a global matrix using
    the provided mappings and the model's global indices.

    Args:
        batch_T: A 2D NumPy array representing the local task-class matrix.
        task_mapping: A dictionary mapping local task IDs to their local
            indices.
        class_mapping: A dictionary mapping local class IDs to their local
            indices.
        model: An object containing the global task and class mappings, with
            methods to fetch global indices.

    Returns:
        np.ndarray: A 2D NumPy array representing the global task-class matrix.

    """

    n_tasks = model.task_mapping.count_documents({})
    n_classes = model.class_mapping.count_documents({})

    # Build local→global index maps
    task_map = {
        v: model.task_mapping.find_one({"_id": k})["index"]
        for k, v in task_mapping.items()
    }
    class_map = {
        v: model.class_mapping.find_one({"_id": k})["index"]
        for k, v in class_mapping.items()
    }

    # Initialize global T
    global_T = np.zeros((n_tasks, n_classes))

    for t_local, t_global in task_map.items():
        for c_local, c_global in class_map.items():
            global_T[t_global, c_global] = batch_T[t_local, c_local]

    return global_T


def em_trace(
    model,
    batch_matrix,
    task_mapping,
    worker_mapping,
    class_mapping,
    maxiter=50,
    epsilon=1e-6,
    prev_globals=None,
):
    """Runs the EM algorithm for one batch and traces the evolution of parameters.

    Args:
        model: Model object with EM-related methods (_m_step, _e_step, _online_update).
        batch_matrix: Matrix of observations for the current batch.
        task_mapping: Mapping of local task IDs to indices.
        worker_mapping: Mapping of local worker IDs to indices.
        class_mapping: Mapping of local class IDs to indices.
        maxiter: Maximum number of EM iterations (default: 50).
        epsilon: Convergence threshold for log-likelihood (default: 1e-6).
        prev_globals: Previous global parameters (optional).

    Returns:
        tuple: (traces, globals_after), where traces is a list of parameter states per iteration,
               and globals_after is a dictionary of global parameters after convergence.
    """
    traces = []
    prev_globals = prev_globals or {}

    batch_T = model._init_T(batch_matrix, task_mapping, class_mapping)

    i, eps, ll = 0, np.inf, []

    while i < maxiter and eps > epsilon:
        batch_rho, batch_pi = model._m_step(batch_matrix, batch_T)
        batch_T, batch_denom = model._e_step(batch_matrix, batch_pi, batch_rho)
        likeli = np.log(np.sum(batch_denom))
        ll.append(likeli)

        batch_rho_globaly_mapped = remap_param_to_global(
            batch_rho,
            class_mapping,
            model.class_mapping,
        )
        batch_pi_globaly_mapped = remap_param_to_global(
            batch_pi,
            worker_mapping,
            model.worker_mapping,
        )
        batch_T_globaly_mapped = remap_T_to_global(
            batch_T,
            task_mapping,
            class_mapping,
            model,
        )

        traces.append(
            {
                "iter": i,
                "rho": batch_rho_globaly_mapped.copy(),
                "pi": batch_pi_globaly_mapped.copy(),
                "pi_tensor": model.build_batch_pi_tensor(
                    batch_pi_globaly_mapped.copy(),
                    class_mapping,
                    worker_mapping,
                ),
                "T": batch_T_globaly_mapped.copy(),
                "likelihood": likeli,
                **{f"global_{k}": v for k, v in prev_globals.items()},
            },
        )
        if i > 0:
            eps = np.abs((ll[-1] - ll[-2]) / (np.abs(ll[-2]) + 1e-12))
        i += 1

    # Online update after convergence
    model._online_update(
        task_mapping,
        worker_mapping,
        class_mapping,
        batch_T,
        batch_rho,
        batch_pi,
    )

    # attach final global snapshot
    globals_after = get_global_params(model)
    if traces:
        traces[-1].update({f"global_{k}": v for k, v in globals_after.items()})

    return traces, globals_after


def prepare_batch(model, batch) -> tuple[Any, dict[str, Any]]:
    """Prepares a single batch for EM processing and returns the observation matrix and mappings.

    Args:
        model: Model object with methods to prepare mappings and process batches.
        batch: Input batch data to prepare.

    Returns:
        tuple: (batch_matrix, mappings), where batch_matrix is the observation matrix,
               and mappings is a dictionary of task, worker, and class mappings.
    """
    task_mapping, worker_mapping, class_mapping = {}, {}, {}
    model._prepare_mapping(batch, task_mapping, worker_mapping, class_mapping)

    # ensure indices exist in global mappings
    model.get_or_create_indices(model.task_mapping, list(task_mapping))
    model.get_or_create_indices(model.worker_mapping, list(worker_mapping))
    model.get_or_create_indices(model.class_mapping, list(class_mapping))

    batch_matrix = model._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )
    mappings = {
        "task_mapping": task_mapping.copy(),
        "worker_mapping": worker_mapping.copy(),
        "class_mapping": class_mapping.copy(),
    }
    return batch_matrix, mappings


def run_em_for_batches(model, batches, maxiter=50):
    """Runs the EM algorithm sequentially for multiple batches, carrying over global parameters.

    Args:
        model: Model object with EM-related methods and global parameter storage.
        batches: List of batch data to process.
        maxiter: Maximum number of EM iterations per batch (default: 50).

    Returns:
        tuple: (all_traces, all_mappings), where all_traces is a list of EM traces for all batches,
               and all_mappings is a list of mappings for each batch.
    """
    model.drop()
    model.t = 1
    all_traces = []
    all_mappings = []
    prev_globals = {}

    for idx, batch in enumerate(batches, 1):
        batch_matrix, mappings = prepare_batch(model, batch)
        traces, globals_after = em_trace(
            model,
            batch_matrix,
            mappings["task_mapping"],
            mappings["worker_mapping"],
            mappings["class_mapping"],
            maxiter=maxiter,
            prev_globals=prev_globals,
        )
        for t in traces:
            t["batch_num"] = idx

        all_traces.extend(traces)
        all_mappings.append(mappings)
        prev_globals = globals_after

        print(f"Finished batch {idx} (iterations: {len(traces)})")

    return all_traces, all_mappings


class PlotlyEMVisualizer:
    """Interactive visualization for EM model states, worker reliability, and confusion matrices.

    Attributes:
        model: The EM model object.
        trace_list: List of EM traces across iterations and batches.
        batch_data_map: Mapping of batch numbers to their data.
        batch_mappings: List of mappings for each batch.
        pi_renderer: Function to render pi (worker reliability) plots.
        labels: Dictionary of task, worker, and class labels.
        n_workers: Number of workers.
        fig: Plotly figure object.
        frames: List of animation frames.
    """

    def __init__(
        self,
        model,
        trace_list,
        batch_data_map,
        batch_mappings,
        colorscale="Viridis",
        pi_renderer=None,
    ):
        self.model = model
        self.trace_list = trace_list
        self.batch_data_map = batch_data_map
        self.batch_mappings = batch_mappings
        self.colorscale = colorscale
        self.pi_renderer = pi_renderer or self._plot_pi

        self.labels = self._get_labels()
        self.n_workers = len(self.labels["workers"])
        self.fig = None
        self.frames = []

    # - Public API -

    def show(self):
        """Build and display the interactive visualization."""
        self.fig = self._create_base_figure()
        self._build_frames()
        self._apply_layout()
        self.fig.update_layout(showlegend=False)
        self.fig.show()

    # - Internal Helpers -

    def _plot_pi(
        self,
        pi,
        label_prefix="batch",
        worker_labels=None,
        class_labels=None,
    ):
        """Generates Plotly traces for pi (worker reliability) based on its shape.

        Args:
            pi: The pi parameter (scalar, vector, matrix, or tensor).
            label_prefix: Prefix for trace names (default: "batch").
            worker_labels: Labels for workers (optional).
            class_labels: Labels for classes (optional).

        Returns:
            list: List of Plotly traces for pi.
        """

        if pi is None:
            return [go.Bar(y=[0], x=["none"], name=f"{label_prefix}_pi")]

        pi = np.array(pi)
        if pi.ndim == 0:
            return [
                go.Bar(
                    y=[pi.item()],
                    x=[f"{label_prefix}_pi"],
                    name=f"{label_prefix}_pi",
                ),
            ]
        if pi.ndim == 1:
            return [
                go.Bar(
                    y=pi,
                    x=worker_labels or list(range(len(pi))),
                    name=f"{label_prefix}_pi",
                ),
            ]
        if pi.ndim == 2:
            return [
                go.Heatmap(
                    z=pi,
                    x=class_labels or list(range(pi.shape[1])),
                    y=worker_labels or list(range(pi.shape[0])),
                    colorscale=self.colorscale,
                    showscale=False,
                    name=f"{label_prefix}_pi",
                ),
            ]
        if pi.ndim == 3:
            return [
                go.Heatmap(
                    z=pi[j],
                    x=class_labels or list(range(pi.shape[2])),
                    y=class_labels or list(range(pi.shape[1])),
                    colorscale=self.colorscale,
                    showscale=False,
                    name=f"{label_prefix}_pi[{j}]",
                )
                for j in range(pi.shape[0])
            ]
        raise ValueError(f"Unsupported pi dimensionality: {pi.ndim}")

    def _get_ids(self, mapping):
        """Extracts ordered IDs from a mapping collection, sorted by index.

        Args:
            mapping: A collection or database interface with `find` and `sort` methods.

        Returns:
            list: A list of IDs sorted by their index.
        """
        return [doc["_id"] for doc in mapping.find().sort("index", 1)]

    def _get_labels(self):
        """Extracts and returns ordered labels for tasks, workers, and classes from model mappings.

        Returns:
            dict: A dictionary with keys 'tasks', 'workers', and 'classes',
                each mapping to a list of sorted IDs.
        """
        return {
            "tasks": self._get_ids(self.model.task_mapping),
            "workers": self._get_ids(self.model.worker_mapping),
            "classes": self._get_ids(self.model.class_mapping),
        }

    def _create_base_figure(self):
        """Creates and initializes a multi-subplot Plotly figure with placeholders for EM visualization.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure with subplots for EM matrices, worker reliability, and tables.
        """
        task_labels = self.labels["tasks"]
        worker_labels = self.labels["workers"]
        class_labels = self.labels["classes"]
        total_cols = max(3, self.n_workers)

        fig = make_subplots(
            rows=3,
            cols=total_cols,
            row_heights=[0.25, 0.25, 0.4],
            vertical_spacing=0.12,
            subplot_titles=[
                "batch_rho (class priors)",
                "batch_T (local task x class)",
                "batch_pi (worker reliability)",
                "user votes",
                "global_rho (class priors)",
                "global_T (local task x class)",
                "global_pi (worker reliability)",
            ]
            + [f"batch_pi of {w}" for w in worker_labels]
            + ["Batch Table"],
            specs=[
                [
                    {"type": "bar"},
                    {"type": "heatmap"},
                    {"type": "bar"},
                    {"type": "table", "rowspan": 2},
                ]
                + [None] * (total_cols - 4),
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "bar"}, None]
                + [None] * (total_cols - 4),
                [{"type": "heatmap"} for _ in range(self.n_workers)]
                + [None] * (total_cols - self.n_workers),
            ],
        )

        # Placeholders for EM matrices
        fig.add_trace(go.Bar(y=[0], x=class_labels), row=1, col=1)
        fig.add_trace(
            go.Heatmap(
                z=[[0]],
                x=class_labels,
                y=task_labels,
                showscale=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(go.Bar(y=[0], x=worker_labels), row=1, col=3)
        fig.add_trace(go.Bar(y=[0], x=class_labels), row=2, col=1)
        fig.add_trace(
            go.Heatmap(
                z=[[0]],
                x=class_labels,
                y=task_labels,
                showscale=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(go.Bar(y=[0], x=worker_labels), row=2, col=3)

        # Worker confusion matrix placeholders
        for i in range(self.n_workers):
            fig.add_trace(
                go.Heatmap(
                    z=np.zeros((len(class_labels), len(class_labels))),
                    x=class_labels,
                    y=class_labels,
                    colorscale=self.colorscale,
                    showscale=False,
                    zmin=0,
                    zmax=1,
                ),
                row=3,
                col=i + 1,
            )

        # Empty table placeholder
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Task", "User", "Species"],
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(values=[[], [], []]),
            ),
            row=1,
            col=4,
        )

        return fig

    def _build_frames(self):
        """Generates animation frames for the Plotly figure based on EM traces and batch data."""
        for i, state in enumerate(self.trace_list):
            batch_num = state.get("batch_num", "?")
            batch_data = self.batch_data_map.get(batch_num, {})
            batch_mapping = next(
                (
                    m
                    for idx, m in enumerate(self.batch_mappings)
                    if idx + 1 == batch_num
                ),
                None,
            )
            batch_class_labels = (
                list(batch_mapping["class_mapping"].keys())
                if batch_mapping
                else self.labels["classes"]
            )

            frame_data = self._make_frame_data(
                state,
                batch_class_labels,
                batch_data,
            )

            frame = go.Frame(
                data=frame_data,
                name=f"iter_{i}",
                layout=go.Layout(
                    title_text=(
                        f"Batch {batch_num}, Iteration {state['iter']} - "
                        f"LL: {state.get('likelihood', 0.0):.3f}"
                    ),
                ),
            )
            self.frames.append(frame)

        self.fig.frames = self.frames

    def _make_frame_data(self, state, batch_class_labels, batch_data):
        """Creates Plotly traces for a single animation frame based on the EM state and batch data.

        Args:
            state: Current EM state dictionary.
            batch_class_labels: Labels for classes in the current batch.
            batch_data: Data for the current batch.

        Returns:
            list: List of Plotly traces for the frame.
        """
        task_labels = self.labels["tasks"]
        worker_labels = self.labels["workers"]
        all_class_labels = self.labels["classes"]

        # Extract matrices
        batch_rho = to_dense(state.get("rho"))
        batch_T = to_dense(state.get("T"))
        global_rho = to_dense(state.get("global_rho"))
        global_T = to_dense(state.get("global_T"))
        pi_tensor = to_dense(state.get("pi_tensor"))

        # Build user table
        rows = [
            (task, user, species)
            for task, users in (batch_data or {}).items()
            for user, species in users.items()
        ]
        cols = list(zip(*rows)) if rows else [[], [], []]

        # Create traces
        frame_data = [
            go.Bar(
                y=batch_rho.flatten() if batch_rho is not None else [0],
                x=batch_class_labels,
            ),
            go.Heatmap(
                z=batch_T if batch_T is not None else np.zeros((1, 1)),
                x=batch_class_labels,
                y=task_labels,
                showscale=False,
                colorscale=self.colorscale,
            ),
        ]
        frame_data.extend(
            self._plot_pi(
                state.get("pi"),
                label_prefix="batch",
                worker_labels=worker_labels,
                class_labels=all_class_labels,
            ),
        )
        frame_data.extend(
            [
                go.Bar(
                    y=global_rho.flatten() if global_rho is not None else [0],
                    x=all_class_labels,
                ),
                go.Heatmap(
                    z=global_T if global_T is not None else np.zeros((1, 1)),
                    x=all_class_labels,
                    y=task_labels,
                    showscale=False,
                    colorscale=self.colorscale,
                ),
            ],
        )
        frame_data.extend(
            self._plot_pi(
                state.get("global_pi"),
                label_prefix="global",
                worker_labels=worker_labels,
                class_labels=all_class_labels,
            ),
        )

        # Worker confusion matrices
        if pi_tensor is not None:
            for j in range(self.n_workers):
                frame_data.append(
                    go.Heatmap(
                        z=pi_tensor[j],
                        x=batch_class_labels,
                        y=batch_class_labels,
                        colorscale=self.colorscale,
                        showscale=True,
                    ),
                )
        else:
            for _ in range(self.n_workers):
                frame_data.append(
                    go.Heatmap(
                        z=np.zeros(
                            (len(batch_class_labels), len(batch_class_labels)),
                        ),
                        x=batch_class_labels,
                        y=batch_class_labels,
                        colorscale=self.colorscale,
                        showscale=False,
                    ),
                )

        # Append user voting table
        frame_data.append(self._make_table_trace(cols))
        return frame_data

    def _make_table_trace(self, cols):
        """Creates a Plotly table trace for displaying batch data.

        Args:
            cols: List of column values for the table.

        Returns:
            plotly.graph_objects.Table: A Plotly table trace.
        """
        return go.Table(
            header=dict(
                values=["Task", "User", "Species"],
                fill_color="lightgrey",
                align="left",
            ),
            cells=dict(
                values=cols,
                align="left",
                fill_color=[["white"] * len(cols[0])],
            ),
        )

    def _apply_layout(self):
        """Applies layout settings to the Plotly figure, including titles, axis labels, and animations."""
        self.fig.update_layout(
            title=f"Iteration 0 - Log-likelihood: {self.trace_list[0].get('likelihood', 0.0):.3f}",
            height=1200,
            width=320 * self.n_workers,
            template="plotly_white",
            sliders=[
                {
                    "x": 0.1,
                    "y": -0.1,
                    "steps": [
                        {
                            "args": [
                                [f"iter_{i}"],
                                {"frame": {"duration": 0, "redraw": True}},
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i in range(len(self.frames))
                    ],
                },
            ],
        )
        for i in range(1, self.n_workers + 1):
            self.fig.update_yaxes(showticklabels=(i == 1), row=3, col=i)


def visualize_model(
    model,
    batches,
    maxiter=50,
    colorscale="Viridis",
    pi_renderer=None,
):
    model_instance = model()

    combined_traces, batch_mappings = run_em_for_batches(
        model_instance,
        batches,
        maxiter=maxiter,
    )

    viz = PlotlyEMVisualizer(
        model=model_instance,
        trace_list=combined_traces,
        batch_data_map={i + 1: b for i, b in enumerate(batches)},
        batch_mappings=batch_mappings,
        colorscale=colorscale,
        pi_renderer=pi_renderer,
    )
    viz.show()
