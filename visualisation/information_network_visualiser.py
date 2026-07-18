from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from information.graph_builder import (
    InformationGraphConfig,
    build_information_graph,
    detect_communities,
)


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
PositionMap: TypeAlias = dict[str, FloatArray]


@dataclass(frozen=True)
class InformationNetworkData:
    matrices: FloatArray
    names: list[str]
    window_ends: IntArray | None = None

    def __post_init__(self) -> None:
        if self.matrices.ndim != 3:
            raise ValueError(
                "matrices must have shape "
                "(n_windows, n_tickers, n_tickers)."
            )

        number_of_windows, rows, columns = self.matrices.shape

        if rows != columns:
            raise ValueError("Every MI matrix must be square.")

        if rows != len(self.names):
            raise ValueError(
                f"Matrices contain {rows} tickers, "
                f"but {len(self.names)} names were supplied."
            )

        if number_of_windows < 1:
            raise ValueError("At least one rolling matrix is required.")

        if self.window_ends is not None:
            if len(self.window_ends) != number_of_windows:
                raise ValueError(
                    "window_ends must contain one value per matrix."
                )


class InformationNetworkVisualiser:
    def __init__(
        self,
        data: InformationNetworkData,
        graph_config: InformationGraphConfig | None = None,
        animation_interval_ms: int = 250,
        layout_iterations: int = 25,
        layout_seed: int = 42,
    ) -> None:
        self.data = data
        self.graph_config = (
            graph_config or InformationGraphConfig()
        )

        self.animation_interval_ms = animation_interval_ms
        self.layout_iterations = layout_iterations
        self.layout_seed = layout_seed

        self.current_index = 0
        self.is_playing = False
        self.positions: PositionMap | None = None
        self.animation: FuncAnimation | None = None
        self._updating_slider = False

        self.figure: Figure
        self.graph_axis: Axes
        self.info_axis: Axes
        self.slider: Slider
        self.play_button: Button
        self.reset_button: Button

        self.edge_collection: LineCollection | None = None
        self.node_collection: PathCollection | None = None
        self.label_artists: list[Text] = []

        self._create_interface()
        self._draw_window(0)

    @property
    def number_of_windows(self) -> int:
        return int(self.data.matrices.shape[0])

    def _create_interface(self) -> None:
        self.figure = plt.figure(figsize=(15, 9))

        self.graph_axis = self.figure.add_axes(
            (0.04, 0.16, 0.72, 0.79)
        )
        self.info_axis = self.figure.add_axes(
            (0.79, 0.16, 0.19, 0.79)
        )

        slider_axis = self.figure.add_axes(
            (0.17, 0.075, 0.57, 0.035)
        )
        play_axis = self.figure.add_axes(
            (0.04, 0.055, 0.09, 0.065)
        )
        reset_axis = self.figure.add_axes(
            (0.79, 0.055, 0.11, 0.065)
        )

        self.slider = Slider(
            ax=slider_axis,
            label="Rolling window",
            valmin=0,
            valmax=self.number_of_windows - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_changed)

        self.play_button = Button(play_axis, "Play")
        self.play_button.on_clicked(self._toggle_playback)

        self.reset_button = Button(reset_axis, "Reset layout")
        self.reset_button.on_clicked(self._reset_layout)

        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key_press,
        )

    def _window_title(self, index: int) -> str:
        if self.data.window_ends is None:
            return (
                "Rolling information network — "
                f"window {index + 1}"
            )

        end_day = int(self.data.window_ends[index])

        return (
            "Rolling information network — "
            f"window {index + 1}, ending day {end_day}"
        )

    def _calculate_layout(
        self,
        graph: nx.Graph | nx.DiGraph,
    ) -> PositionMap:
        if graph.number_of_nodes() == 0:
            return {}

        layout_graph = graph.copy()

        spring_weights = np.asarray(
            [
                float(attributes.get("spring_weight", 0.0))
                for _, _, attributes in layout_graph.edges(data=True)
            ],
            dtype=float,
        )

        maximum_weight = (
            float(np.max(spring_weights))
            if spring_weights.size
            else 0.0
        )

        for _, _, attributes in layout_graph.edges(data=True):
            raw_weight = float(
                attributes.get("spring_weight", 0.0)
            )

            attributes["layout_weight"] = (
                raw_weight / maximum_weight
                if maximum_weight > 0.0
                else 1.0
            )

        initial_positions = self.positions

        layout = nx.spring_layout(
            layout_graph,
            pos=initial_positions,
            iterations=self.layout_iterations,
            weight="layout_weight",
            seed=self.layout_seed,
            k=1.8 / np.sqrt(max(graph.number_of_nodes(), 1)),
        )

        return {
            str(node): np.asarray(position, dtype=float)
            for node, position in layout.items()
        }

    @staticmethod
    def _community_mapping(
        graph: nx.Graph | nx.DiGraph,
    ) -> tuple[list[set[str]], dict[str, int]]:
        communities = detect_communities(graph)

        mapping: dict[str, int] = {}

        for index, community in enumerate(communities):
            for ticker in community:
                mapping[ticker] = index

        return communities, mapping

    @staticmethod
    def _normalise(
        values: FloatArray,
    ) -> FloatArray:
        if values.size == 0:
            return values

        minimum = float(np.min(values))
        maximum = float(np.max(values))

        if np.isclose(minimum, maximum):
            return np.ones_like(values)

        return (values - minimum) / (maximum - minimum)

    @staticmethod
    def _node_sizes(
        graph: nx.Graph | nx.DiGraph,
        nodes: list[str],
    ) -> FloatArray:
        if isinstance(graph, nx.DiGraph):
            weighted_degree = dict(
                graph.out_degree(weight="mi")
            )
        else:
            weighted_degree = dict(
                graph.degree(weight="mi")
            )

        values = np.asarray(
            [
                float(weighted_degree.get(node, 0.0))
                for node in nodes
            ],
            dtype=float,
        )

        normalised = (
            InformationNetworkVisualiser._normalise(values)
        )

        return 350.0 + 1_200.0 * normalised

    @staticmethod
    def _edge_segments(
        graph: nx.Graph | nx.DiGraph,
        positions: PositionMap,
    ) -> tuple[list[list[FloatArray]], FloatArray]:
        segments: list[list[FloatArray]] = []
        weights: list[float] = []

        for source, target, attributes in graph.edges(data=True):
            if source not in positions or target not in positions:
                continue

            start = positions[str(source)]
            end = positions[str(target)]

            segments.append([start, end])
            weights.append(float(attributes["mi"]))

        return segments, np.asarray(weights, dtype=float)

    @staticmethod
    def _edge_widths(weights: FloatArray) -> FloatArray:
        normalised = InformationNetworkVisualiser._normalise(
            weights
        )

        return 0.35 + 7.5 * np.power(normalised, 1.4)

    @staticmethod
    def _edge_colours(weights: FloatArray) -> FloatArray:
        normalised = InformationNetworkVisualiser._normalise(
            weights
        )

        alphas = 0.12 + 0.78 * normalised

        colours = np.zeros((len(weights), 4), dtype=float)
        colours[:, :3] = 0.15
        colours[:, 3] = alphas

        return colours

    def _draw_edges(
        self,
        graph: nx.Graph | nx.DiGraph,
    ) -> None:
        assert self.positions is not None

        segments, weights = self._edge_segments(
            graph,
            self.positions,
        )

        if not segments:
            self.edge_collection = None
            return

        widths = self._edge_widths(weights)
        colours = self._edge_colours(weights)

        self.edge_collection = LineCollection(
            segments,
            linewidths=widths,
            colors=colours,
            zorder=1,
        )

        self.graph_axis.add_collection(self.edge_collection)

        if graph.is_directed():
            self._draw_arrowheads(
                graph=graph,
                weights=weights,
            )

    def _draw_arrowheads(
        self,
        graph: nx.Graph | nx.DiGraph,
        weights: FloatArray,
    ) -> None:
        assert self.positions is not None

        normalised = self._normalise(weights)

        for edge_index, (source, target) in enumerate(
            graph.edges()
        ):
            start = self.positions[str(source)]
            end = self.positions[str(target)]

            direction = end - start
            distance = float(np.linalg.norm(direction))

            if distance <= 1e-12:
                continue

            unit_direction = direction / distance

            arrow_end = end - 0.045 * unit_direction
            arrow_start = arrow_end - 0.08 * unit_direction

            alpha = float(
                0.2 + 0.75 * normalised[edge_index]
            )
            width = float(
                0.6 + 2.2 * normalised[edge_index]
            )

            self.graph_axis.annotate(
                "",
                xy=tuple(arrow_end),
                xytext=tuple(arrow_start),
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": width,
                    "alpha": alpha,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )

    def _draw_nodes(
        self,
        graph: nx.Graph | nx.DiGraph,
        community_mapping: dict[str, int],
    ) -> None:
        assert self.positions is not None

        nodes = list(graph.nodes)

        coordinates = np.asarray(
            [self.positions[str(node)] for node in nodes],
            dtype=float,
        )

        node_sizes = self._node_sizes(graph, nodes)

        colours = np.asarray(
            [
                community_mapping.get(str(node), 0)
                for node in nodes
            ],
            dtype=float,
        )

        self.node_collection = self.graph_axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=node_sizes,
            c=colours,
            cmap="tab20",
            alpha=0.92,
            linewidths=1.0,
            edgecolors="black",
            zorder=3,
        )

    def _draw_labels(
        self,
        graph: nx.Graph | nx.DiGraph,
    ) -> None:
        assert self.positions is not None

        self.label_artists = []

        for node in graph.nodes:
            x_position, y_position = self.positions[str(node)]

            label = self.graph_axis.text(
                x_position,
                y_position,
                str(node),
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

            self.label_artists.append(label)

    def _set_axis_limits(self) -> None:
        if not self.positions:
            self.graph_axis.set_xlim(-1.0, 1.0)
            self.graph_axis.set_ylim(-1.0, 1.0)
            return

        coordinates = np.asarray(
            list(self.positions.values()),
            dtype=float,
        )

        minimums = coordinates.min(axis=0)
        maximums = coordinates.max(axis=0)

        ranges = maximums - minimums
        ranges = np.where(ranges <= 1e-12, 1.0, ranges)

        padding = ranges * 0.18

        self.graph_axis.set_xlim(
            minimums[0] - padding[0],
            maximums[0] + padding[0],
        )
        self.graph_axis.set_ylim(
            minimums[1] - padding[1],
            maximums[1] + padding[1],
        )

    def _draw_window(self, index: int) -> None:
        index = int(
            np.clip(
                index,
                0,
                self.number_of_windows - 1,
            )
        )
        self.current_index = index

        matrix = self.data.matrices[index]

        graph = build_information_graph(
            matrix=matrix,
            names=self.data.names,
            config=self.graph_config,
        )

        self.positions = self._calculate_layout(graph)

        communities, community_mapping = (
            self._community_mapping(graph)
        )

        self.graph_axis.clear()
        self.info_axis.clear()

        self._draw_edges(graph)
        self._draw_nodes(graph, community_mapping)
        self._draw_labels(graph)
        self._set_axis_limits()

        self.graph_axis.set_title(
            self._window_title(index),
            fontsize=14,
        )
        self.graph_axis.set_axis_off()

        self._draw_information_panel(
            graph=graph,
            communities=communities,
            matrix=matrix,
        )

        if not self._updating_slider:
            self._updating_slider = True
            self.slider.set_val(index)
            self._updating_slider = False

        self.figure.canvas.draw_idle()

    def _draw_information_panel(
        self,
        graph: nx.Graph | nx.DiGraph,
        communities: list[set[str]],
        matrix: FloatArray,
    ) -> None:
        self.info_axis.set_axis_off()

        edge_values = np.asarray(
            [
                float(attributes["mi"])
                for _, _, attributes in graph.edges(data=True)
            ],
            dtype=float,
        )

        off_diagonal_mask = ~np.eye(
            matrix.shape[0],
            dtype=bool,
        )

        off_diagonal = matrix[off_diagonal_mask]
        positive_values = off_diagonal[
            off_diagonal > 0.0
        ]

        average_mi = (
            float(np.mean(positive_values))
            if positive_values.size
            else 0.0
        )

        strongest_mi = (
            float(np.max(edge_values))
            if edge_values.size
            else 0.0
        )

        number_of_nodes = graph.number_of_nodes()

        possible_edges = (
            number_of_nodes * (number_of_nodes - 1)
        )

        if not graph.is_directed():
            possible_edges //= 2

        density = (
            graph.number_of_edges() / possible_edges
            if possible_edges > 0
            else 0.0
        )

        largest_community = max(
            (
                len(community)
                for community in communities
            ),
            default=0,
        )

        lines = [
            "NETWORK STATISTICS",
            "",
            (
                f"Window: {self.current_index + 1}/"
                f"{self.number_of_windows}"
            ),
            f"Nodes: {number_of_nodes}",
            f"Visible edges: {graph.number_of_edges()}",
            f"Graph density: {density:.3f}",
            "",
            f"Communities: {len(communities)}",
            f"Largest community: {largest_community}",
            "",
            f"Mean matrix MI: {average_mi:.5f}",
            f"Strongest visible MI: {strongest_mi:.5f}",
            (
                "Edge threshold: "
                f"{float(graph.graph['threshold']):.5f}"
            ),
            "",
            "LARGEST COMMUNITIES",
            "",
        ]

        ordered_communities = sorted(
            communities,
            key=len,
            reverse=True,
        )

        for number, community in enumerate(
            ordered_communities[:5],
            start=1,
        ):
            members = ", ".join(sorted(community))

            if len(members) > 32:
                members = members[:29] + "..."

            lines.append(
                f"{number}. {len(community)} nodes"
            )
            lines.append(f"   {members}")

        self.info_axis.text(
            0.0,
            1.0,
            "\n".join(lines),
            transform=self.info_axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    def _on_slider_changed(self, value: float) -> None:
        if self._updating_slider:
            return

        self._draw_window(int(value))

    def _toggle_playback(
        self,
        _: Event | None = None,
    ) -> None:
        self.is_playing = not self.is_playing

        self.play_button.label.set_text(
            "Pause" if self.is_playing else "Play"
        )

        if self.animation is None:
            self.animation = FuncAnimation(
                self.figure,
                self._advance_animation,
                interval=self.animation_interval_ms,
                cache_frame_data=False,
            )

    def _advance_animation(self, _: int) -> None:
        if not self.is_playing:
            return

        next_index = self.current_index + 1

        if next_index >= self.number_of_windows:
            next_index = 0

        self._draw_window(next_index)

    def _reset_layout(
        self,
        _: Event,
    ) -> None:
        self.positions = None
        self._draw_window(self.current_index)

    def _on_key_press(self, event: KeyEvent) -> None:
        if event.key == "right":
            self._draw_window(
                min(
                    self.current_index + 1,
                    self.number_of_windows - 1,
                )
            )

        elif event.key == "left":
            self._draw_window(
                max(self.current_index - 1, 0)
            )

        elif event.key == " ":
            self._toggle_playback(None)

    def show(self) -> None:
        plt.show()


def load_information_network_data(
    path: str | Path,
) -> InformationNetworkData:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as archive:
        available_keys = set(archive.files)

        matrix_key = next(
            (
                key
                for key in (
                    "matrices",
                    "rolling_matrices",
                    "mi_matrices",
                    "rolling_mi",
                )
                if key in available_keys
            ),
            None,
        )

        if matrix_key is None:
            raise KeyError(
                "Could not find rolling matrices. Expected one of: "
                "matrices, rolling_matrices, mi_matrices, rolling_mi. "
                f"Found: {sorted(available_keys)}"
            )

        name_key = next(
            (
                key
                for key in (
                    "names",
                    "tickers",
                    "symbols",
                )
                if key in available_keys
            ),
            None,
        )

        if name_key is None:
            raise KeyError(
                "Could not find ticker names. Expected one of: "
                "names, tickers, symbols."
            )

        matrices = np.asarray(
            archive[matrix_key],
            dtype=np.float64,
        )

        names = (
            archive[name_key]
            .astype(str)
            .tolist()
        )

        window_ends: IntArray | None = None

        for key in (
            "window_ends",
            "end_days",
            "indices",
        ):
            if key in available_keys:
                window_ends = np.asarray(
                    archive[key],
                    dtype=np.int64,
                )
                break

    return InformationNetworkData(
        matrices=matrices,
        names=names,
        window_ends=window_ends,
    )