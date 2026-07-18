from __future__ import annotations

from dataclasses import dataclass
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from information.rolling_information import RollingInformationData

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulatorConfig:
    percentile: float = 90.0
    top_k: int = 4
    spring_exponent: float = 1.5

    repulsion_strength: float = 2.0
    spring_strength: float = 18.0
    rest_length: float = 1.4
    centre_strength: float = 0.15

    damping: float = 0.88
    time_step: float = 0.018
    physics_substeps: int = 3
    maximum_speed: float = 5.0

    transition_seconds: float = 0.7
    window_interval_seconds: float = 1.5

def _window_description(
    data: RollingInformationData,
    index: int,
) -> str:
    end_day = int(data.window_ends[index])
    start_day = max(0, end_day - data.window_size + 1)

    return (
        f"days {start_day}–{end_day}, "
        f"lag {data.lag}"
    )


class InformationNetworkSimulator(QtWidgets.QMainWindow):
    def __init__(
        self,
        data: RollingInformationData,
        config: SimulatorConfig,
    ) -> None:
        super().__init__()

        self.data = data
        self.config = config

        self.number_of_windows = int(data.matrices.shape[0])
        self.number_of_nodes = len(data.names)

        random_generator = np.random.default_rng(42)

        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.number_of_nodes,
            endpoint=False,
        )

        self.positions = np.column_stack(
            (
                4.0 * np.cos(angles),
                4.0 * np.sin(angles),
            )
        )

        self.positions += random_generator.normal(
            scale=0.15,
            size=self.positions.shape,
        )

        self.velocities = np.zeros_like(self.positions)

        self.current_window = 0
        self.previous_matrix = self.data.matrices[0].copy()
        self.target_matrix = self.data.matrices[0].copy()
        self.display_matrix = self.target_matrix.copy()

        self.transition_progress = 1.0
        self.playing = False
        self.elapsed_since_window_change = 0.0

        self.visible_edges: list[tuple[int, int, float]] = []
        self.edge_items: list[pg.PlotDataItem] = []
        self.label_items: list[pg.TextItem] = []

        self.setWindowTitle(
            "Dynamic Mutual-Information Network"
        )
        self.resize(1500, 950)

        self._create_interface()
        self._set_target_window(0, immediate=True)

        self.physics_timer = QtCore.QTimer(self)
        self.physics_timer.timeout.connect(self._tick)
        self.physics_timer.start(16)

    def _create_interface(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QtWidgets.QVBoxLayout(central_widget)

        content_layout = QtWidgets.QHBoxLayout()
        root_layout.addLayout(content_layout, stretch=1)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.setBackground(None)

        content_layout.addWidget(
            self.plot_widget,
            stretch=5,
        )

        self.stats_box = QtWidgets.QPlainTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumWidth(315)

        content_layout.addWidget(
            self.stats_box,
            stretch=1,
        )

        controls_layout = QtWidgets.QHBoxLayout()
        root_layout.addLayout(controls_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playing)
        controls_layout.addWidget(self.play_button)

        self.reset_button = QtWidgets.QPushButton(
            "Reset physics"
        )
        self.reset_button.clicked.connect(
            self._reset_physics
        )
        controls_layout.addWidget(self.reset_button)

        self.window_label = QtWidgets.QLabel()
        controls_layout.addWidget(self.window_label)

        self.slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.number_of_windows - 1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(
            self._slider_changed
        )

        controls_layout.addWidget(
            self.slider,
            stretch=1,
        )

        controls_layout.addWidget(
            QtWidgets.QLabel("Playback speed")
        )

        self.speed_box = QtWidgets.QDoubleSpinBox()
        self.speed_box.setRange(0.1, 10.0)
        self.speed_box.setSingleStep(0.1)
        self.speed_box.setValue(
            self.config.window_interval_seconds
        )
        self.speed_box.setSuffix(" s/window")
        controls_layout.addWidget(self.speed_box)

        self.node_item = pg.ScatterPlotItem(
            pxMode=True,
            pen=pg.mkPen(40, 40, 40, 220, width=1.2),
        )
        self.plot_widget.addItem(self.node_item)

        for name in self.data.names:
            label = pg.TextItem(
                text=name,
                anchor=(0.5, 0.5),
            )
            label.setZValue(20)
            self.plot_widget.addItem(label)
            self.label_items.append(label)

    def _slider_changed(self, value: int) -> None:
        self._set_target_window(int(value))

    def _toggle_playing(self) -> None:
        print("PLAY CLICKED")

        self.playing = not self.playing

        print("playing =", self.playing)

        self.play_button.setText(
            "Pause" if self.playing else "Play"
        )

        self.elapsed_since_window_change = 0.0

    def _reset_physics(self) -> None:
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.number_of_nodes,
            endpoint=False,
        )

        self.positions[:, 0] = 4.0 * np.cos(angles)
        self.positions[:, 1] = 4.0 * np.sin(angles)
        self.velocities.fill(0.0)

    def _set_target_window(
        self,
        index: int,
        immediate: bool = False,
    ) -> None:
        index = int(
            np.clip(
                index,
                0,
                self.number_of_windows - 1,
            )
        )

        self.current_window = index
        self.previous_matrix = self.display_matrix.copy()
        self.target_matrix = self.data.matrices[index].copy()

        self.transition_progress = (
            1.0 if immediate else 0.0
        )

        if immediate:
            self.display_matrix = self.target_matrix.copy()

        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

        self._update_window_label()

    def _update_window_label(self) -> None:
        self.window_label.setText(
            f"Window {self.current_window + 1}/"
            f"{self.number_of_windows}: "
            f"{_window_description(self.data, self.current_window)}"
        )

    def _update_transition(
        self,
        frame_seconds: float,
    ) -> None:
        if self.transition_progress >= 1.0:
            self.display_matrix = self.target_matrix
            return

        duration = max(
            self.config.transition_seconds,
            1e-6,
        )

        self.transition_progress = min(
            1.0,
            self.transition_progress
            + frame_seconds / duration,
        )

        # Smoothstep interpolation.
        progress = self.transition_progress
        smooth_progress = (
            progress * progress * (3.0 - 2.0 * progress)
        )

        self.display_matrix = (
            (1.0 - smooth_progress) * self.previous_matrix
            + smooth_progress * self.target_matrix
        )

    def _pruned_edges(
        self,
        matrix: FloatArray,
    ) -> list[tuple[int, int, float]]:
        number_of_nodes = matrix.shape[0]

        off_diagonal = matrix[
            ~np.eye(number_of_nodes, dtype=bool)
        ]

        positive_values = off_diagonal[
            np.isfinite(off_diagonal)
            & (off_diagonal > 0.0)
        ]

        if positive_values.size == 0:
            return []

        threshold = float(
            np.percentile(
                positive_values,
                self.config.percentile,
            )
        )

        edges: list[tuple[int, int, float]] = []

        for source in range(number_of_nodes):
            row = matrix[source].copy()
            row[source] = -np.inf

            candidates = np.flatnonzero(row >= threshold)

            if candidates.size > self.config.top_k:
                ordered = candidates[
                    np.argsort(row[candidates])[::-1]
                ]
                candidates = ordered[: self.config.top_k]

            for target in candidates:
                weight = float(row[target])

                if weight > 0.0 and np.isfinite(weight):
                    edges.append(
                        (source, int(target), weight)
                    )

        return edges

    def _spring_edges(
        self,
        directed_edges: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, float]]:
        combined: dict[tuple[int, int], float] = {}

        for source, target, weight in directed_edges:
            edge = (
                min(source, target),
                max(source, target),
            )

            combined[edge] = max(
                combined.get(edge, 0.0),
                weight,
            )

        return [
            (source, target, weight)
            for (source, target), weight
            in combined.items()
        ]

    def _calculate_forces(
        self,
        spring_edges: list[tuple[int, int, float]],
    ) -> FloatArray:
        positions = self.positions
        node_count = self.number_of_nodes

        forces = np.zeros_like(positions)

        # Pairwise Coulomb-style repulsion.
        differences = (
            positions[:, np.newaxis, :]
            - positions[np.newaxis, :, :]
        )

        squared_distances = np.sum(
            differences * differences,
            axis=2,
        )

        squared_distances += np.eye(node_count)
        squared_distances = np.maximum(
            squared_distances,
            0.035,
        )

        distances = np.sqrt(squared_distances)

        directions = differences / distances[:, :, np.newaxis]

        repulsion_magnitudes = (
            self.config.repulsion_strength
            / squared_distances
        )

        np.fill_diagonal(repulsion_magnitudes, 0.0)

        forces += np.sum(
            directions
            * repulsion_magnitudes[:, :, np.newaxis],
            axis=1,
        )

        if spring_edges:
            maximum_weight = max(
                weight
                for _, _, weight in spring_edges
            )

            maximum_weight = max(maximum_weight, 1e-12)

            for source, target, weight in spring_edges:
                displacement = (
                    positions[target] - positions[source]
                )

                distance = float(
                    np.linalg.norm(displacement)
                )

                if distance <= 1e-9:
                    continue

                direction = displacement / distance

                normalised_weight = weight / maximum_weight

                spring_constant = (
                    self.config.spring_strength
                    * normalised_weight
                    ** self.config.spring_exponent
                )

                # Stronger links also prefer a slightly shorter distance.
                preferred_length = (
                    self.config.rest_length
                    * (
                        1.15
                        - 0.55 * normalised_weight
                    )
                )

                extension = distance - preferred_length

                spring_force = (
                    spring_constant
                    * extension
                    * direction
                )

                forces[source] += spring_force
                forces[target] -= spring_force

        # Pull the entire graph gently towards the origin.
        forces -= (
            self.config.centre_strength
            * positions
        )

        return forces

    def _physics_step(
        self,
        spring_edges: list[tuple[int, int, float]],
    ) -> None:
        forces = self._calculate_forces(spring_edges)

        time_step = self.config.time_step

        self.velocities += forces * time_step
        self.velocities *= self.config.damping

        speeds = np.linalg.norm(
            self.velocities,
            axis=1,
        )

        too_fast = speeds > self.config.maximum_speed

        if np.any(too_fast):
            self.velocities[too_fast] *= (
                self.config.maximum_speed
                / speeds[too_fast]
            )[:, np.newaxis]

        self.positions += self.velocities * time_step

    def _community_colours(
        self,
        edges: list[tuple[int, int, float]],
    ) -> list[object]:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.number_of_nodes))

        for source, target, weight in edges:
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += weight
            else:
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                )

        if graph.number_of_edges() == 0:
            communities = [
                {node}
                for node in graph.nodes
            ]
        else:
            communities = list(
                nx.community.greedy_modularity_communities(
                    graph,
                    weight="weight",
                )
            )

        palette = [
            pg.intColor(
                index,
                hues=max(len(communities), 1),
                alpha=220,
            )
            for index in range(max(len(communities), 1))
        ]

        brushes: list[object] = [
            pg.mkBrush(180, 180, 180, 220)
            for _ in range(self.number_of_nodes)
        ]

        for community_index, community in enumerate(
            communities
        ):
            brush = pg.mkBrush(
                palette[community_index]
            )

            for node in community:
                brushes[int(node)] = brush

        return brushes

    def _node_sizes(
        self,
        edges: list[tuple[int, int, float]],
    ) -> FloatArray:
        strength = np.zeros(
            self.number_of_nodes,
            dtype=np.float64,
        )

        for source, _, weight in edges:
            strength[source] += weight

        maximum = float(np.max(strength))

        if maximum <= 0.0:
            return np.full(
                self.number_of_nodes,
                18.0,
            )

        return 16.0 + 20.0 * strength / maximum

    def _ensure_edge_items(
        self,
        edge_count: int,
    ) -> None:
        while len(self.edge_items) < edge_count:
            item = pg.PlotDataItem()
            item.setZValue(1)
            self.plot_widget.addItem(item)
            self.edge_items.append(item)

        for index, item in enumerate(self.edge_items):
            item.setVisible(index < edge_count)

    def _render_edges(
        self,
        edges: list[tuple[int, int, float]],
    ) -> None:
        self._ensure_edge_items(len(edges))

        if not edges:
            return

        weights = np.asarray(
            [weight for _, _, weight in edges],
            dtype=np.float64,
        )

        minimum = float(np.min(weights))
        maximum = float(np.max(weights))

        if np.isclose(minimum, maximum):
            normalised = np.ones_like(weights)
        else:
            normalised = (
                (weights - minimum)
                / (maximum - minimum)
            )

        for edge_index, (
            source,
            target,
            _,
        ) in enumerate(edges):
            strength = float(normalised[edge_index])

            width = 0.7 + 6.0 * strength**1.3
            alpha = int(45 + 190 * strength)

            item = self.edge_items[edge_index]

            item.setData(
                x=[
                    float(self.positions[source, 0]),
                    float(self.positions[target, 0]),
                ],
                y=[
                    float(self.positions[source, 1]),
                    float(self.positions[target, 1]),
                ],
                pen=pg.mkPen(
                    180,
                    190,
                    205,
                    alpha,
                    width=width,
                ),
            )

    def _render_nodes(
        self,
        edges: list[tuple[int, int, float]],
    ) -> None:
        brushes = self._community_colours(edges)
        sizes = self._node_sizes(edges)

        spots = [
            {
                "pos": (
                    float(self.positions[index, 0]),
                    float(self.positions[index, 1]),
                ),
                "size": float(sizes[index]),
                "brush": brushes[index],
                "data": self.data.names[index],
            }
            for index in range(self.number_of_nodes)
        ]

        self.node_item.setData(spots=spots)

        for index, label in enumerate(self.label_items):
            label.setPos(
                float(self.positions[index, 0]),
                float(self.positions[index, 1]),
            )

    def _update_statistics(
        self,
        edges: list[tuple[int, int, float]],
    ) -> None:
        matrix = self.display_matrix

        off_diagonal = matrix[
            ~np.eye(matrix.shape[0], dtype=bool)
        ]

        positive = off_diagonal[
            off_diagonal > 0.0
        ]

        mean_mi = (
            float(np.mean(positive))
            if positive.size
            else 0.0
        )

        maximum_mi = (
            max(
                (weight for _, _, weight in edges),
                default=0.0,
            )
        )

        strongest = sorted(
            edges,
            key=lambda edge: edge[2],
            reverse=True,
        )[:10]

        lines = [
            "INFORMATION NETWORK",
            "",
            (
                f"Window: {self.current_window + 1}/"
                f"{self.number_of_windows}"
            ),
            (
                "Range: "
                + _window_description(
                    self.data,
                    self.current_window,
                )
            ),
            "",
            f"Nodes: {self.number_of_nodes}",
            f"Visible directed edges: {len(edges)}",
            f"Mean matrix MI: {mean_mi:.6f}",
            f"Strongest visible MI: {maximum_mi:.6f}",
            "",
            "STRONGEST RELATIONSHIPS",
            "",
        ]

        for source, target, weight in strongest:
            lines.append(
                f"{self.data.names[source]:>5} → "
                f"{self.data.names[target]:<5} "
                f"{weight:.6f}"
            )

        self.stats_box.setPlainText("\n".join(lines))

    def _advance_window_if_needed(self, frame_seconds: float) -> None:
        print("tick", self.playing)

        if not self.playing:
            return

        self.elapsed_since_window_change += frame_seconds

        interval = float(self.speed_box.value())

        if self.elapsed_since_window_change < interval:
            return

        self.elapsed_since_window_change = 0.0

        next_window = (
            self.current_window + 1
        ) % self.number_of_windows

        print("current =", self.current_window)
        self._set_target_window(next_window)

    def _tick(self) -> None:
        frame_seconds = 0.016

        self._advance_window_if_needed(frame_seconds)
        self._update_transition(frame_seconds)

        directed_edges = self._pruned_edges(
            self.display_matrix
        )
        spring_edges = self._spring_edges(
            directed_edges
        )

        for _ in range(self.config.physics_substeps):
            self._physics_step(spring_edges)

        self._render_edges(directed_edges)
        self._render_nodes(directed_edges)
        self._update_statistics(directed_edges)


def run_information_network_simulator(
    data: RollingInformationData,
    config: SimulatorConfig,
) -> None:
    app = QtWidgets.QApplication.instance()

    owns_application = app is None

    if app is None:
        app = QtWidgets.QApplication([])

    simulator = InformationNetworkSimulator(
        data=data,
        config=config,
    )
    simulator.show()

    if owns_application:
        app.exec()