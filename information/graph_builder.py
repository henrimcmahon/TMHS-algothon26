from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class InformationGraphConfig:
    """
    Controls how an MI matrix is converted into a readable graph.

    percentile:
        Remove edges below this percentile of positive off-diagonal MI values.

    top_k:
        Retain at most the k strongest outgoing relationships per ticker.
        Set to None to disable.

    spring_exponent:
        Spring strength = MI ** spring_exponent. Values above 1 emphasise
        stronger relationships and suppress weak ones.
    """

    percentile: float = 90.0
    top_k: int | None = 4
    spring_exponent: float = 1.5
    minimum_mi: float = 0.0
    directed: bool = True


def _validate_inputs(
    matrix: FloatMatrix,
    names: list[str],
    config: InformationGraphConfig,
) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("MI matrix must be square.")

    if matrix.shape[0] != len(names):
        raise ValueError(
            f"Matrix contains {matrix.shape[0]} instruments, "
            f"but {len(names)} names were supplied."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("MI matrix contains non-finite values.")

    if not 0.0 <= config.percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    if config.top_k is not None and config.top_k < 1:
        raise ValueError("top_k must be positive or None.")

    if config.spring_exponent <= 0.0:
        raise ValueError("spring_exponent must be positive.")


def _edge_threshold(
    matrix: FloatMatrix,
    config: InformationGraphConfig,
) -> float:
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    positive = off_diagonal[off_diagonal > config.minimum_mi]

    if positive.size == 0:
        return float("inf")

    percentile_threshold = float(
        np.percentile(positive, config.percentile)
    )

    return max(config.minimum_mi, percentile_threshold)


def _candidate_targets(
    matrix: FloatMatrix,
    source_index: int,
    threshold: float,
    top_k: int | None,
) -> NDArray[np.int64]:
    row = matrix[source_index].copy()
    row[source_index] = -np.inf

    candidates = np.flatnonzero(row >= threshold)

    if top_k is None or candidates.size <= top_k:
        return candidates.astype(np.int64)

    ordered = candidates[np.argsort(row[candidates])[::-1]]
    return ordered[:top_k].astype(np.int64)


def build_information_graph(
    matrix: FloatMatrix,
    names: list[str],
    config: InformationGraphConfig | None = None,
) -> nx.Graph | nx.DiGraph:
    """
    Convert a mutual-information matrix into a sparse weighted graph.

    For a directed lagged-MI matrix, matrix[i, j] represents:

        ticker i at time t  ->  ticker j at time t + lag
    """
    config = config or InformationGraphConfig()

    matrix = np.asarray(matrix, dtype=np.float64)
    _validate_inputs(matrix, names, config)

    graph: nx.Graph | nx.DiGraph
    graph = nx.DiGraph() if config.directed else nx.Graph()

    graph.add_nodes_from(names)

    threshold = _edge_threshold(matrix, config)

    for source_index, source_name in enumerate(names):
        targets = _candidate_targets(
            matrix=matrix,
            source_index=source_index,
            threshold=threshold,
            top_k=config.top_k,
        )

        for target_index in targets:
            target_name = names[int(target_index)]
            mutual_information = float(
                matrix[source_index, target_index]
            )

            if not config.directed and graph.has_edge(
                source_name,
                target_name,
            ):
                existing = float(
                    graph[source_name][target_name]["mi"]
                )

                if mutual_information <= existing:
                    continue

            graph.add_edge(
                source_name,
                target_name,
                mi=mutual_information,
                spring_weight=mutual_information
                ** config.spring_exponent,
            )

    graph.graph["threshold"] = threshold
    graph.graph["maximum_mi"] = max(
        (
            float(attributes["mi"])
            for _, _, attributes in graph.edges(data=True)
        ),
        default=0.0,
    )

    return graph


def undirected_community_graph(
    graph: nx.Graph | nx.DiGraph,
) -> nx.Graph:
    """
    Symmetrise a directed information graph for community detection.

    Reciprocal directed edges are combined by summing their MI.
    """
    if not graph.is_directed():
        return graph.copy()

    result = nx.Graph()
    result.add_nodes_from(graph.nodes)

    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes["mi"])

        if result.has_edge(source, target):
            result[source][target]["mi"] += weight
            result[source][target]["spring_weight"] += float(
                attributes["spring_weight"]
            )
        else:
            result.add_edge(
                source,
                target,
                mi=weight,
                spring_weight=float(attributes["spring_weight"]),
            )

    return result


def detect_communities(
    graph: nx.Graph | nx.DiGraph,
) -> list[set[str]]:
    undirected = undirected_community_graph(graph)

    if undirected.number_of_edges() == 0:
        return [{str(node)} for node in undirected.nodes]

    communities = nx.community.greedy_modularity_communities(
        undirected,
        weight="mi",
    )

    return [set(community) for community in communities]