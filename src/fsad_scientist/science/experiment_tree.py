from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExperimentPhase(StrEnum):
    FEASIBILITY = "feasibility"
    SENSITIVITY = "sensitivity"
    MAIN_STUDY = "main_study"
    REPLICATION = "replication"
    ABLATION = "ablation"
    CROSS_DATASET = "cross_dataset"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass(slots=True)
class ExperimentNode:
    id: str
    hypothesis_id: str
    phase: ExperimentPhase
    parent_id: str | None
    information_gain: float
    falsification_value: float
    estimated_cost: float
    novelty: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    config: dict[str, object] = field(default_factory=dict)
    error_history: list[str] = field(default_factory=list)

    @property
    def priority(self) -> float:
        cost = max(self.estimated_cost, 0.01)
        return (self.information_gain * self.falsification_value) / cost + 0.05 * self.novelty


class ProgressiveExperimentTree:
    """Best-first experiment tree with explicit phase and failure state."""

    def __init__(self, *, max_nodes: int, max_debug_depth: int = 2) -> None:
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes
        self.max_debug_depth = max_debug_depth
        self.nodes: dict[str, ExperimentNode] = {}
        self.children: dict[str | None, list[str]] = {}

    def add(self, node: ExperimentNode) -> None:
        if len(self.nodes) >= self.max_nodes:
            raise RuntimeError("experiment tree node budget exhausted")
        if node.id in self.nodes:
            raise ValueError(f"duplicate experiment node: {node.id}")
        if node.parent_id is not None and node.parent_id not in self.nodes:
            raise KeyError(f"unknown parent node: {node.parent_id}")
        self.nodes[node.id] = node
        self.children.setdefault(node.parent_id, []).append(node.id)

    def select_next(self) -> ExperimentNode | None:
        candidates = [node for node in self.nodes.values() if node.status == NodeStatus.PENDING]
        if not candidates:
            return None
        return sorted(candidates, key=lambda node: (-node.priority, node.id))[0]

    def record_failure(self, node_id: str, error: str) -> NodeStatus:
        node = self.nodes[node_id]
        node.error_history.append(error)
        node.status = (
            NodeStatus.PENDING
            if len(node.error_history) <= self.max_debug_depth
            else NodeStatus.FAILED
        )
        return node.status

    def mark_running(self, node_id: str) -> None:
        self.nodes[node_id].status = NodeStatus.RUNNING

    def mark_succeeded(self, node_id: str) -> None:
        self.nodes[node_id].status = NodeStatus.SUCCEEDED

    def prune_descendants(self, node_id: str) -> list[str]:
        pruned: list[str] = []
        frontier = list(self.children.get(node_id, []))
        while frontier:
            child_id = frontier.pop()
            child = self.nodes[child_id]
            if child.status == NodeStatus.PENDING:
                child.status = NodeStatus.PRUNED
                pruned.append(child_id)
            frontier.extend(self.children.get(child_id, []))
        return sorted(pruned)

