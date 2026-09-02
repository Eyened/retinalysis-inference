"""Structured artery/vein inference on a noisy vessel skeleton.

The observed skeleton is split into segments around configurable junction regions.
Each junction proposes partitions of its ports into terminations, continuations, or
bifurcations. A mixed-integer program jointly selects segment labels and junction
configurations while constraining the artery and vein graphs to rooted forests.

Spline fitting follows the arclength-parameterized ``UnivariateSpline`` approach
used by ``rtnls_vascx.vascx.shared.splines.SplineInterpolation``. Thickness is
sampled from the vessel-mask distance transform outside junction regions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from itertools import combinations, product
from math import acos, expm1, inf, log, pi

import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from skimage.measure import label as connected_labels
from skimage.morphology import binary_dilation, disk, skeletonize

ARTERY = 1
VEIN = 2
DISCARD = 0
MIXED = 3
_EPSILON = 1e-6


@dataclass(frozen=True)
class AVInferenceConfig:
    """Configuration for structured segment-level AV inference.

    Every cost has its own weight. Set a weight to zero to disable that term for
    an ablation without changing graph extraction or solver constraints.
    """

    vessel_threshold: float = 0.5
    crossing_threshold: float = 0.5
    junction_radius: int = 4
    min_segment_size: int = 8
    max_ports_per_junction: int = 6
    max_configurations_per_junction: int = 24
    enable_compound_bifurcations: bool = True
    enable_mixed_segments: bool = True
    mixed_av_confidence_threshold: float = 0.65
    mixed_av_min_fraction: float = 0.15
    mixed_crossing_fraction_threshold: float = 0.2
    mixed_segment_cost: float = 2.5
    mixed_evidence_cost_weight: float = 1.0
    mixed_single_class_cost_weight: float = 5.0
    spline_error_fraction: float = 0.05
    spline_fit_length: float = 12.0
    thickness_exponent: float = 2.5
    vessel_cost_weight: float = 1.0
    av_cost_weight: float = 1.0
    direction_cost_weight: float = 1.0
    max_bifurcation_deflection_degrees: float = 100.0
    max_continuation_deflection_degrees: float = 100.0
    continuation_angle_cost_at_90_degrees: float = 10.0
    continuation_angle_cost_sharpness: float = 10.0
    thickness_cost_weight: float = 0.5
    crossing_cost_weight: float = 1.0
    av_compatibility_cost_weight: float = 0.5
    disc_root_radius: int = 6
    bounds_boundary_width: int = 32
    disc_root_penalty: float = 0.0
    bounds_root_penalty: float = 4.0
    internal_root_penalty: float = 8.0
    root_distance_cost_weight: float = 1.0
    root_cost_weight: float = 2.0
    fallback_root_subtree_cost_weight: float = 0.5
    solver_time_limit: float = 10.0
    solver_mip_rel_gap: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.vessel_threshold <= 1:
            raise ValueError("vessel_threshold must be in [0, 1]")
        if not 0 <= self.crossing_threshold <= 1:
            raise ValueError("crossing_threshold must be in [0, 1]")
        if self.junction_radius < 1:
            raise ValueError("junction_radius must be positive")
        if self.min_segment_size < 2:
            raise ValueError("min_segment_size must be at least 2")
        if self.max_ports_per_junction < 3:
            raise ValueError("max_ports_per_junction must be at least 3")
        if self.max_configurations_per_junction < 1:
            raise ValueError("max_configurations_per_junction must be positive")
        if not 0.5 <= self.mixed_av_confidence_threshold <= 1:
            raise ValueError("mixed_av_confidence_threshold must be in [0.5, 1]")
        if not 0 <= self.mixed_av_min_fraction <= 0.5:
            raise ValueError("mixed_av_min_fraction must be in [0, 0.5]")
        if not 0 <= self.mixed_crossing_fraction_threshold <= 1:
            raise ValueError("mixed_crossing_fraction_threshold must be in [0, 1]")
        if self.spline_fit_length <= 0:
            raise ValueError("spline_fit_length must be positive")
        if self.thickness_exponent <= 0:
            raise ValueError("thickness_exponent must be positive")
        for name in (
            "max_bifurcation_deflection_degrees",
            "max_continuation_deflection_degrees",
        ):
            if not 0 < getattr(self, name) <= 180:
                raise ValueError(f"{name} must be in (0, 180]")
        if self.continuation_angle_cost_sharpness <= 0:
            raise ValueError("continuation_angle_cost_sharpness must be positive")
        if self.disc_root_radius < 0:
            raise ValueError("disc_root_radius cannot be negative")
        if self.bounds_boundary_width < 1:
            raise ValueError("bounds_boundary_width must be positive")
        for name in (
            "vessel_cost_weight",
            "av_cost_weight",
            "direction_cost_weight",
            "continuation_angle_cost_at_90_degrees",
            "thickness_cost_weight",
            "crossing_cost_weight",
            "av_compatibility_cost_weight",
            "mixed_segment_cost",
            "mixed_evidence_cost_weight",
            "mixed_single_class_cost_weight",
            "disc_root_penalty",
            "bounds_root_penalty",
            "internal_root_penalty",
            "root_distance_cost_weight",
            "root_cost_weight",
            "fallback_root_subtree_cost_weight",
            "solver_time_limit",
            "solver_mip_rel_gap",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class SegmentSpline:
    """Arclength-parameterized spline fitted to ordered row/column points."""

    points: np.ndarray
    row_spline: UnivariateSpline
    col_spline: UnivariateSpline
    total_distance: float

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        *,
        error_fraction: float,
    ) -> SegmentSpline | None:
        points = np.asarray(points, dtype=float) + 0.5
        if len(points) < 4:
            return None
        step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate(([True], step_lengths > 0))
        points = points[keep]
        if len(points) < 4:
            return None
        step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distance = np.concatenate(([0.0], np.cumsum(step_lengths)))
        total_distance = float(distance[-1])
        if total_distance <= 0:
            return None
        parameter = distance / total_distance
        degree = min(3, len(points) - 1)
        smoothing_scale = max(error_fraction, _EPSILON) / 0.05

        def fit_axis(values: np.ndarray) -> UnivariateSpline:
            variance = float(np.mean(np.diff(values) ** 2) / 2.0)
            smoothing = variance * len(points) * smoothing_scale
            return UnivariateSpline(parameter, values, k=degree, s=smoothing)

        row_spline = fit_axis(points[:, 0])
        col_spline = fit_axis(points[:, 1])
        if not (
            np.isfinite(row_spline(parameter)).all()
            and np.isfinite(col_spline(parameter)).all()
        ):
            return None
        return cls(points, row_spline, col_spline, total_distance)

    def point(self, parameter: float) -> np.ndarray:
        parameter = float(np.clip(parameter, 0.0, 1.0))
        return np.asarray(
            [self.row_spline(parameter), self.col_spline(parameter)], dtype=float
        )

    def direction_from_endpoint(self, endpoint: int, distance: float) -> np.ndarray:
        fraction = min(float(distance) / self.total_distance, 1.0)
        if endpoint == 0:
            direction = self.point(fraction) - self.point(0.0)
        else:
            direction = self.point(1.0 - fraction) - self.point(1.0)
        return _unit_vector(direction)


@dataclass
class SegmentInfo:
    id: int
    skeleton: np.ndarray
    pixels: np.ndarray
    spline: SegmentSpline | None
    thickness: float
    vessel_probability: float
    artery_probability: float
    vein_probability: float
    artery_dominant_fraction: float
    vein_dominant_fraction: float
    crossing_fraction: float
    is_microsegment: bool
    junction_contact_count: int

    @property
    def av_log_odds(self) -> float:
        return float(
            log(self.artery_probability + _EPSILON)
            - log(self.vein_probability + _EPSILON)
        )

    @property
    def initial_label(self) -> int:
        return ARTERY if self.artery_probability >= self.vein_probability else VEIN


@dataclass(frozen=True)
class Port:
    id: int
    junction_id: int
    segment_id: int
    endpoint: int
    position: tuple[float, float]
    tangent: tuple[float, float]
    thickness: float


@dataclass(frozen=True)
class LinkSpec:
    first_segment: int
    second_segment: int
    directed: bool = False


@dataclass(frozen=True)
class PortGroup:
    port_ids: tuple[int, ...]
    parent_port_id: int | None = None
    continuation_port_id: int | None = None


@dataclass(frozen=True)
class JunctionConfiguration:
    index: int
    groups: tuple[PortGroup, ...]
    links: tuple[LinkSpec, ...]
    cost: float
    overlap: bool


@dataclass
class JunctionInfo:
    id: int
    pixels: np.ndarray
    port_ids: tuple[int, ...]
    crossing_probability: float
    configurations: tuple[JunctionConfiguration, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RootAnchor:
    """A possible source for one artery or vein tree component.

    ``position`` is the graph attachment point. For a junction-propagated disc
    root, ``contact_position`` is the reached skeleton pixel beside the OD.
    """

    id: int
    segment_id: int
    endpoint: int | None
    kind: str
    distance: float
    port_id: int | None = None
    junction_id: int | None = None
    position: tuple[float, float] | None = None
    contact_position: tuple[float, float] | None = None
    path_distance: float = 0.0


@dataclass
class AVInferenceProblem:
    logits: np.ndarray
    vessel: np.ndarray
    vessel_probability: np.ndarray
    crossing_probability: np.ndarray
    segment_image: np.ndarray
    nearest_segments: np.ndarray
    junction_image: np.ndarray
    segments: dict[int, SegmentInfo]
    ports: dict[int, Port]
    junctions: dict[int, JunctionInfo]
    root_anchors: dict[int, tuple[RootAnchor, ...]]


@dataclass
class AVInferenceResult:
    mask: np.ndarray
    changed_segment_mask: np.ndarray
    discarded_mask: np.ndarray
    segment_id_image: np.ndarray
    segment_labels: dict[int, int]
    mixed_segment_ids: set[int]
    microsegment_ids: set[int]
    selected_configurations: dict[int, int]
    forest_edges: dict[int, list[tuple[int, int]]]
    selected_roots: dict[int, list[RootAnchor]]
    objective: float | None
    solver_status: str


@dataclass
class _VariableRegistry:
    costs: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    integrality: list[int] = field(default_factory=list)

    def add(
        self,
        cost: float = 0.0,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        integer: bool = True,
    ) -> int:
        index = len(self.costs)
        self.costs.append(float(cost))
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.integrality.append(1 if integer else 0)
        return index


@dataclass
class _ConstraintBuilder:
    rows: list[dict[int, float]] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)

    def add(self, values: dict[int, float], lower: float, upper: float) -> None:
        self.rows.append({key: value for key, value in values.items() if value})
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def linear_constraint(self, n_variables: int) -> LinearConstraint:
        row_indices: list[int] = []
        col_indices: list[int] = []
        data: list[float] = []
        for row_index, values in enumerate(self.rows):
            for col_index, value in values.items():
                row_indices.append(row_index)
                col_indices.append(col_index)
                data.append(value)
        matrix = coo_array(
            (data, (row_indices, col_indices)),
            shape=(len(self.rows), n_variables),
        ).tocsc()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def _av_probabilities(logits: np.ndarray) -> np.ndarray:
    av_logits = logits[..., 1:3]
    shifted = av_logits - av_logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _unit_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= _EPSILON:
        return np.zeros(2, dtype=float)
    return value / norm


def _angle(first: Iterable[float], second: Iterable[float]) -> float:
    first_vector = _unit_vector(np.asarray(first, dtype=float))
    second_vector = _unit_vector(np.asarray(second, dtype=float))
    if not np.any(first_vector) or not np.any(second_vector):
        return pi
    return acos(float(np.clip(np.dot(first_vector, second_vector), -1.0, 1.0)))


def _order_skeleton_points(coordinates: np.ndarray) -> np.ndarray:
    """Order one junction-free 8-connected skeleton component as a path."""
    nodes = {tuple(map(int, point)) for point in np.asarray(coordinates)}
    if len(nodes) <= 1:
        return np.asarray(sorted(nodes), dtype=np.int32)
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in nodes:
        adjacent = [
            (row + drow, col + dcol)
            for drow in (-1, 0, 1)
            for dcol in (-1, 0, 1)
            if (drow or dcol) and (row + drow, col + dcol) in nodes
        ]
        neighbours[(row, col)] = sorted(adjacent)
    endpoints = sorted(node for node in nodes if len(neighbours[node]) <= 1)
    current = endpoints[0] if endpoints else min(nodes)
    ordered = [current]
    visited = {current}
    previous: tuple[int, int] | None = None
    while len(visited) < len(nodes):
        candidates = [node for node in neighbours[current] if node not in visited]
        if not candidates:
            remaining = nodes - visited
            if not remaining:
                break
            next_node = min(
                remaining,
                key=lambda node: (
                    (node[0] - current[0]) ** 2 + (node[1] - current[1]) ** 2
                ),
            )
        elif len(candidates) == 1 or previous is None:
            next_node = candidates[0]
        else:
            incoming = _unit_vector(np.subtract(current, previous))
            next_node = max(
                candidates,
                key=lambda node: float(
                    np.dot(incoming, _unit_vector(np.subtract(node, current)))
                ),
            )
        previous, current = current, next_node
        ordered.append(current)
        visited.add(current)
    return np.asarray(ordered, dtype=np.int32)


def _segment_direction(
    segment: SegmentInfo,
    endpoint: int,
    fit_length: float,
) -> np.ndarray:
    if segment.is_microsegment:
        if endpoint == 0:
            return _unit_vector(segment.skeleton[-1] - segment.skeleton[0])
        return _unit_vector(segment.skeleton[0] - segment.skeleton[-1])
    if segment.spline is not None:
        direction = segment.spline.direction_from_endpoint(endpoint, fit_length)
        if np.any(direction):
            return direction
    points = segment.skeleton
    if len(points) < 2:
        return np.zeros(2, dtype=float)
    sample_index = min(len(points) - 1, max(1, round(fit_length)))
    if endpoint == 0:
        return _unit_vector(points[sample_index] - points[0])
    return _unit_vector(points[-1 - sample_index] - points[-1])


def _restricted_partitions(
    values: tuple[int, ...],
    *,
    max_group_size: int = 3,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if not values:
        yield ()
        return
    first, remaining = values[0], values[1:]
    for group_size in range(1, min(max_group_size, 4) + 1):
        if group_size > len(values):
            continue
        for partners in combinations(remaining, group_size - 1):
            group = tuple(sorted((first, *partners)))
            partner_set = set(partners)
            rest = tuple(value for value in remaining if value not in partner_set)
            for partition in _restricted_partitions(
                rest, max_group_size=max_group_size
            ):
                yield (group, *partition)


def _configuration_groups(
    partition: tuple[tuple[int, ...], ...],
) -> Iterable[tuple[PortGroup, ...]]:
    group_options: list[tuple[PortGroup, ...]] = []
    for group in partition:
        if len(group) == 3:
            group_options.append(
                tuple(PortGroup(group, parent_port_id) for parent_port_id in group)
            )
        elif len(group) == 4:
            group_options.append(tuple(_compound_groups(group)))
        else:
            group_options.append((PortGroup(group),))
    yield from product(*group_options)


def _compound_groups(port_ids: tuple[int, ...]) -> Iterable[PortGroup]:
    """Enumerate oriented two-bifurcation interpretations of four ports."""
    if len(port_ids) != 4:
        return
    for parent_port_id in port_ids:
        for continuation_port_id in port_ids:
            if continuation_port_id == parent_port_id:
                continue
            yield PortGroup(port_ids, parent_port_id, continuation_port_id)


def _group_links(group: PortGroup, ports: dict[int, Port]) -> tuple[LinkSpec, ...]:
    if len(group.port_ids) == 1:
        return ()
    segment_ids = [ports[port_id].segment_id for port_id in group.port_ids]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("A connected port group cannot contain one segment twice")
    if len(group.port_ids) == 2:
        return (LinkSpec(segment_ids[0], segment_ids[1], directed=False),)
    parent_segment = ports[int(group.parent_port_id)].segment_id
    return tuple(
        LinkSpec(parent_segment, segment_id, directed=True)
        for segment_id in segment_ids
        if segment_id != parent_segment
    )


def _directed_deflection(
    parent_tangent: np.ndarray,
    child_tangent: np.ndarray,
) -> float:
    """Return the local turn from the incoming parent to an outgoing child."""
    return _angle(-np.asarray(parent_tangent), np.asarray(child_tangent))


def _bifurcation_direction_cost(
    parent_tangent: np.ndarray,
    child_tangents: tuple[np.ndarray, np.ndarray],
    config: AVInferenceConfig,
) -> float:
    deflections = [
        _directed_deflection(parent_tangent, child_tangent)
        for child_tangent in child_tangents
    ]
    maximum = config.max_bifurcation_deflection_degrees * pi / 180.0
    if any(deflection > maximum + _EPSILON for deflection in deflections):
        return inf
    children_direction = _unit_vector(child_tangents[0] + child_tangents[1])
    trunk_cost = (_angle(-parent_tangent, children_direction) / pi) ** 2
    child_cost = np.mean([(deflection / pi) ** 2 for deflection in deflections])
    return float(trunk_cost + 0.5 * child_cost)


def _continuation_direction_cost(
    first_tangent: np.ndarray,
    second_tangent: np.ndarray,
    config: AVInferenceConfig,
) -> float:
    """Return a steep local bend cost for a two-port continuation."""
    deflection = pi - _angle(first_tangent, second_tangent)
    maximum = config.max_continuation_deflection_degrees * pi / 180.0
    if deflection > maximum + _EPSILON:
        return inf
    normalized = deflection / (pi / 2.0)
    sharpness = config.continuation_angle_cost_sharpness
    return float(
        config.continuation_angle_cost_at_90_degrees
        * expm1(sharpness * normalized**2)
        / expm1(sharpness)
    )


def _direction_cost(
    group: PortGroup,
    ports: dict[int, Port],
    config: AVInferenceConfig,
) -> float:
    group_ports = [ports[port_id] for port_id in group.port_ids]
    if len(group_ports) == 1:
        return 0.0
    if len(group_ports) == 2:
        return _continuation_direction_cost(
            np.asarray(group_ports[0].tangent),
            np.asarray(group_ports[1].tangent),
            config,
        )
    parent = ports[int(group.parent_port_id)]
    children = [port for port in group_ports if port.id != parent.id]
    if len(children) != 2:
        raise ValueError("Ordinary bifurcations require exactly three ports")
    return _bifurcation_direction_cost(
        np.asarray(parent.tangent),
        tuple(np.asarray(child.tangent) for child in children),
        config,
    )


def _thickness_cost(
    group: PortGroup,
    ports: dict[int, Port],
    exponent: float,
) -> float:
    group_ports = [ports[port_id] for port_id in group.port_ids]
    if len(group_ports) == 1:
        return 0.0
    if len(group_ports) == 2:
        ratio = (group_ports[0].thickness + _EPSILON) / (
            group_ports[1].thickness + _EPSILON
        )
        return float(log(ratio) ** 2)
    parent = ports[int(group.parent_port_id)]
    children = [port for port in group_ports if port.id != parent.id]
    parent_power = parent.thickness**exponent
    children_power = sum(child.thickness**exponent for child in children)
    return float(log((parent_power + _EPSILON) / (children_power + _EPSILON)) ** 2)


def _av_compatibility_cost(
    group: PortGroup,
    ports: dict[int, Port],
    segments: dict[int, SegmentInfo],
) -> float:
    if len(group.port_ids) == 1:
        return 0.0
    evidence = np.asarray(
        [
            np.tanh(segments[ports[port_id].segment_id].av_log_odds / 2.0)
            for port_id in group.port_ids
        ],
        dtype=float,
    )
    confidence = np.maximum(np.abs(evidence), 0.05)
    mean = float(np.average(evidence, weights=confidence))
    return float(np.average((evidence - mean) ** 2, weights=confidence))


def _compound_group_cost(
    group: PortGroup,
    ports: dict[int, Port],
    segments: dict[int, SegmentInfo],
    config: AVInferenceConfig,
) -> float:
    """Score four ports as two sequential bifurcations on one trunk."""
    if len(group.port_ids) != 4 or group.continuation_port_id is None:
        raise ValueError("Compound bifurcations require four ports and a continuation")
    parent = ports[int(group.parent_port_id)]
    continuation = ports[group.continuation_port_id]
    branches = [
        ports[port_id]
        for port_id in group.port_ids
        if port_id not in {parent.id, continuation.id}
    ]
    if len(branches) != 2:
        raise ValueError("Compound bifurcations require two side branches")

    parent_tangent = np.asarray(parent.tangent)
    continuation_tangent = np.asarray(continuation.tangent)
    continuation_cost = _continuation_direction_cost(
        parent_tangent, continuation_tangent, config
    )
    if not np.isfinite(continuation_cost):
        return inf
    trunk_direction = _unit_vector(-parent_tangent + continuation_tangent)
    if not np.any(trunk_direction):
        trunk_direction = _unit_vector(-parent_tangent)
    if not np.any(trunk_direction):
        trunk_direction = _unit_vector(continuation_tangent)

    parent_power = parent.thickness**config.thickness_exponent
    continuation_power = continuation.thickness**config.thickness_exponent
    ordered_costs = []
    for first_branch, second_branch in (branches, branches[::-1]):
        first_tangent = np.asarray(first_branch.tangent)
        second_tangent = np.asarray(second_branch.tangent)
        direction_cost = (
            continuation_cost
            + _bifurcation_direction_cost(
                parent_tangent,
                (first_tangent, trunk_direction),
                config,
            )
            + _bifurcation_direction_cost(
                -trunk_direction,
                (second_tangent, continuation_tangent),
                config,
            )
        )
        if not np.isfinite(direction_cost):
            continue

        first_power = first_branch.thickness**config.thickness_exponent
        second_power = second_branch.thickness**config.thickness_exponent
        remaining_power = max(parent_power - first_power, _EPSILON)
        first_balance = (
            log((parent_power + _EPSILON) / (first_power + remaining_power + _EPSILON))
            ** 2
        )
        second_balance = (
            log(
                (remaining_power + _EPSILON)
                / (second_power + continuation_power + _EPSILON)
            )
            ** 2
        )
        thickness_cost = first_balance + second_balance
        ordered_costs.append(
            config.direction_cost_weight * direction_cost
            + config.thickness_cost_weight * thickness_cost
        )

    if not ordered_costs:
        return inf
    av_cost = _av_compatibility_cost(group, ports, segments)
    return float(min(ordered_costs) + config.av_compatibility_cost_weight * av_cost)


def _binary_nll(value: bool, probability: float) -> float:
    probability = float(np.clip(probability, _EPSILON, 1.0 - _EPSILON))
    return -log(probability if value else 1.0 - probability)


def _make_configurations(
    junction: JunctionInfo,
    ports: dict[int, Port],
    segments: dict[int, SegmentInfo],
    config: AVInferenceConfig,
) -> tuple[JunctionConfiguration, ...]:
    port_ids = tuple(sorted(junction.port_ids))
    if len(port_ids) > config.max_ports_per_junction:
        groups = tuple(PortGroup((port_id,)) for port_id in port_ids)
        return (JunctionConfiguration(0, groups, (), 0.0, False),)

    configurations: list[JunctionConfiguration] = []
    max_group_size = 4 if config.enable_compound_bifurcations else 3
    for partition in _restricted_partitions(port_ids, max_group_size=max_group_size):
        for groups in _configuration_groups(partition):
            try:
                links = tuple(
                    link for group in groups for link in _group_links(group, ports)
                )
            except ValueError:
                continue
            non_single_groups = sum(len(group.port_ids) > 1 for group in groups)
            overlap = non_single_groups >= 2
            group_cost = 0.0
            for group in groups:
                if group.continuation_port_id is not None:
                    group_cost += _compound_group_cost(group, ports, segments, config)
                else:
                    direction_cost = _direction_cost(group, ports, config)
                    if not np.isfinite(direction_cost):
                        group_cost = inf
                        break
                    group_cost += config.direction_cost_weight * direction_cost
                    group_cost += config.thickness_cost_weight * _thickness_cost(
                        group, ports, config.thickness_exponent
                    )
                    group_cost += (
                        config.av_compatibility_cost_weight
                        * _av_compatibility_cost(group, ports, segments)
                    )
            if not np.isfinite(group_cost):
                continue
            crossing_cost = config.crossing_cost_weight * _binary_nll(
                overlap, junction.crossing_probability
            )
            configurations.append(
                JunctionConfiguration(
                    index=len(configurations),
                    groups=groups,
                    links=links,
                    cost=float(group_cost + crossing_cost),
                    overlap=overlap,
                )
            )

    configurations.sort(
        key=lambda candidate: (
            candidate.cost,
            tuple(group.port_ids for group in candidate.groups),
            tuple(
                group.parent_port_id if group.parent_port_id is not None else -1
                for group in candidate.groups
            ),
            tuple(
                group.continuation_port_id
                if group.continuation_port_id is not None
                else -1
                for group in candidate.groups
            ),
        )
    )
    all_singletons = next(
        (
            candidate
            for candidate in configurations
            if all(len(group.port_ids) == 1 for group in candidate.groups)
        ),
        None,
    )
    best_compound = next(
        (
            candidate
            for candidate in configurations
            if any(group.continuation_port_id is not None for group in candidate.groups)
        ),
        None,
    )
    retained = configurations[: config.max_configurations_per_junction]
    if all_singletons is not None and all_singletons not in retained:
        retained = [*retained[:-1], all_singletons]
    if (
        best_compound is not None
        and best_compound not in retained
        and len(retained) > 1
    ):
        replace_index = next(
            (
                index
                for index in range(len(retained) - 1, -1, -1)
                if retained[index] != all_singletons
            ),
            None,
        )
        if replace_index is not None:
            retained[replace_index] = best_compound
    retained.sort(key=lambda candidate: configurations.index(candidate))
    return tuple(
        JunctionConfiguration(
            index=index,
            groups=candidate.groups,
            links=candidate.links,
            cost=candidate.cost,
            overlap=candidate.overlap,
        )
        for index, candidate in enumerate(retained)
    )


def _binary_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
    name: str,
    *,
    default: bool,
) -> np.ndarray:
    if mask is None:
        return np.full(shape, default, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} differs from logits {shape}")
    return result


def _inside_boundary_distance(bounds_mask: np.ndarray) -> np.ndarray:
    """Distance to the fundus-mask exterior, including the image border."""
    padded = np.pad(bounds_mask, 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1]


def _skeleton_geodesic_from_seeds(
    skeleton: np.ndarray,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 8-connected distances and the seed reached by each pixel."""
    distances = np.full(skeleton.shape, np.inf, dtype=np.float32)
    source_rows = np.full(skeleton.shape, -1, dtype=np.int32)
    source_cols = np.full(skeleton.shape, -1, dtype=np.int32)
    queue: list[tuple[float, int, int]] = []
    for row, col in np.argwhere(skeleton & seeds):
        row, col = int(row), int(col)
        distances[row, col] = 0.0
        source_rows[row, col] = row
        source_cols[row, col] = col
        heappush(queue, (0.0, row, col))

    neighbours = tuple(
        (row_offset, col_offset, float(np.hypot(row_offset, col_offset)))
        for row_offset in (-1, 0, 1)
        for col_offset in (-1, 0, 1)
        if row_offset or col_offset
    )
    height, width = skeleton.shape
    while queue:
        distance, row, col = heappop(queue)
        if distance > distances[row, col] + _EPSILON:
            continue
        for row_offset, col_offset, step in neighbours:
            next_row, next_col = row + row_offset, col + col_offset
            if not (
                0 <= next_row < height
                and 0 <= next_col < width
                and skeleton[next_row, next_col]
            ):
                continue
            next_distance = distance + step
            if next_distance + _EPSILON >= distances[next_row, next_col]:
                continue
            distances[next_row, next_col] = next_distance
            source_rows[next_row, next_col] = source_rows[row, col]
            source_cols[next_row, next_col] = source_cols[row, col]
            heappush(queue, (next_distance, next_row, next_col))
    return distances, source_rows, source_cols


def _junction_disc_contacts(
    skeleton: np.ndarray,
    junction_image: np.ndarray,
    segments: dict[int, SegmentInfo],
    ports: dict[int, Port],
    junctions: dict[int, JunctionInfo],
    disc_distance: np.ndarray,
    config: AVInferenceConfig,
) -> dict[tuple[int, int], tuple[Port, tuple[int, int], float]]:
    """Find ports that retain a skeleton path to the OD through a junction.

    Junction pixels are removed before segments are formed. This calculation uses
    the original skeleton so that removing a large junction cannot erase an
    anatomical optic-disc contact. The returned path distance is diagnostic; the
    root cost uses the source pixel's actual distance to the disc.
    """
    contacts: dict[tuple[int, int], tuple[Port, tuple[int, int], float]] = {}
    disc_contact = disc_distance <= config.disc_root_radius
    if not junctions or not np.any(disc_contact):
        return contacts
    square_kernel = np.ones((3, 3), dtype=bool)

    for junction_id, junction in junctions.items():
        region = junction_image == junction_id
        local_region = ndimage.binary_dilation(region, structure=square_kernel)
        local_skeleton = skeleton & local_region
        if not np.any(local_skeleton & disc_contact):
            continue

        coordinates = np.argwhere(local_region)
        lower = np.maximum(coordinates.min(axis=0) - 1, 0)
        upper = np.minimum(coordinates.max(axis=0) + 2, skeleton.shape)
        row_slice = slice(int(lower[0]), int(upper[0]))
        col_slice = slice(int(lower[1]), int(upper[1]))
        local = local_skeleton[row_slice, col_slice]
        local_disc_contact = disc_contact[row_slice, col_slice]
        distances, source_rows, source_cols = _skeleton_geodesic_from_seeds(
            local, local_disc_contact
        )

        for port_id in junction.port_ids:
            port = ports[port_id]
            segment_skeleton = segments[port.segment_id].skeleton
            endpoint = (
                segment_skeleton[0] if port.endpoint == 0 else segment_skeleton[-1]
            )
            local_row = int(endpoint[0] - lower[0])
            local_col = int(endpoint[1] - lower[1])
            if not (
                0 <= local_row < local.shape[0]
                and 0 <= local_col < local.shape[1]
                and local[local_row, local_col]
            ):
                continue
            path_distance = float(distances[local_row, local_col])
            if not np.isfinite(path_distance):
                continue
            source = (
                int(source_rows[local_row, local_col] + lower[0]),
                int(source_cols[local_row, local_col] + lower[1]),
            )
            key = (port.segment_id, port.endpoint)
            previous = contacts.get(key)
            if previous is None or path_distance < previous[2]:
                contacts[key] = (port, source, path_distance)
    return contacts


def _make_root_anchors(
    segments: dict[int, SegmentInfo],
    ports: dict[int, Port],
    junctions: dict[int, JunctionInfo],
    skeleton: np.ndarray,
    junction_image: np.ndarray,
    disc_mask: np.ndarray,
    bounds_mask: np.ndarray,
    config: AVInferenceConfig,
) -> dict[int, tuple[RootAnchor, ...]]:
    """Create preferred disc, fallback bounds, and emergency internal roots."""
    disc_distance = (
        ndimage.distance_transform_edt(~disc_mask)
        if np.any(disc_mask)
        else np.full(disc_mask.shape, np.inf)
    )
    bounds_distance = _inside_boundary_distance(bounds_mask)
    junction_disc_contacts = _junction_disc_contacts(
        skeleton,
        junction_image,
        segments,
        ports,
        junctions,
        disc_distance,
        config,
    )
    port_endpoints = {(port.segment_id, port.endpoint) for port in ports.values()}
    anchors: dict[int, tuple[RootAnchor, ...]] = {}
    next_anchor_id = 1

    for segment_id, segment in sorted(segments.items()):
        if segment.is_microsegment:
            anchors[segment_id] = ()
            continue
        candidates: list[RootAnchor] = []
        for endpoint, point in ((0, segment.skeleton[0]), (1, segment.skeleton[-1])):
            row, col = map(int, point)
            distance_to_disc = float(disc_distance[row, col])
            if distance_to_disc <= config.disc_root_radius:
                candidates.append(
                    RootAnchor(
                        id=next_anchor_id,
                        segment_id=segment_id,
                        endpoint=endpoint,
                        kind="disc",
                        distance=distance_to_disc,
                        position=(float(row), float(col)),
                        contact_position=(float(row), float(col)),
                    )
                )
                next_anchor_id += 1
            elif (segment_id, endpoint) in junction_disc_contacts:
                port, source, path_distance = junction_disc_contacts[
                    segment_id, endpoint
                ]
                source_row, source_col = source
                candidates.append(
                    RootAnchor(
                        id=next_anchor_id,
                        segment_id=segment_id,
                        endpoint=endpoint,
                        kind="disc",
                        distance=float(disc_distance[source_row, source_col]),
                        port_id=port.id,
                        junction_id=port.junction_id,
                        position=port.position,
                        contact_position=(float(source_row), float(source_col)),
                        path_distance=path_distance,
                    )
                )
                next_anchor_id += 1

            distance_to_bounds = float(bounds_distance[row, col])
            is_terminal = (segment_id, endpoint) not in port_endpoints
            if (
                is_terminal
                and bounds_mask[row, col]
                and distance_to_bounds <= config.bounds_boundary_width
            ):
                candidates.append(
                    RootAnchor(
                        id=next_anchor_id,
                        segment_id=segment_id,
                        endpoint=endpoint,
                        kind="bounds",
                        distance=distance_to_bounds,
                        position=(float(row), float(col)),
                    )
                )
                next_anchor_id += 1

        candidates.append(
            RootAnchor(
                id=next_anchor_id,
                segment_id=segment_id,
                endpoint=None,
                kind="internal",
                distance=0.0,
            )
        )
        next_anchor_id += 1
        anchors[segment_id] = tuple(candidates)
    return anchors


def _orient_two_port_microsegments(
    segments: dict[int, SegmentInfo],
    ports: dict[int, Port],
    fit_length: float,
) -> None:
    """Assign opposite endpoints jointly to each usable microbridge."""
    ports_by_segment: dict[int, list[int]] = {segment_id: [] for segment_id in segments}
    for port_id, port in ports.items():
        ports_by_segment[port.segment_id].append(port_id)

    for segment_id, port_ids in ports_by_segment.items():
        segment = segments[segment_id]
        if not segment.is_microsegment or len(port_ids) != 2:
            continue
        first_id, second_id = sorted(port_ids)
        first_position = np.asarray(ports[first_id].position)
        second_position = np.asarray(ports[second_id].position)
        first_endpoint = segment.skeleton[0]
        second_endpoint = segment.skeleton[-1]
        direct_cost = float(
            np.linalg.norm(first_position - first_endpoint)
            + np.linalg.norm(second_position - second_endpoint)
        )
        reverse_cost = float(
            np.linalg.norm(first_position - second_endpoint)
            + np.linalg.norm(second_position - first_endpoint)
        )
        endpoints = (0, 1) if direct_cost <= reverse_cost else (1, 0)
        for port_id, endpoint in zip((first_id, second_id), endpoints, strict=True):
            tangent = _segment_direction(segment, endpoint, fit_length)
            ports[port_id] = replace(
                ports[port_id],
                endpoint=endpoint,
                tangent=(float(tangent[0]), float(tangent[1])),
            )


def _exclude_artifact_microsegment_ports(
    segments: dict[int, SegmentInfo],
    ports: dict[int, Port],
    junctions: dict[int, JunctionInfo],
) -> set[int]:
    """Remove lone ports created by microsegments trapped between junctions.

    A real short junction bridge exposes at least two usable ports. A component
    classified as a microsegment from multiple junction contacts but exposing
    fewer than two ports cannot carry graph flow and is forced to DISCARD by the
    optimizer. Keeping its lone port would nevertheless inflate the junction's
    degree and can trigger the high-port all-singletons fallback.
    """
    port_counts = {segment_id: 0 for segment_id in segments}
    for port in ports.values():
        port_counts[port.segment_id] += 1
    artifact_segment_ids = {
        segment_id
        for segment_id, segment in segments.items()
        if segment.is_microsegment
        and segment.junction_contact_count >= 2
        and port_counts[segment_id] < 2
    }
    if not artifact_segment_ids:
        return set()

    for port_id in list(ports):
        if ports[port_id].segment_id in artifact_segment_ids:
            del ports[port_id]
    for junction_id in list(junctions):
        junction = junctions[junction_id]
        junction.port_ids = tuple(
            port_id for port_id in junction.port_ids if port_id in ports
        )
        if not junction.port_ids:
            del junctions[junction_id]
    return artifact_segment_ids


def _extract_problem(
    logits: np.ndarray,
    config: AVInferenceConfig,
    disc_mask: np.ndarray | None,
    bounds_mask: np.ndarray | None,
) -> AVInferenceProblem | None:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[-1] < 4:
        raise ValueError(f"Expected HWC AV logits with four heads, got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("AV logits contain non-finite values")

    disc_mask = _binary_mask(disc_mask, logits.shape[:2], "disc_mask", default=False)
    bounds_mask = _binary_mask(
        bounds_mask, logits.shape[:2], "bounds_mask", default=True
    )
    vessel_probability = _sigmoid(logits[..., 0])
    crossing_probability = _sigmoid(logits[..., 3])
    av_probability = _av_probabilities(logits)
    vessel = (vessel_probability >= config.vessel_threshold) & bounds_mask & ~disc_mask
    skeleton = skeletonize(vessel)
    if int(skeleton.sum()) < config.min_segment_size:
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    ) - skeleton.astype(np.uint8)
    graph_nodes = skeleton & (neighbour_count > 2)
    crossing_seed = skeleton & binary_dilation(
        vessel & (crossing_probability >= config.crossing_threshold), disk(1)
    )
    junction_seed = graph_nodes | crossing_seed
    junction_mask = binary_dilation(junction_seed, disk(config.junction_radius))
    junction_image = connected_labels(junction_mask, connectivity=2).astype(np.int32)

    raw_components = connected_labels(skeleton & ~junction_mask, connectivity=2)
    segment_image = np.zeros(raw_components.shape, dtype=np.int32)
    segment_topology: dict[int, tuple[bool, int]] = {}
    square_kernel = np.ones((3, 3), dtype=bool)
    disc_contact_region = (
        ndimage.distance_transform_edt(~disc_mask) <= config.disc_root_radius
        if np.any(disc_mask)
        else np.zeros(disc_mask.shape, dtype=bool)
    )
    bounds_boundary = bounds_mask & (
        _inside_boundary_distance(bounds_mask) <= config.bounds_boundary_width
    )
    next_segment_id = 1
    for component_id in range(1, int(raw_components.max()) + 1):
        component = raw_components == component_id
        component_size = int(component.sum())
        shell = ndimage.binary_dilation(component, structure=square_kernel) & ~component
        junction_contact_count = 0
        for junction_id in np.unique(junction_image[shell]):
            if junction_id <= 0:
                continue
            contacts = shell & (junction_image == junction_id)
            junction_contact_count += int(
                connected_labels(contacts, connectivity=2).max()
            )
        is_microsegment = (
            component_size < config.min_segment_size and junction_contact_count >= 2
        )
        touches_root_region = bool(
            np.any(component & disc_contact_region)
            or np.any(component & bounds_boundary)
        )
        if (
            component_size < config.min_segment_size
            and not is_microsegment
            and not touches_root_region
        ):
            continue
        segment_image[component] = next_segment_id
        segment_topology[next_segment_id] = (
            is_microsegment,
            junction_contact_count,
        )
        next_segment_id += 1
    if next_segment_id == 1:
        return None

    _, nearest = ndimage.distance_transform_edt(segment_image == 0, return_indices=True)
    nearest_segments = segment_image[nearest[0], nearest[1]]
    radius_image = ndimage.distance_transform_edt(vessel)
    segments: dict[int, SegmentInfo] = {}
    for segment_id in range(1, next_segment_id):
        skeleton_points = _order_skeleton_points(
            np.argwhere(segment_image == segment_id)
        )
        pixels_mask = vessel & (nearest_segments == segment_id)
        evidence_mask = pixels_mask & ~junction_mask
        if not np.any(evidence_mask):
            evidence_mask = pixels_mask
        pixels = np.argwhere(pixels_mask)
        is_microsegment, junction_contact_count = segment_topology[segment_id]
        spline = (
            None
            if is_microsegment
            else SegmentSpline.fit(
                skeleton_points,
                error_fraction=config.spline_error_fraction,
            )
        )
        skeleton_widths = (
            2.0 * radius_image[skeleton_points[:, 0], skeleton_points[:, 1]]
        )
        segment_artery_probability = av_probability[..., 0][evidence_mask]
        confidence = config.mixed_av_confidence_threshold
        segments[segment_id] = SegmentInfo(
            id=segment_id,
            skeleton=skeleton_points,
            pixels=pixels,
            spline=spline,
            thickness=float(np.median(skeleton_widths)),
            vessel_probability=float(np.mean(vessel_probability[evidence_mask])),
            artery_probability=float(np.mean(segment_artery_probability)),
            vein_probability=float(np.mean(av_probability[..., 1][evidence_mask])),
            artery_dominant_fraction=float(
                np.mean(segment_artery_probability >= confidence)
            ),
            vein_dominant_fraction=float(
                np.mean(segment_artery_probability <= 1.0 - confidence)
            ),
            crossing_fraction=float(
                np.mean(
                    crossing_probability[evidence_mask] >= config.crossing_threshold
                )
            ),
            is_microsegment=is_microsegment,
            junction_contact_count=junction_contact_count,
        )

    ports: dict[int, Port] = {}
    junctions: dict[int, JunctionInfo] = {}
    next_port_id = 1
    for junction_id in range(1, int(junction_image.max()) + 1):
        region = junction_image == junction_id
        shell = ndimage.binary_dilation(region, structure=square_kernel) & ~region
        contact_segments = [
            int(value) for value in np.unique(segment_image[shell]) if value > 0
        ]
        junction_port_ids: list[int] = []
        for segment_id in contact_segments:
            contact = shell & (segment_image == segment_id)
            contact_components = connected_labels(contact, connectivity=2)
            for contact_id in range(1, int(contact_components.max()) + 1):
                contact_points = np.argwhere(contact_components == contact_id)
                if not len(contact_points):
                    continue
                segment = segments[segment_id]
                center = contact_points.mean(axis=0)
                endpoint_distances = [
                    float(np.linalg.norm(center - segment.skeleton[0])),
                    float(np.linalg.norm(center - segment.skeleton[-1])),
                ]
                endpoint = int(np.argmin(endpoint_distances))
                tangent = _segment_direction(
                    segment, endpoint, config.spline_fit_length
                )
                sample_count = min(
                    len(segment.skeleton), max(3, 2 * config.junction_radius)
                )
                sample_points = (
                    segment.skeleton[:sample_count]
                    if endpoint == 0
                    else segment.skeleton[-sample_count:]
                )
                thickness = float(
                    np.median(radius_image[sample_points[:, 0], sample_points[:, 1]])
                    * 2.0
                )
                port = Port(
                    id=next_port_id,
                    junction_id=junction_id,
                    segment_id=segment_id,
                    endpoint=endpoint,
                    position=(float(center[0]), float(center[1])),
                    tangent=(float(tangent[0]), float(tangent[1])),
                    thickness=thickness,
                )
                ports[port.id] = port
                junction_port_ids.append(port.id)
                next_port_id += 1
        if not junction_port_ids:
            continue
        region_pixels = np.argwhere(region)
        junction_vessel_probabilities = crossing_probability[region & vessel]
        if not len(junction_vessel_probabilities):
            junction_vessel_probabilities = crossing_probability[region]
        junctions[junction_id] = JunctionInfo(
            id=junction_id,
            pixels=region_pixels,
            port_ids=tuple(junction_port_ids),
            crossing_probability=float(np.quantile(junction_vessel_probabilities, 0.9)),
        )

    _orient_two_port_microsegments(segments, ports, config.spline_fit_length)
    _exclude_artifact_microsegment_ports(segments, ports, junctions)
    for junction in junctions.values():
        junction.configurations = _make_configurations(
            junction, ports, segments, config
        )

    root_anchors = _make_root_anchors(
        segments,
        ports,
        junctions,
        skeleton,
        junction_image,
        disc_mask,
        bounds_mask,
        config,
    )

    return AVInferenceProblem(
        logits=logits,
        vessel=vessel,
        vessel_probability=vessel_probability,
        crossing_probability=crossing_probability,
        segment_image=segment_image,
        nearest_segments=nearest_segments,
        junction_image=junction_image,
        segments=segments,
        ports=ports,
        junctions=junctions,
        root_anchors=root_anchors,
    )


def _root_anchor_cost(
    anchor: RootAnchor,
    segment: SegmentInfo,
    median_thickness: float,
    config: AVInferenceConfig,
) -> float:
    base_penalty = {
        "disc": config.disc_root_penalty,
        "bounds": config.bounds_root_penalty,
        "internal": config.internal_root_penalty,
    }[anchor.kind]
    if anchor.kind == "disc":
        distance_scale = max(config.disc_root_radius, 1)
    elif anchor.kind == "bounds":
        distance_scale = config.bounds_boundary_width
    else:
        distance_scale = 1
    distance_cost = config.root_distance_cost_weight * anchor.distance / distance_scale
    thickness_cost = (
        config.root_cost_weight * median_thickness / max(segment.thickness, _EPSILON)
    )
    return float(base_penalty + distance_cost + thickness_cost)


def _segment_unary_costs(
    segment: SegmentInfo,
    config: AVInferenceConfig,
    *,
    mixed_candidate: bool = False,
) -> dict[int, float]:
    vessel_probability = float(
        np.clip(segment.vessel_probability, _EPSILON, 1.0 - _EPSILON)
    )
    artery_probability = float(
        np.clip(segment.artery_probability, _EPSILON, 1.0 - _EPSILON)
    )
    vein_probability = float(
        np.clip(segment.vein_probability, _EPSILON, 1.0 - _EPSILON)
    )
    evidence_scale = min(np.sqrt(len(segment.skeleton)), 4.0)
    vessel_keep = -log(vessel_probability)
    artery_cost = evidence_scale * (
        config.vessel_cost_weight * vessel_keep
        + config.av_cost_weight * -log(artery_probability)
    )
    vein_cost = evidence_scale * (
        config.vessel_cost_weight * vessel_keep
        + config.av_cost_weight * -log(vein_probability)
    )
    mixture_evidence = _mixed_evidence(segment)
    single_class_penalty = (
        config.mixed_single_class_cost_weight * evidence_scale * mixture_evidence**2
        if mixed_candidate
        else 0.0
    )
    mixed_cost = (
        min(artery_cost, vein_cost)
        + config.mixed_segment_cost
        + config.mixed_evidence_cost_weight * -log(mixture_evidence)
    )
    return {
        ARTERY: artery_cost + single_class_penalty,
        VEIN: vein_cost + single_class_penalty,
        MIXED: mixed_cost,
        DISCARD: evidence_scale
        * config.vessel_cost_weight
        * -log(1.0 - vessel_probability)
        + single_class_penalty,
    }


def _mixed_evidence(segment: SegmentInfo) -> float:
    """Score overlap and spatially separated A/V evidence within one segment."""
    side_by_side = min(
        2.0 * segment.artery_dominant_fraction,
        2.0 * segment.vein_dominant_fraction,
        1.0,
    )
    crossing = float(np.clip(segment.crossing_fraction, 0.0, 1.0))
    # Noisy-OR lets either explanation support MIXED and strengthens a
    # combination of partial overlap and side-by-side evidence.
    return float(max(1.0 - (1.0 - side_by_side) * (1.0 - crossing), _EPSILON))


def _mixed_candidate_segments(
    problem: AVInferenceProblem,
    config: AVInferenceConfig,
) -> set[int]:
    """Return conservative mixed candidates supported by evidence and topology."""
    if not config.enable_mixed_segments:
        return set()
    connected_endpoints: dict[int, set[int]] = {
        segment_id: set() for segment_id in problem.segments
    }
    for port in problem.ports.values():
        connected_endpoints[port.segment_id].add(port.endpoint)
    for segment_id, anchors in problem.root_anchors.items():
        connected_endpoints[segment_id].update(
            anchor.endpoint
            for anchor in anchors
            if anchor.kind != "internal" and anchor.endpoint is not None
        )

    candidates: set[int] = set()
    for segment_id, segment in problem.segments.items():
        has_side_by_side_evidence = (
            min(
                segment.artery_dominant_fraction,
                segment.vein_dominant_fraction,
            )
            >= config.mixed_av_min_fraction
        )
        has_mixed_evidence = (
            has_side_by_side_evidence
            or segment.crossing_fraction >= config.mixed_crossing_fraction_threshold
        )
        has_two_sided_topology = len(connected_endpoints[segment_id]) >= 2
        if has_mixed_evidence and has_two_sided_topology:
            candidates.add(segment_id)
    return candidates


def _solve_problem(
    problem: AVInferenceProblem,
    config: AVInferenceConfig,
) -> (
    tuple[
        dict[int, int],
        dict[int, int],
        dict[int, list[tuple[int, int]]],
        dict[int, list[RootAnchor]],
        float | None,
        str,
    ]
    | None
):
    variables = _VariableRegistry()
    constraints = _ConstraintBuilder()
    segment_ids = tuple(sorted(problem.segments))
    n_segments = len(segment_ids)
    big_m = float(n_segments + 1)

    x: dict[tuple[int, int], int] = {}
    root: dict[tuple[int, int], list[int]] = {}
    root_flow: dict[tuple[int, int], list[int]] = {}
    root_metadata: dict[int, tuple[int, RootAnchor]] = {}
    depth: dict[tuple[int, int], int] = {}
    thicknesses = np.asarray(
        [problem.segments[segment_id].thickness for segment_id in segment_ids]
    )
    median_thickness = float(np.median(thicknesses)) if len(thicknesses) else 1.0
    mixed_candidates = _mixed_candidate_segments(problem, config)

    for segment_id in segment_ids:
        unary = _segment_unary_costs(
            problem.segments[segment_id],
            config,
            mixed_candidate=segment_id in mixed_candidates,
        )
        for state in (ARTERY, VEIN, MIXED, DISCARD):
            upper = 1.0 if state != MIXED or segment_id in mixed_candidates else 0.0
            x[segment_id, state] = variables.add(unary[state], upper=upper)
        constraints.add(
            {x[segment_id, state]: 1.0 for state in (ARTERY, VEIN, MIXED, DISCARD)},
            1.0,
            1.0,
        )
        for class_id in (ARTERY, VEIN):
            depth[segment_id, class_id] = variables.add(
                0.0, lower=0.0, upper=big_m, integer=False
            )
            constraints.add(
                {
                    depth[segment_id, class_id]: 1.0,
                    x[segment_id, class_id]: -big_m,
                    x[segment_id, MIXED]: -big_m,
                },
                -inf,
                0.0,
            )
            root[segment_id, class_id] = []
            root_flow[segment_id, class_id] = []
            for anchor in problem.root_anchors[segment_id]:
                root_variable = variables.add(
                    _root_anchor_cost(
                        anchor,
                        problem.segments[segment_id],
                        median_thickness,
                        config,
                    )
                )
                root[segment_id, class_id].append(root_variable)
                root_metadata[root_variable] = (class_id, anchor)
                flow_cost = (
                    0.0
                    if anchor.kind == "disc"
                    else config.fallback_root_subtree_cost_weight
                )
                root_flow_variable = variables.add(
                    flow_cost, upper=big_m, integer=False
                )
                root_flow[segment_id, class_id].append(root_flow_variable)
                constraints.add(
                    {root_flow_variable: 1.0, root_variable: -big_m}, -inf, 0.0
                )
                constraints.add(
                    {
                        root_variable: 1.0,
                        x[segment_id, class_id]: -1.0,
                        x[segment_id, MIXED]: -1.0,
                    },
                    -inf,
                    0.0,
                )
                constraints.add(
                    {
                        depth[segment_id, class_id]: 1.0,
                        root_variable: big_m,
                    },
                    -inf,
                    big_m,
                )

    q: dict[tuple[int, int], int] = {}
    connected_configurations: dict[int, list[int]] = {
        port_id: [] for port_id in problem.ports
    }
    incoming: dict[tuple[int, int], list[int]] = {
        (segment_id, class_id): []
        for segment_id in segment_ids
        for class_id in (ARTERY, VEIN)
    }
    outgoing: dict[tuple[int, int], list[int]] = {
        (segment_id, class_id): []
        for segment_id in segment_ids
        for class_id in (ARTERY, VEIN)
    }
    arc_metadata: dict[int, tuple[int, int, int]] = {}
    arc_flow: dict[int, int] = {}

    for junction_id, junction in sorted(problem.junctions.items()):
        for candidate in junction.configurations:
            q[junction_id, candidate.index] = variables.add(candidate.cost)
        constraints.add(
            {
                q[junction_id, candidate.index]: 1.0
                for candidate in junction.configurations
            },
            1.0,
            1.0,
        )
        for candidate in junction.configurations:
            candidate_variable = q[junction_id, candidate.index]
            non_single_groups = [
                group for group in candidate.groups if len(group.port_ids) > 1
            ]
            link_group_indices = [
                group_index
                for group_index, group in enumerate(non_single_groups)
                for _ in _group_links(group, problem.ports)
            ]
            group_arcs_by_class: dict[tuple[int, int], list[int]] = {
                (group_index, class_id): []
                for group_index in range(len(non_single_groups))
                for class_id in (ARTERY, VEIN)
            }
            for group in candidate.groups:
                if len(group.port_ids) > 1:
                    for port_id in group.port_ids:
                        connected_configurations[port_id].append(candidate_variable)
            for group_index, link in zip(
                link_group_indices, candidate.links, strict=True
            ):
                arc_variables_by_class: dict[int, list[int]] = {}
                directions = [(link.first_segment, link.second_segment)]
                if not link.directed:
                    directions.append((link.second_segment, link.first_segment))
                for class_id in (ARTERY, VEIN):
                    class_arcs: list[int] = []
                    for parent_id, child_id in directions:
                        arc = variables.add(0.0)
                        class_arcs.append(arc)
                        incoming[child_id, class_id].append(arc)
                        outgoing[parent_id, class_id].append(arc)
                        arc_metadata[arc] = (parent_id, child_id, class_id)
                        flow = variables.add(0.0, upper=big_m, integer=False)
                        arc_flow[arc] = flow
                        constraints.add({flow: 1.0, arc: -big_m}, -inf, 0.0)
                        constraints.add(
                            {
                                arc: 1.0,
                                x[parent_id, class_id]: -1.0,
                            },
                            -inf,
                            0.0,
                        )
                        constraints.add(
                            {
                                arc: 1.0,
                                x[child_id, class_id]: -1.0,
                            },
                            -inf,
                            0.0,
                        )
                        constraints.add(
                            {
                                depth[child_id, class_id]: 1.0,
                                depth[parent_id, class_id]: -1.0,
                                arc: -big_m,
                            },
                            1.0 - big_m,
                            inf,
                        )
                    arc_variables_by_class[class_id] = class_arcs
                    group_arcs_by_class[group_index, class_id].extend(class_arcs)
                    constraints.add(
                        {
                            **{arc: 1.0 for arc in class_arcs},
                            candidate_variable: -1.0,
                        },
                        -inf,
                        0.0,
                    )

                all_arcs = [
                    arc
                    for class_arcs in arc_variables_by_class.values()
                    for arc in class_arcs
                ]
                constraints.add(
                    {
                        **{arc: 1.0 for arc in all_arcs},
                        candidate_variable: -1.0,
                    },
                    0.0,
                    inf,
                )

            # An overlap is a partition into physically separate vessel paths.
            # Retinal arteries do not cross arteries, and veins do not cross
            # veins, so no two connected groups may be active in the same AV
            # class. This still permits one artery group and one vein group.
            if candidate.overlap:
                group_class_active: dict[tuple[int, int], int] = {}
                for group_index in range(len(non_single_groups)):
                    for class_id in (ARTERY, VEIN):
                        active = variables.add(0.0)
                        group_class_active[group_index, class_id] = active
                        class_arcs = group_arcs_by_class[group_index, class_id]
                        for arc in class_arcs:
                            constraints.add({arc: 1.0, active: -1.0}, -inf, 0.0)
                        constraints.add(
                            {
                                active: 1.0,
                                **{arc: -1.0 for arc in class_arcs},
                            },
                            -inf,
                            0.0,
                        )
                for class_id in (ARTERY, VEIN):
                    constraints.add(
                        {
                            group_class_active[group_index, class_id]: 1.0
                            for group_index in range(len(non_single_groups))
                        },
                        -inf,
                        1.0,
                    )

    # A physical mixed port is duplicated into one artery and one vein port.
    # These logical ports can connect to different neighbours at the same
    # junction, which is necessary when adjacent A/V vessels were extracted as
    # one segment. The ordinary physical partition keeps treating that port as
    # a singleton whenever MIXED is selected.
    mixed_endpoint_connections: dict[tuple[int, int, int], list[int]] = {
        (segment_id, endpoint, class_id): []
        for segment_id in mixed_candidates
        for endpoint in (0, 1)
        for class_id in (ARTERY, VEIN)
    }
    mixed_port_connections: dict[tuple[int, int], list[int]] = {
        (port_id, class_id): []
        for port_id in problem.ports
        for class_id in (ARTERY, VEIN)
    }
    for junction in problem.junctions.values():
        for first_port_id, second_port_id in combinations(junction.port_ids, 2):
            first_port = problem.ports[first_port_id]
            second_port = problem.ports[second_port_id]
            first_id = first_port.segment_id
            second_id = second_port.segment_id
            if first_id == second_id or not (
                first_id in mixed_candidates or second_id in mixed_candidates
            ):
                continue
            pair_group = PortGroup((first_port_id, second_port_id))
            direction_cost = _direction_cost(pair_group, problem.ports, config)
            if not np.isfinite(direction_cost):
                continue
            pair_cost = (
                config.direction_cost_weight * direction_cost
                + config.thickness_cost_weight
                * _thickness_cost(pair_group, problem.ports, config.thickness_exponent)
            )
            for class_id in (ARTERY, VEIN):
                connection = variables.add(pair_cost)
                mixed_port_connections[first_port_id, class_id].append(connection)
                mixed_port_connections[second_port_id, class_id].append(connection)
                if first_id in mixed_candidates:
                    mixed_endpoint_connections[
                        first_id, first_port.endpoint, class_id
                    ].append(connection)
                if second_id in mixed_candidates:
                    mixed_endpoint_connections[
                        second_id, second_port.endpoint, class_id
                    ].append(connection)
                constraints.add(
                    {
                        connection: 1.0,
                        x[first_id, MIXED]: -1.0,
                        x[second_id, MIXED]: -1.0,
                    },
                    -inf,
                    0.0,
                )
                for segment_id in (first_id, second_id):
                    constraints.add(
                        {
                            connection: 1.0,
                            x[segment_id, class_id]: -1.0,
                            x[segment_id, MIXED]: -1.0,
                        },
                        -inf,
                        0.0,
                    )

                pair_arcs: list[int] = []
                for parent_id, child_id in (
                    (first_id, second_id),
                    (second_id, first_id),
                ):
                    arc = variables.add(0.0)
                    pair_arcs.append(arc)
                    incoming[child_id, class_id].append(arc)
                    outgoing[parent_id, class_id].append(arc)
                    arc_metadata[arc] = (parent_id, child_id, class_id)
                    flow = variables.add(0.0, upper=big_m, integer=False)
                    arc_flow[arc] = flow
                    constraints.add({flow: 1.0, arc: -big_m}, -inf, 0.0)
                    constraints.add({arc: 1.0, connection: -1.0}, -inf, 0.0)
                    constraints.add(
                        {
                            depth[child_id, class_id]: 1.0,
                            depth[parent_id, class_id]: -1.0,
                            arc: -big_m,
                        },
                        1.0 - big_m,
                        inf,
                    )
                constraints.add(
                    {
                        pair_arcs[0]: 1.0,
                        pair_arcs[1]: 1.0,
                        connection: -1.0,
                    },
                    0.0,
                    0.0,
                )

    for (port_id, class_id), connections in mixed_port_connections.items():
        port = problem.ports[port_id]
        constraints.add(
            {
                **{connection: 1.0 for connection in connections},
                x[port.segment_id, class_id]: -1.0,
                x[port.segment_id, MIXED]: -1.0,
            },
            -inf,
            0.0,
        )

    for (
        segment_id,
        endpoint,
        class_id,
    ), connections in mixed_endpoint_connections.items():
        values = {connection: 1.0 for connection in connections}
        for root_variable, anchor in zip(
            root[segment_id, class_id],
            problem.root_anchors[segment_id],
            strict=True,
        ):
            if anchor.kind == "internal" or anchor.endpoint != endpoint:
                continue
            mixed_root = variables.add(0.0)
            constraints.add(
                {mixed_root: 1.0, root_variable: -1.0},
                -inf,
                0.0,
            )
            constraints.add(
                {mixed_root: 1.0, x[segment_id, MIXED]: -1.0},
                -inf,
                0.0,
            )
            constraints.add(
                {
                    mixed_root: 1.0,
                    root_variable: -1.0,
                    x[segment_id, MIXED]: -1.0,
                },
                -1.0,
                inf,
            )
            values[mixed_root] = 1.0
        values[x[segment_id, MIXED]] = -1.0
        constraints.add(values, 0.0, 0.0)

    ports_by_segment: dict[int, list[int]] = {
        segment_id: [] for segment_id in segment_ids
    }
    for port_id, port in problem.ports.items():
        ports_by_segment[port.segment_id].append(port_id)
    for segment_id, segment in problem.segments.items():
        if not segment.is_microsegment:
            continue
        port_ids = ports_by_segment[segment_id]
        if len(port_ids) < 2:
            constraints.add(
                {
                    x[segment_id, ARTERY]: 1.0,
                    x[segment_id, VEIN]: 1.0,
                },
                0.0,
                0.0,
            )
            continue
        for port_id in port_ids:
            values = {variable: 1.0 for variable in connected_configurations[port_id]}
            values[x[segment_id, ARTERY]] = -1.0
            values[x[segment_id, VEIN]] = -1.0
            constraints.add(values, 0.0, 0.0)

    for segment_id in segment_ids:
        for class_id in (ARTERY, VEIN):
            values = {arc: 1.0 for arc in incoming[segment_id, class_id]}
            for root_variable in root[segment_id, class_id]:
                values[root_variable] = 1.0
            values[x[segment_id, class_id]] = -1.0
            values[x[segment_id, MIXED]] = -1.0
            constraints.add(values, 0.0, 0.0)

            # Each active segment generates one unit of flow. It travels
            # opposite the directed forest edges and is absorbed by the
            # selected root. Charging flow only at fallback roots makes their
            # cost proportional to the disconnected subtree size.
            flow_balance = {
                x[segment_id, class_id]: -1.0,
                x[segment_id, MIXED]: -1.0,
            }
            for variable in root_flow[segment_id, class_id]:
                flow_balance[variable] = 1.0
            for arc in incoming[segment_id, class_id]:
                flow_balance[arc_flow[arc]] = 1.0
            for arc in outgoing[segment_id, class_id]:
                flow_balance[arc_flow[arc]] = -1.0
            constraints.add(flow_balance, 0.0, 0.0)

    n_variables = len(variables.costs)
    result = milp(
        c=np.asarray(variables.costs),
        integrality=np.asarray(variables.integrality, dtype=np.int8),
        bounds=Bounds(np.asarray(variables.lower), np.asarray(variables.upper)),
        constraints=constraints.linear_constraint(n_variables),
        options={
            "time_limit": config.solver_time_limit,
            "mip_rel_gap": config.solver_mip_rel_gap,
            "presolve": True,
        },
    )
    if result.x is None:
        return None

    labels = {
        segment_id: max(
            (ARTERY, VEIN, MIXED, DISCARD),
            key=lambda class_id: result.x[x[segment_id, class_id]],
        )
        for segment_id in segment_ids
    }
    selected = {
        junction_id: max(
            (candidate.index for candidate in junction.configurations),
            key=lambda candidate_index: result.x[q[junction_id, candidate_index]],
        )
        for junction_id, junction in problem.junctions.items()
    }
    forest_edges = {ARTERY: [], VEIN: []}
    for arc, (parent_id, child_id, class_id) in arc_metadata.items():
        if result.x[arc] > 0.5:
            forest_edges[class_id].append((parent_id, child_id))
    selected_roots = {ARTERY: [], VEIN: []}
    for root_variable, (class_id, anchor) in root_metadata.items():
        if result.x[root_variable] > 0.5:
            selected_roots[class_id].append(anchor)
    return (
        labels,
        selected,
        forest_edges,
        selected_roots,
        result.fun,
        str(result.message),
    )


def _basic_mask(
    logits: np.ndarray,
    config: AVInferenceConfig,
    disc_mask: np.ndarray | None,
    bounds_mask: np.ndarray | None,
) -> np.ndarray:
    disc_mask = _binary_mask(disc_mask, logits.shape[:2], "disc_mask", default=False)
    bounds_mask = _binary_mask(
        bounds_mask, logits.shape[:2], "bounds_mask", default=True
    )
    vessel = (
        (_sigmoid(logits[..., 0]) >= config.vessel_threshold) & bounds_mask & ~disc_mask
    )
    crossing = vessel & (_sigmoid(logits[..., 3]) >= config.crossing_threshold)
    result = np.zeros(vessel.shape, dtype=np.uint8)
    result[vessel] = np.where(logits[..., 1][vessel] >= logits[..., 2][vessel], 1, 2)
    result[crossing] = 3
    return result


def _fallback_result(
    logits: np.ndarray,
    config: AVInferenceConfig,
    disc_mask: np.ndarray | None,
    bounds_mask: np.ndarray | None,
    status: str,
) -> AVInferenceResult:
    mask = _basic_mask(logits, config, disc_mask, bounds_mask)
    return AVInferenceResult(
        mask=mask,
        changed_segment_mask=np.zeros(mask.shape, dtype=bool),
        discarded_mask=np.zeros(mask.shape, dtype=bool),
        segment_id_image=np.zeros(mask.shape, dtype=np.int32),
        segment_labels={},
        mixed_segment_ids=set(),
        microsegment_ids=set(),
        selected_configurations={},
        forest_edges={ARTERY: [], VEIN: []},
        selected_roots={ARTERY: [], VEIN: []},
        objective=None,
        solver_status=status,
    )


def infer_artery_vein(
    logits: np.ndarray,
    *,
    disc_mask: np.ndarray | None = None,
    bounds_mask: np.ndarray | None = None,
    **kwargs: float,
) -> AVInferenceResult:
    """Infer anatomically rooted acyclic AV forests and rasterize them."""
    config = AVInferenceConfig(**kwargs)
    problem = _extract_problem(logits, config, disc_mask, bounds_mask)
    if problem is None:
        return _fallback_result(
            logits,
            config,
            disc_mask,
            bounds_mask,
            "insufficient vessel segments",
        )
    try:
        solution = _solve_problem(problem, config)
    except Exception as error:  # noqa: BLE001 - inference safely falls back
        return _fallback_result(
            logits,
            config,
            disc_mask,
            bounds_mask,
            f"optimization failed: {error}",
        )
    if solution is None:
        return _fallback_result(
            logits,
            config,
            disc_mask,
            bounds_mask,
            "optimization found no solution",
        )
    labels, selected, forest_edges, selected_roots, objective, status = solution
    basic_mask = _basic_mask(logits, config, disc_mask, bounds_mask)

    mask = np.zeros(problem.vessel.shape, dtype=np.uint8)
    changed_segment_mask = np.zeros(problem.vessel.shape, dtype=bool)
    discarded_mask = np.zeros(problem.vessel.shape, dtype=bool)
    segment_id_image = np.where(problem.vessel, problem.nearest_segments, 0).astype(
        np.int32
    )
    for segment_id, class_id in labels.items():
        segment_pixels = problem.vessel & (problem.nearest_segments == segment_id)
        if class_id == DISCARD:
            discarded_mask[segment_pixels] = True
            continue
        if class_id == MIXED:
            mask[segment_pixels] = basic_mask[segment_pixels]
            continue
        mask[segment_pixels] = class_id
        if class_id != problem.segments[segment_id].initial_label:
            changed_segment_mask[segment_pixels] = True

    mixed_segment_ids = {
        segment_id for segment_id, class_id in labels.items() if class_id == MIXED
    }
    mixed_junction_ids = {
        port.junction_id
        for port in problem.ports.values()
        if port.segment_id in mixed_segment_ids
    }
    for junction_id, configuration_index in selected.items():
        junction = problem.junctions[junction_id]
        configuration = junction.configurations[configuration_index]
        region = problem.junction_image == junction_id
        region_vessel = region & problem.vessel
        if junction_id in mixed_junction_ids:
            mask[region_vessel] = basic_mask[region_vessel]
            continue
        non_single_groups = [
            group for group in configuration.groups if len(group.port_ids) > 1
        ]
        if configuration.overlap:
            mask[region_vessel] = 3
            continue
        connected_segments = {
            problem.ports[port_id].segment_id
            for group in non_single_groups
            for port_id in group.port_ids
        }
        if connected_segments:
            keep = region_vessel & np.isin(
                problem.nearest_segments, list(connected_segments)
            )
            mask[region_vessel & ~keep] = 0
        else:
            mask[region_vessel] = 0

    mixed_pixels = problem.vessel & np.isin(
        problem.nearest_segments, list(mixed_segment_ids)
    )
    mask[mixed_pixels] = basic_mask[mixed_pixels]

    return AVInferenceResult(
        mask=mask,
        changed_segment_mask=changed_segment_mask,
        discarded_mask=discarded_mask,
        segment_id_image=segment_id_image,
        segment_labels=labels,
        mixed_segment_ids=mixed_segment_ids,
        microsegment_ids={
            segment_id
            for segment_id, segment in problem.segments.items()
            if segment.is_microsegment
        },
        selected_configurations=selected,
        forest_edges=forest_edges,
        selected_roots=selected_roots,
        objective=float(objective) if objective is not None else None,
        solver_status=status,
    )
