from __future__ import annotations

from itertools import pairwise

import networkx as nx
import numpy as np
import pytest
import torch
from rtnls_inference.artery_vein import (
    AV_HEAD_NAMES,
    av_logits_to_legacy_probabilities,
    load_av_head_logits,
    save_av_head_logits,
)
from rtnls_inference.ensembles.av_inference.optim import (
    AVInferenceConfig,
    JunctionInfo,
    Port,
    PortGroup,
    SegmentInfo,
    _bifurcation_direction_cost,
    _compound_group_cost,
    _configuration_groups,
    _direction_cost,
    _exclude_artifact_microsegment_ports,
    _make_configurations,
    _orient_two_port_microsegments,
    _restricted_partitions,
    infer_artery_vein,
)
from rtnls_inference.ensembles.ensemble_artery_vein import (
    AV_REFINEMENT_VARIANTS,
    artery_vein_refinement_variants,
    halo_sliding_window_inference,
    normalize_refinement_mode,
    refine_artery_vein_graph,
    refine_artery_vein_graph_full_with_flips,
)
from skimage.draw import line
from skimage.morphology import binary_dilation, disk


class _CenterPredictor(torch.nn.Module):
    def forward(self, image):
        center = image[:, :1, 8:24, 8:24]
        return center.repeat(1, len(AV_HEAD_NAMES), 1, 1)


class _RecordingCenterPredictor(_CenterPredictor):
    def __init__(self):
        super().__init__()
        self.tiles = []

    def forward(self, image):
        self.tiles.append(image.detach().clone())
        return super().forward(image)


class _CenterEnsemble(torch.nn.Module):
    def forward(self, image):
        prediction = _CenterPredictor()(image)
        return torch.stack([prediction, prediction + 2.0])


def test_named_head_logits_roundtrip(tmp_path):
    logits = np.random.default_rng(3).normal(size=(23, 29, 4)).astype(np.float32)
    path = tmp_path / "sample.npz"
    save_av_head_logits(path, logits)
    loaded = load_av_head_logits(path)
    assert loaded.shape == logits.shape
    np.testing.assert_allclose(loaded, logits, atol=2e-3)


def test_named_head_logits_rejects_legacy_tensor(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, logits=np.zeros((20, 20, 4), dtype=np.float16))
    try:
        load_av_head_logits(path)
    except ValueError as error:
        assert "Invalid AV head-logit artifact" in str(error)
    else:
        raise AssertionError("Legacy logits should not be accepted")


def test_named_head_logits_rejects_schema_v1(tmp_path):
    path = tmp_path / "schema_v1.npz"
    payload = {name: np.zeros((20, 20), dtype=np.float16) for name in AV_HEAD_NAMES}
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        source_shape=np.asarray((20, 20), dtype=np.int32),
        **payload,
    )
    try:
        load_av_head_logits(path)
    except ValueError as error:
        assert "Unsupported AV head-logit schema 1" in str(error)
    else:
        raise AssertionError("Schema-v1 logits should not be accepted")


def test_named_head_logits_rejects_malformed_shape_and_nonfinite_values(tmp_path):
    malformed_path = tmp_path / "malformed.npz"
    payload = {name: np.zeros((20, 20), dtype=np.float16) for name in AV_HEAD_NAMES}
    payload["artery"] = np.zeros((19, 20), dtype=np.float16)
    np.savez(
        malformed_path,
        schema_version=np.asarray(2, dtype=np.int16),
        source_shape=np.asarray((20, 20), dtype=np.int32),
        **payload,
    )
    with pytest.raises(ValueError, match="has shape"):
        load_av_head_logits(malformed_path)

    nonfinite = np.zeros((20, 20, 4), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_av_head_logits(tmp_path / "nonfinite.npz", nonfinite)


def test_halo_stitching_reconstructs_center_predictions():
    image = torch.linspace(-1, 1, 70 * 74).reshape(1, 1, 70, 74)
    output = halo_sliding_window_inference(
        image,
        _CenterPredictor(),
        context_size=32,
        output_size=16,
        overlap=0.5,
        sw_batch_size=3,
    )
    assert output.shape == (1, 4, 70, 74)
    torch.testing.assert_close(output[:, 0], image[:, 0], atol=1e-5, rtol=1e-5)


def test_halo_stitching_uses_zero_context_padding():
    predictor = _RecordingCenterPredictor()
    image = torch.ones((1, 1, 16, 16))

    output = halo_sliding_window_inference(
        image,
        predictor,
        context_size=32,
        output_size=16,
        overlap=0.5,
    )

    tile = predictor.tiles[0]
    assert torch.count_nonzero(tile[..., :8, :]) == 0
    assert torch.count_nonzero(tile[..., -8:, :]) == 0
    assert torch.count_nonzero(tile[..., :, :8]) == 0
    assert torch.count_nonzero(tile[..., :, -8:]) == 0
    torch.testing.assert_close(tile[..., 8:24, 8:24], image)
    torch.testing.assert_close(output[:, 0], image[:, 0])


def test_halo_stitching_preserves_model_axis():
    image = torch.zeros((2, 1, 40, 42))
    output = halo_sliding_window_inference(
        image,
        _CenterEnsemble(),
        context_size=32,
        output_size=16,
        overlap=0.5,
        sw_batch_size=2,
    )
    assert output.shape == (2, 2, 4, 40, 42)
    torch.testing.assert_close(
        output[:, 1] - output[:, 0], torch.full_like(output[:, 0], 2)
    )


def test_legacy_projection_is_normalized():
    logits = np.random.default_rng(5).normal(size=(17, 19, 4)).astype(np.float32)
    probabilities = av_logits_to_legacy_probabilities(logits)
    assert probabilities.shape == (17, 19, 4)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)


def test_graph_refinement_preserves_support_and_crossings():
    logits = np.full((48, 48, 4), -8.0, dtype=np.float32)
    logits[23:26, 5:43, 0] = 8.0
    logits[5:43, 23:26, 0] = 8.0
    logits[23:26, 5:43, 1] = 3.0
    logits[23:26, 5:43, 2] = -3.0
    logits[5:43, 23:26, 1] = -3.0
    logits[5:43, 23:26, 2] = 3.0
    logits[22:27, 22:27, 3] = 8.0
    refined = refine_artery_vein_graph(
        logits,
        junction_radius=2,
        min_segment_size=3,
        solver_time_limit=5.0,
    )
    vessel = logits[..., 0] > 0
    assert np.array_equal(refined > 0, vessel)
    assert np.all(refined[22:27, 22:27][vessel[22:27, 22:27]] == 3)


def test_refinement_mode_aliases():
    assert normalize_refinement_mode("none") == "basic"
    assert normalize_refinement_mode("simple") == "refinement_simple"
    assert normalize_refinement_mode("full") == "refinement_full"


def test_all_refinement_variants_preserve_vessels_and_crossings():
    logits = np.full((48, 48, 4), -8.0, dtype=np.float32)
    logits[23:26, 5:43, 0] = 8.0
    logits[5:43, 23:26, 0] = 8.0
    logits[23:26, 5:43, 1] = 3.0
    logits[23:26, 5:43, 2] = -3.0
    logits[5:43, 23:26, 1] = -3.0
    logits[5:43, 23:26, 2] = 3.0
    logits[22:27, 22:27, 3] = 8.0

    variants = artery_vein_refinement_variants(
        logits,
        junction_radius=2,
        min_segment_size=3,
        solver_time_limit=5.0,
    )

    assert tuple(variants) == AV_REFINEMENT_VARIANTS
    vessel = logits[..., 0] > 0
    crossing = vessel & (logits[..., 3] > 0)
    for result in variants.values():
        assert np.array_equal(result > 0, vessel)
        assert np.all(result[crossing] == 3)


def test_structured_crossing_produces_two_acyclic_forests():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 5:59, 0] = 8.0
    logits[5:59, 30:33, 0] = 8.0
    logits[30:33, 5:59, 1] = 6.0
    logits[30:33, 5:59, 2] = -6.0
    logits[5:59, 30:33, 1] = -6.0
    logits[5:59, 30:33, 2] = 6.0
    logits[27:36, 27:36, 3] = 8.0

    result = infer_artery_vein(
        logits,
        junction_radius=3,
        min_segment_size=3,
        solver_time_limit=5.0,
    )

    assert set(result.segment_labels.values()) == {1, 2}
    observed_segment_ids = set(np.unique(result.segment_id_image)) - {0}
    assert observed_segment_ids == set(result.segment_labels)
    assert np.all(result.mask[27:36, 27:36][logits[27:36, 27:36, 0] > 0] == 3)
    for class_id in (1, 2):
        graph = nx.DiGraph(result.forest_edges[class_id])
        graph.add_nodes_from(
            segment_id
            for segment_id, label in result.segment_labels.items()
            if label == class_id
        )
        assert nx.is_directed_acyclic_graph(graph)
        assert nx.number_weakly_connected_components(graph) == len(
            result.selected_roots[class_id]
        )


@pytest.mark.parametrize(
    ("class_id", "class_channel"),
    [(1, 1), (2, 2)],
)
def test_same_class_crossing_is_resolved_as_one_connected_tree(class_id, class_channel):
    logits = np.full((96, 96, 4), -8.0, dtype=np.float32)
    logits[46:51, 5:91, 0] = 8.0
    logits[15:81, 47:50, 0] = 8.0
    vessel = logits[..., 0] > 0
    logits[vessel, class_channel] = 6.0
    logits[42:55, 42:55, 3] = 8.0

    result = infer_artery_vein(
        logits,
        junction_radius=3,
        min_segment_size=3,
        solver_time_limit=5.0,
    )

    other_class = 3 - class_id
    assert set(result.segment_labels.values()) == {class_id}
    assert len(result.selected_roots[class_id]) == 1
    assert not result.selected_roots[other_class]
    vessel_graph = nx.DiGraph(result.forest_edges[class_id])
    vessel_graph.add_nodes_from(result.segment_labels)
    assert nx.number_weakly_connected_components(vessel_graph) == 1


def test_disc_mask_removes_vessels_and_provides_preferred_root():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 3:61, 0] = 8.0
    logits[30:33, 3:61, 1] = 6.0
    disc_mask = np.zeros((64, 64), dtype=bool)
    disc_mask[26:38, 5:14] = True
    bounds_mask = np.zeros((64, 64), dtype=bool)
    bounds_mask[2:62, 2:62] = True

    result = infer_artery_vein(
        logits,
        disc_mask=disc_mask,
        bounds_mask=bounds_mask,
        junction_radius=2,
        min_segment_size=3,
        disc_root_radius=4,
        bounds_boundary_width=5,
        solver_time_limit=5.0,
    )

    assert not np.any(result.mask[disc_mask])
    assert not np.any(result.mask[~bounds_mask])
    assert result.selected_roots[1]
    assert {root.kind for root in result.selected_roots[1]} == {"disc"}
    assert not result.selected_roots[2]


def test_disc_contact_is_preserved_through_one_port_junction():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 20:61, 0] = 8.0
    logits[30:33, 20:61, 1] = 6.0
    logits[27:36, 20:28, 3] = 8.0
    disc_mask = np.zeros((64, 64), dtype=bool)
    disc_mask[24:40, 12:21] = True
    bounds_mask = np.zeros((64, 64), dtype=bool)
    bounds_mask[2:62, 2:62] = True

    result = infer_artery_vein(
        logits,
        disc_mask=disc_mask,
        bounds_mask=bounds_mask,
        junction_radius=5,
        min_segment_size=3,
        disc_root_radius=2,
        bounds_boundary_width=5,
        solver_time_limit=5.0,
    )

    assert len(result.segment_labels) == 1
    assert len(result.selected_roots[1]) == 1
    root = result.selected_roots[1][0]
    assert root.kind == "disc"
    assert root.junction_id is not None
    assert root.port_id is not None
    assert root.position is not None
    assert root.path_distance > 0


def test_disc_contacting_junction_can_root_separate_artery_and_vein_trees():
    logits = np.full((80, 80, 4), -8.0, dtype=np.float32)
    artery = np.zeros((80, 80), dtype=bool)
    vein = np.zeros((80, 80), dtype=bool)
    for target, class_mask in (((12, 72), artery), ((68, 72), vein)):
        rows, cols = line(40, 20, *target)
        class_mask[rows, cols] = True
        class_mask[:] = binary_dilation(class_mask, disk(1))
    vessel = artery | vein
    logits[vessel, 0] = 8.0
    logits[artery, 1] = 6.0
    logits[artery, 2] = -6.0
    logits[vein, 1] = -6.0
    logits[vein, 2] = 6.0
    logits[34:47, 18:31, 3] = 8.0
    disc_mask = np.zeros((80, 80), dtype=bool)
    disc_mask[31:50, 10:21] = True
    bounds_mask = np.zeros((80, 80), dtype=bool)
    bounds_mask[2:78, 2:78] = True

    result = infer_artery_vein(
        logits,
        disc_mask=disc_mask,
        bounds_mask=bounds_mask,
        junction_radius=5,
        min_segment_size=3,
        disc_root_radius=2,
        bounds_boundary_width=6,
        solver_time_limit=5.0,
    )

    assert set(result.segment_labels.values()) == {1, 2}
    assert not result.forest_edges[1]
    assert not result.forest_edges[2]
    for class_id in (1, 2):
        assert len(result.selected_roots[class_id]) == 1
        root = result.selected_roots[class_id][0]
        assert root.kind == "disc"
        assert root.junction_id is not None
        assert root.port_id is not None
        assert root.path_distance > 0


def test_bounds_root_is_preferred_over_internal_fallback():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 6:36, 0] = 8.0
    logits[30:33, 6:36, 2] = 6.0
    bounds_mask = np.zeros((64, 64), dtype=bool)
    bounds_mask[2:62, 2:62] = True

    result = infer_artery_vein(
        logits,
        bounds_mask=bounds_mask,
        junction_radius=2,
        min_segment_size=3,
        bounds_boundary_width=6,
        solver_time_limit=5.0,
    )

    assert [root.kind for root in result.selected_roots[2]] == ["bounds"]
    assert not result.selected_roots[1]


def test_short_junction_bridge_is_retained_as_connected_microsegment():
    logits = np.full((96, 96, 4), -8.0, dtype=np.float32)
    logits[47:50, 5:91, 0] = 8.0
    logits[47:50, 5:91, 1] = 6.0
    logits[18:49, 39:42, 0] = 8.0
    logits[18:49, 39:42, 1] = 6.0
    logits[48:79, 51:54, 0] = 8.0
    logits[48:79, 51:54, 1] = 6.0

    result = infer_artery_vein(
        logits,
        junction_radius=3,
        min_segment_size=8,
        solver_time_limit=5.0,
    )

    assert len(result.microsegment_ids) == 1
    microsegment_id = next(iter(result.microsegment_ids))
    assert result.segment_labels[microsegment_id] == 1
    assert not any(
        root.segment_id == microsegment_id
        for roots in result.selected_roots.values()
        for root in roots
    )
    incident_edges = [
        edge
        for edges in result.forest_edges.values()
        for edge in edges
        if microsegment_id in edge
    ]
    assert len(incident_edges) >= 2


def _test_port(port_id, angle_degrees, *, thickness=2.0):
    angle = np.deg2rad(angle_degrees)
    return Port(
        id=port_id,
        junction_id=1,
        segment_id=port_id,
        endpoint=0,
        position=(0.0, 0.0),
        tangent=(float(np.cos(angle)), float(np.sin(angle))),
        thickness=thickness,
    )


def _test_segment(segment_id):
    skeleton = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int32)
    return SegmentInfo(
        id=segment_id,
        skeleton=skeleton,
        pixels=skeleton,
        spline=None,
        thickness=2.0,
        vessel_probability=0.99,
        artery_probability=0.9,
        vein_probability=0.1,
        artery_dominant_fraction=1.0,
        vein_dominant_fraction=0.0,
        crossing_fraction=0.0,
        is_microsegment=False,
        junction_contact_count=1,
    )


def test_one_ended_microsegment_port_is_excluded_but_real_bridge_is_retained():
    artifact = _test_segment(1)
    artifact.is_microsegment = True
    artifact.junction_contact_count = 2
    bridge = _test_segment(2)
    bridge.is_microsegment = True
    bridge.junction_contact_count = 2
    regular = _test_segment(3)
    segments = {segment.id: segment for segment in (artifact, bridge, regular)}

    def port(port_id, junction_id, segment_id, endpoint, position=(0.0, 0.0)):
        return Port(
            id=port_id,
            junction_id=junction_id,
            segment_id=segment_id,
            endpoint=endpoint,
            position=position,
            tangent=(1.0, 0.0),
            thickness=2.0,
        )

    ports = {
        10: port(10, 1, 1, 0),
        20: port(20, 1, 2, 0),
        21: port(21, 2, 2, 0, position=(0.0, 3.0)),
        30: port(30, 1, 3, 0),
    }
    junctions = {
        1: JunctionInfo(
            id=1,
            pixels=np.asarray([[0, 0]], dtype=np.int32),
            port_ids=(10, 20, 30),
            crossing_probability=0.5,
        ),
        2: JunctionInfo(
            id=2,
            pixels=np.asarray([[0, 1]], dtype=np.int32),
            port_ids=(21,),
            crossing_probability=0.5,
        ),
    }

    _orient_two_port_microsegments(segments, ports, fit_length=12.0)
    excluded = _exclude_artifact_microsegment_ports(segments, ports, junctions)

    assert excluded == {1}
    assert set(ports) == {20, 21, 30}
    assert ports[20].endpoint == 0
    assert ports[21].endpoint == 1
    assert np.dot(ports[20].tangent, ports[21].tangent) == pytest.approx(-1.0)
    assert junctions[1].port_ids == (20, 30)
    assert junctions[2].port_ids == (21,)


def test_continuation_cost_is_exponential_and_has_100_degree_hard_limit():
    config = AVInferenceConfig()
    first = _test_port(1, 0.0)

    def cost(deflection_degrees):
        second = _test_port(2, 180.0 - deflection_degrees)
        return _direction_cost(
            PortGroup((first.id, second.id)),
            {first.id: first, second.id: second},
            config,
        )

    assert cost(0.0) == pytest.approx(0.0)
    assert cost(45.0) < cost(75.0) < cost(90.0)
    assert cost(90.0) == pytest.approx(config.continuation_angle_cost_at_90_degrees)
    assert cost(95.0) > 3.0 * cost(90.0)
    assert np.isfinite(cost(100.0))
    assert np.isinf(cost(101.0))


def test_bifurcation_children_have_100_degree_local_deflection_limit():
    config = AVInferenceConfig()
    parent_tangent = np.asarray((-1.0, 0.0))

    def child(deflection_degrees):
        angle = np.deg2rad(deflection_degrees)
        return np.asarray((np.cos(angle), np.sin(angle)))

    assert np.isfinite(
        _bifurcation_direction_cost(
            parent_tangent,
            (child(100.0), child(-20.0)),
            config,
        )
    )
    assert np.isinf(
        _bifurcation_direction_cost(
            parent_tangent,
            (child(101.0), child(-20.0)),
            config,
        )
    )


def test_backwards_children_are_removed_from_compound_and_higher_order_configs():
    config = AVInferenceConfig(max_configurations_per_junction=10_000)
    ports = {
        1: _test_port(1, 180.0, thickness=4.0),
        2: _test_port(2, 0.0, thickness=3.0),
        3: _test_port(3, 101.0, thickness=1.5),
        4: _test_port(4, -25.0, thickness=1.5),
        5: _test_port(5, -90.0, thickness=2.0),
    }
    segments = {segment_id: _test_segment(segment_id) for segment_id in ports}
    compound = PortGroup(
        (1, 2, 3, 4),
        parent_port_id=1,
        continuation_port_id=2,
    )
    assert np.isinf(_compound_group_cost(compound, ports, segments, config))

    junction = JunctionInfo(
        id=1,
        pixels=np.asarray([[0, 0]], dtype=np.int32),
        port_ids=tuple(ports),
        crossing_probability=0.01,
    )
    configurations = _make_configurations(junction, ports, segments, config)
    parent_groups = [
        group
        for candidate in configurations
        for group in candidate.groups
        if group.parent_port_id == 1 and len(group.port_ids) >= 3
    ]

    assert parent_groups
    assert all(3 not in group.port_ids for group in parent_groups)


def test_compound_bifurcation_connects_four_artery_ports():
    logits = np.full((96, 96, 4), -8.0, dtype=np.float32)
    logits[46:51, 5:91, 0] = 8.0
    logits[15:81, 47:50, 0] = 8.0
    logits[logits[..., 0] > 0, 1] = 6.0

    without_compound = infer_artery_vein(
        logits,
        junction_radius=3,
        min_segment_size=3,
        enable_compound_bifurcations=False,
        solver_time_limit=5.0,
    )
    with_compound = infer_artery_vein(
        logits,
        junction_radius=3,
        min_segment_size=3,
        enable_compound_bifurcations=True,
        solver_time_limit=5.0,
    )

    assert len(without_compound.selected_roots[1]) == 2
    assert len(with_compound.selected_roots[1]) == 1
    graph = nx.DiGraph(with_compound.forest_edges[1])
    graph.add_nodes_from(with_compound.segment_labels)
    assert nx.number_weakly_connected_components(graph) == 1


def test_six_ports_support_two_overlapping_bifurcations():
    partitions = list(_restricted_partitions(tuple(range(6))))
    assert any(sorted(map(len, partition)) == [3, 3] for partition in partitions)


def test_six_ports_support_compound_bifurcation_plus_continuation():
    partitions = list(_restricted_partitions(tuple(range(6)), max_group_size=4))
    partition = next(
        partition for partition in partitions if sorted(map(len, partition)) == [2, 4]
    )
    configurations = list(_configuration_groups(partition))

    assert configurations
    assert all(
        any(group.continuation_port_id is not None for group in groups)
        for groups in configurations
    )


def test_av_unary_cost_can_be_disabled_for_ablation():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 5:59, 0] = 8.0
    logits[5:33, 30:33, 0] = 8.0
    logits[30:33, 5:30, 1] = 6.0
    logits[30:33, 5:30, 2] = -6.0
    logits[5:30, 30:33, 1] = 6.0
    logits[5:30, 30:33, 2] = -6.0
    logits[30:33, 35:59, 1] = -1.0
    logits[30:33, 35:59, 2] = 1.0

    default, _ = refine_artery_vein_graph_full_with_flips(
        logits,
        junction_radius=2,
        min_segment_size=3,
        solver_time_limit=5.0,
    )
    swapped_logits = logits.copy()
    swapped_logits[..., [1, 2]] = swapped_logits[..., [2, 1]]
    default_swapped, _ = refine_artery_vein_graph_full_with_flips(
        swapped_logits,
        junction_radius=2,
        min_segment_size=3,
        solver_time_limit=5.0,
    )
    no_av_cost, _ = refine_artery_vein_graph_full_with_flips(
        logits,
        junction_radius=2,
        min_segment_size=3,
        av_cost_weight=0.0,
        solver_time_limit=5.0,
    )
    no_av_cost_swapped, _ = refine_artery_vein_graph_full_with_flips(
        swapped_logits,
        junction_radius=2,
        min_segment_size=3,
        av_cost_weight=0.0,
        solver_time_limit=5.0,
    )

    assert default[31, 50] != default_swapped[31, 50]
    np.testing.assert_array_equal(no_av_cost, no_av_cost_swapped)


@pytest.mark.parametrize("mixed_mode", ["side_by_side", "overlap"])
def test_artery_and_vein_can_share_one_physical_segment(mixed_mode):
    shape = (96, 96)
    artery = np.zeros(shape, dtype=bool)
    vein = np.zeros(shape, dtype=bool)
    paths = (
        (artery, ((27, 5), (49, 31), (49, 65), (27, 91))),
        (vein, ((69, 5), (52, 31), (52, 65), (69, 91))),
    )
    for class_mask, points in paths:
        for start, end in pairwise(points):
            rows, cols = line(*start, *end)
            class_mask[rows, cols] = True
        class_mask[:] = binary_dilation(class_mask, disk(2))

    vessel = artery | vein
    logits = np.full((*shape, 4), -8.0, dtype=np.float32)
    logits[vessel, 0] = 8.0
    logits[artery & ~vein, 1] = 6.0
    logits[artery & ~vein, 2] = -6.0
    logits[vein & ~artery, 1] = -6.0
    logits[vein & ~artery, 2] = 6.0
    logits[artery & vein, 1:3] = 0.0
    if mixed_mode == "overlap":
        rows, columns = np.indices(shape)
        overlap = vessel & (columns >= 35) & (columns <= 61) & np.isin(rows, (47, 54))
        logits[overlap, 1] = 6.0
        logits[overlap, 2] = -6.0
        logits[overlap, 3] = 8.0

    disc_mask = np.zeros(shape, dtype=bool)
    disc_mask[15:81, :7] = True
    bounds_mask = np.ones(shape, dtype=bool)

    result = infer_artery_vein(
        logits,
        disc_mask=disc_mask,
        bounds_mask=bounds_mask,
        junction_radius=3,
        min_segment_size=3,
        disc_root_radius=3,
        bounds_boundary_width=3,
        solver_time_limit=5.0,
        solver_mip_rel_gap=0.0,
    )

    center_segment_id = int(result.segment_id_image[50, 48])
    assert center_segment_id in result.mixed_segment_ids
    for class_id in (1, 2):
        assert any(center_segment_id in edge for edge in result.forest_edges[class_id])

    expected = np.zeros(shape, dtype=np.uint8)
    retained_vessel = vessel & ~disc_mask
    expected[retained_vessel] = np.where(
        logits[..., 1][retained_vessel] >= logits[..., 2][retained_vessel],
        1,
        2,
    )
    crossing = retained_vessel & (logits[..., 3] >= 0)
    expected[crossing] = 3
    mixed_pixels = result.segment_id_image == center_segment_id
    np.testing.assert_array_equal(result.mask[mixed_pixels], expected[mixed_pixels])

    without_mixed = infer_artery_vein(
        logits,
        disc_mask=disc_mask,
        bounds_mask=bounds_mask,
        junction_radius=3,
        min_segment_size=3,
        disc_root_radius=3,
        bounds_boundary_width=3,
        enable_mixed_segments=False,
        solver_time_limit=5.0,
        solver_mip_rel_gap=0.0,
    )
    assert without_mixed.segment_labels[center_segment_id] in {1, 2}
