import pytest
import torch

from rtnls_inference.ensembles.ensemble_halo_segmentation import (
    halo_sliding_window_inference,
)


@pytest.mark.parametrize("members", [False, True])
def test_halo_tiler_reconstructs_center_crops_without_losing_border_pixels(members):
    image = torch.arange(2 * 3 * 19 * 27, dtype=torch.float32).reshape(2, 3, 19, 27)

    def predictor(tiles):
        centers = tiles[..., 2:-2, 2:-2]
        return torch.stack([centers, centers + 1]) if members else centers

    output = halo_sliding_window_inference(
        image,
        predictor,
        context_size=12,
        output_size=8,
        overlap=0.5,
        sw_batch_size=3,
    )
    expected = torch.stack([image, image + 1], dim=1) if members else image
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "prediction, error, message",
    [
        (None, TypeError, "halo prediction.*torch.Tensor"),
        (torch.zeros(1, 1, 4, 4, dtype=torch.int64), TypeError, "floating point"),
        (torch.zeros(1, 1, 4), ValueError, "Halo predictor must return"),
        (
            torch.zeros(2, 1, 4, 4),
            ValueError,
            "halo prediction batch size must be 1, got 2",
        ),
        (torch.zeros(1, 1, 3, 4), ValueError, "Halo predictor returned"),
    ],
)
def test_halo_predictor_reports_malformed_tile_outputs(prediction, error, message):
    with pytest.raises(error, match=message):
        halo_sliding_window_inference(
            torch.zeros(1, 3, 4, 4),
            lambda image: prediction,
            context_size=8,
            output_size=4,
        )


def test_halo_predictor_rejects_member_count_changes_between_tiles():
    calls = 0

    def predictor(image):
        nonlocal calls
        calls += 1
        return torch.zeros(calls, len(image), 2, 4, 4)

    with pytest.raises(ValueError, match="member/channel counts must remain"):
        halo_sliding_window_inference(
            torch.zeros(1, 3, 8, 8),
            predictor,
            context_size=8,
            output_size=4,
        )
