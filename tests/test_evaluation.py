import torch
from torch import nn

from util.evaluation import predict_density_sliding_window, sliding_window_starts


class PixelwiseDensity(nn.Module):
    def forward(self, images, query=None):
        del query
        return images[:, 0] * 3.0 + 0.5


def test_sliding_window_starts_cover_far_edge():
    assert sliding_window_starts(300, 384, 128) == [0]
    assert sliding_window_starts(384, 384, 128) == [0]
    assert sliding_window_starts(700, 384, 128) == [0, 128, 256, 316]


def test_overlap_averaging_reconstructs_pixelwise_prediction():
    image = torch.rand(1, 3, 517, 701)
    expected = image[0, 0] * 3.0 + 0.5
    actual = predict_density_sliding_window(
        PixelwiseDensity(), image, window_size=384, stride=128
    )
    assert actual.shape == (517, 701)
    torch.testing.assert_close(actual, expected)

