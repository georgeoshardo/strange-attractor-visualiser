import numpy as np

from strange_attractor_visualiser.core.density import binned_density


def test_binned_density_is_normalised_and_finds_dense_regions():
    dense_x = np.zeros(50)
    dense_y = np.zeros(50)
    sparse_x = np.array([10.0])
    sparse_y = np.array([10.0])

    density = binned_density(
        np.concatenate([dense_x, sparse_x]),
        np.concatenate([dense_y, sparse_y]),
        bins=32,
    )

    assert density.shape == (51,)
    assert np.isfinite(density).all()
    assert density.min() >= 0.0
    assert density.max() <= 1.0
    assert density[0] > density[-1]


def test_binned_density_handles_degenerate_coordinates():
    density = binned_density(np.ones(5), np.ones(5), bins=32)

    assert np.array_equal(density, np.zeros(5))


def test_binned_density_handles_empty_input():
    density = binned_density(np.array([]), np.array([]), bins=32)

    assert density.shape == (0,)
