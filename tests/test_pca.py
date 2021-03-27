# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest
from motmot.geometry import orthogonal_bases, get_components_zipped

from tomial_odometry._pca import PCA

pytestmark = pytest.mark.order(2)


def assert_eigenvectors_equal(x, y):
    """Validate an array of eigenvectors."""
    # Eigenvector sign is arbitrary so a pair of eigenvectors x[0] and y[0]
    # may satisfy x[0] == y[0] or x[0] == -y[0].
    assert (np.isclose(x, y).all(-1) | np.isclose(x, -y).all(-1)).all()


# Generate a cuboid shaped cloud of points.
# This cloud has 6 x 6 x 6 points spanning 0.06 x 0.6 x 6.
_x = np.arange(6)
simple_points = np.c_[tuple(
    i.ravel() for i in np.meshgrid(_x / 100, _x / 10, _x))]


def test_trivial():
    # Feed `simple_points` straight into PCA.
    self = PCA(simple_points)
    assert self.eigenvalues.tolist() == sorted(self.eigenvalues)

    # The shortest, middle-length and widest directions are already the
    # x, y, and z axes in that order so its eigenvectors should just be those 3
    # basis vectors i.e. eye(3).
    assert_eigenvectors_equal(np.eye(3), self.eigenvectors)


# Remap `simple_points` into some wonky coordinate system.
# `wonky_axes` are the new basis vectors for this coordinate system.
wonky_axes = np.array(orthogonal_bases([2, 3, 1]))
wonky_points = simple_points @ wonky_axes


def test_wonky():
    # Sanity check I got the above matrix multiplication correct.
    assert get_components_zipped(wonky_points, *wonky_axes) \
        == pytest.approx(simple_points)

    self = PCA(wonky_points)
    assert self.eigenvalues.tolist() == sorted(self.eigenvalues)
    assert_eigenvectors_equal(self.eigenvectors, wonky_axes)


def test_weighted():
    """Test weighted PCA by mixing `simple_points` with `wonky_points` but
    using weights of 1s and 0s to un-mix them."""
    mixed = np.append(wonky_points, simple_points, axis=1).reshape((-1, 3))
    is_simple = np.arange(len(mixed)) % 2

    # Ignore wonky, include simple.
    self = PCA(mixed, weights=is_simple)
    assert_eigenvectors_equal(self.eigenvectors, np.eye(3))

    # Ignore simple, include wonky.
    self = PCA(mixed, weights=1 - is_simple)
    assert_eigenvectors_equal(self.eigenvectors, wonky_axes)

    # Weights should be allowed to be shape (n, 1) as well as (n,).
    self = PCA(mixed, weights=is_simple[:, np.newaxis])
    assert_eigenvectors_equal(self.eigenvectors, np.eye(3))

    # A constant weight multiplier shouldn't make any difference.
    self = PCA(mixed, weights=is_simple * 10)
    assert_eigenvectors_equal(self.eigenvectors, np.eye(3))
