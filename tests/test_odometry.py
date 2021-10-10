# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest
from motmot import Mesh, geometry
from tomial_tooth_collection_api import model

from tomial_odometry import Odometry

# This model is oriented with TRIOS's rather odd convention of:
# - X axis: left
# - Y axis: vertically up
# - Z axis: forwards
mesh = Mesh(model('3D scan model_Mandibular_export'))


def test_walk_through():
    """Walk through each step of the mini-algorithm, validating using a model
    whose orientation is known."""

    self = Odometry._debug(mesh, "L")

    self._apply_pca()
    # Check each eigenvector is assigned to the correct direction.
    # Eigenvectors have no sign and could therefore point in the opposite
    # direction - hence the abs().
    assert abs(self._eX[0]) > .9
    assert abs(self._eZ[1]) > .9
    assert abs(self._eY[2]) > .9

    # --- Sign checking ---

    # Vertical axis.
    self._check_eZ_sign()
    assert self._eZ[1] > .9
    assert self.occlusal[1] > .9

    self.arch_type = "U"
    assert self._eZ[1] > .9
    assert self.occlusal[1] < -.9

    self._check_eZ_sign()
    assert self.occlusal[1] > .9
    assert self._eZ[1] < -.9

    self.occlusal *= -1
    self._check_eZ_sign()
    assert self.occlusal[1] > .9
    assert self._eZ[1] < -.9

    self.arch_type = "L"
    self._check_eZ_sign()
    assert self._eZ[1] > .9
    assert self.occlusal[1] > .9

    # Forwards/backwards axis.
    self._check_eY_sign()
    assert self._eY[2] > .9
    self._check_eY_sign()
    assert self._eY[2] > .9
    self._eY[:] *= -1
    self._check_eY_sign()
    assert self._eY[2] > .9

    # Left/right axis.
    self._check_eX_sign()
    assert self._eX[0] < -.9
    self._check_eX_sign()
    assert self._eX[0] < -.9
    self._eX[:] *= -1
    self._check_eX_sign()
    assert self._eX[0] < -.9

    # Finish off the rest of the algorithm.
    # This will redundantly rerun the steps above but that doesn't matter.
    self._run()

    assert self.right[0] < -.9
    assert self.forwards[2] > .9
    assert self.up[1] > .9

    assert np.array_equal(self.axes, [self.right, self.forwards, self.up])


def test_univ_vector_types():
    self = Odometry(mesh, "L")
    assert isinstance(self.up, geometry.UnitVector)
    assert isinstance(self.forwards, geometry.UnitVector)
    assert isinstance(self.occlusal, geometry.UnitVector)

    self.up = [1, 0, 0]
    assert isinstance(self.up, geometry.UnitVector)
    assert isinstance(self.occlusal, geometry.UnitVector)
    self.forwards = [0, 2, 1]
    assert isinstance(self.forwards, geometry.UnitVector)

    self = Odometry.dummy("L")
    assert isinstance(self.forwards, geometry.UnitVector)
    assert isinstance(self.occlusal, geometry.UnitVector)


def test_normalised():
    """After removing the center of mass and applying the rotation matrix given
    by an Odometry, applying the odometry algorithm again should give a
    center of mass of (0, 0, 0) and rotation matrix eye(3)."""
    self = Odometry(mesh, "L")

    normalised = mesh.copy()
    normalised.translate(-self.center_of_mass)
    normalised.rotate_using_matrix(self.axes.T)

    simple_odom = Odometry(normalised, "L")
    assert simple_odom.axes == pytest.approx(np.eye(3), abs=1e-5)
    assert simple_odom.center_of_mass == pytest.approx(0, abs=1e-5)


def test_dummy_and_horizontal_conversions():
    m = [[0, 1, 0], [.8, 0, .6], [.6, 0, -.8]]
    assert np.linalg.det(m) == 1

    self = Odometry.dummy("U", m, [1, 2, 3])
    assert self.up.tolist() == m[2]

    # to_horizontal(v) should be [m[0] · v, m[1] · v]
    assert self.to_horizontal([10, 100, 1000]).tolist() == [100, 608]
    assert self.to_horizontal([[1, 2, 3], [4, 5, 6]]) \
        == pytest.approx(np.array([[2, 2.6], [5, 6.8]]))

    assert self.from_horizontal([1, 0]).tolist() == m[0]
    assert self.from_horizontal([0, 2]).tolist() == [i * 2 for i in m[1]]

    points_2d = [100, 608]
    points_3d = self.from_horizontal(points_2d)
    assert self.up.tolist() == m[-1]
    assert (-self.occlusal).tolist() == m[-1]
    assert self.up(points_3d) == 0
    assert [self.right(points_3d), self.forwards(points_3d)] == points_2d

    points_3d = self.from_horizontal(points_2d, up=10)
    assert self.right(points_3d) == pytest.approx(points_2d[0])
    assert self.forwards(points_3d) == pytest.approx(points_2d[1])
    assert self.up(points_3d) == pytest.approx(10)

    points_3d = self.from_horizontal(points_2d, occlusal=99)
    assert self.occlusal(points_3d) == pytest.approx(99)


def test_hints():
    """Test sign hints by lying about the signs and checking the lies propagate.
    """
    hints = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    self = Odometry(mesh, "L", hints=hints)
    assert self.right[0] < -.9
    assert self.forwards[2] < -.9
    assert self.up[1] < -.9

    self._check_eZ_sign()
    self._check_eY_sign()
    self._check_eX_sign()

    assert self.right[0] < -.9
    assert self.forwards[2] > .9
    assert self.up[1] > .9
