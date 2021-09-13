# -*- coding: utf-8 -*-
"""
"""

from typing import Tuple

import numpy as np
from motmot import Mesh, geometry

from tomial_odometry._base_odometry import BaseOdometry
from tomial_odometry._pca import PCA


class Odometry(BaseOdometry):
    """Find the position and orientation of any dental model.

    Orientation is described with unit-vectors. The following directions are
    available as attributes. The unit-vectors are of
    :class:`motmot.geometry.UnitVector` type.

    +-------------------+------------------------------------------------------+
    | Attribute         | Description                                          |
    +===================+======================================================+
    | :attr:`right`     | Both from the patient's perspective.                 |
    +-------------------+                                                      |
    | :attr:`forwards`  |                                                      |
    +-------------------+------------------------------------------------------+
    | :attr:`up`        | To the roof irregardless of                          |
    |                   | whether the model is maxillary of mandibular.        |
    +-------------------+------------------------------------------------------+
    | :attr:`occlusal`  | Alias for the direction of the teeth. Up if it is    |
    |                   | a lower jaw or down if it is an upper jaw.           |
    +-------------------+------------------------------------------------------+

    Position is described with the model's center of mass which is the mean
    of all the mesh's vertices (which isn't much use in practice).

    Use of this class makes for more flexible code than direct inspection of
    the raw x, y or z components. The following example takes a sub-sample of
    only the top (occlusally) 5mm so as to only include the tips of teeth. It
    is fully independent of initial orientation, the scanner used and whether
    the jaw is upper or lower.

    .. code-block:: python

        from motmot import Mesh
        from tomial_odometry import Odometry

        mesh = Mesh("my teeth.stl")
        odom = Odometry(mesh)

        heights = odom.occlusal(mesh.centers)
        include_mask = heights > heights.max() - 5.0

        cropped = mesh.crop(include_mask)

    .. note::

        *Odometry* is a fancy collective term, borrowed from robotics meaning
        position and orientation.

    """
    _eX: geometry.UnitVector
    _eY: geometry.UnitVector
    _eZ: geometry.UnitVector

    @property
    def right(self) -> geometry.UnitVector:
        """Right from the patient's perspective. Use ``-right`` to get a *left*
         vector."""
        return self._eX

    @property
    def forwards(self) -> geometry.UnitVector:
        """Out the patient's mouth. Or *anterior* in techincal speak. Use
        ``-forwards`` to get a *backwards* *(posterior)* vector."""
        return self._eY

    @property
    def up(self) -> geometry.UnitVector:
        """Vertical up or to the roof. Use ``-up`` to get a *down* vector."""
        return self._eZ

    @right.setter
    def right(self, x):
        self._eX = geometry.UnitVector(x)

    @forwards.setter
    def forwards(self, x):
        self._eY = geometry.UnitVector(x)

    @up.setter
    def up(self, x):
        self._eZ = geometry.UnitVector(x)

    @property
    def occlusal(self) -> geometry.UnitVector:
        """:attr:`up` for a mandibular model, down for a maxillary model."""
        return self.up if self.arch_type == "L" else -self.up

    @occlusal.setter
    def occlusal(self, x):
        self._eZ = geometry.UnitVector(x) * (1 if self.arch_type == "L" else -1)

    def __init__(self, mesh: Mesh, arch_type=None, hints=None):
        """
        Args:
            mesh:
                An opened STL file of a dental model.
            arch_type:
                ``'U'`` for upper jaw, ``'L'`` for lower jaw.

        The **arch_type**, if not explicitly specified then it is ???

        """
        self._init(mesh, arch_type, hints)
        self._run()

    @classmethod
    def _debug(cls, mesh, arch_type=None, hints=None):
        self = cls.__new__(cls)
        self._init(mesh, arch_type, hints)
        return self

    def _init(self, mesh, arch_type, hints):
        self._mesh = mesh
        self.arch_type = arch_type
        if hints is not None:
            assert hints.shape == (3, 3)
        self.hints = hints

    def _run(self):
        """Run all steps."""
        # I've written each function in the order they get used so reading
        # this process should just be a case of scrolling down though this
        # class.

        # Use PCA to get 3 perpendicular (to each other) vectors. These
        # will be the  basis vectors
        self._apply_pca()

        if self.hints is None:
            # Now to check signs for each vector.
            self._check_eZ_sign()
            self._check_eY_sign()
            self._check_eX_sign()
        else:
            for (uv, hint) in zip([self._eX, self._eY, self._eZ], self.hints):
                uv[:] *= np.sign(uv(hint))

        # Check we haven't accidentally mirrored it.
        # Read as == 1 with rounding tolerance.
        # Would be -1 if mirrored.
        assert 1.001 > np.linalg.det(self.axes) > 0.999

        self._adjust_eZ_to_tips()

    def _apply_pca(self):
        """Run PCA to get the shortest, middlemost and longest axes.

        I'm choosing to define the axes using engineering convention:

        * eX  left -> right (patient's left and right)
        * eY  back -> front (going out of the patients mouth)
        * eZ  bottom -> top

        A dental model is longer than it is tall and is wider than it is long.
        So in ascending order of covariance (as numpy.linalg.eigh returns) they
        should be eZ, eY, eX.

        """
        weights = (.05 - self._mesh.areas).clip(min=0)
        pca = PCA(self._mesh.centers, weights)

        self.center_of_mass = pca.center_of_mass
        self._eZ, self._eY, self._eX = map(geometry.UnitVector,
                                           pca.eigenvectors)

    def _check_eZ_sign(self):
        """Check/correct the sign of the vertical axis.

        The triangle density is much higher on the occlusal surface so a mean of
        ``mesh.units`` gives a decent approximation of occlusal.
        """
        # Get an approximate occlusal from the mesh's unit normals.
        self._mesh_occlusal = geometry.UnitVector(
            [i.sum() for i in self._mesh.units.T])

        # Compare with the PCA's occlusal.
        agreement = self._mesh_occlusal(self.occlusal)

        assert abs(agreement) > 0.75, agreement  # typically > .95
        self._eZ[:] *= np.sign(agreement)  # and swap sign if necessary

    def _check_eY_sign(self):
        """Check/correct the sign of the forwards/backwards axis.

        Fit a weighted quadratic curve to the horizontal components of every
        mesh polygons' center. This curve will approximate the jaw line. If
        the sign is correct, curve should be ⋂ shaped (negative x² coefficient).
        If it is ⋃ shaped then the y-axis needs flipping.

        Note that the fit is quite poor and shouldn't be used for anything
        precise.
        """
        # Extract the horizontal components, removing the center of mass
        x, y = (e(self._mesh.centers) - e(self.center_of_mass)
                for e in (self._eX, self._eY))

        # I tried a few different weightings to improve the fit.
        # They all get roughly the same results.

        # Prioritise the occlusal facing triangles
        # weights = geometry.inner_product(self._normals, self._mesh_occlusal)

        # Prioritise the more occlusal points
        weights = self._mesh_occlusal(self._mesh.centers)

        # Don't allow negative weights - turns out they don't do any harm.
        # weights.clip(min=0, out=weights)

        # Prioritise non occlusal facing triangles.
        # This is supposed to capture the labial and lingual vertical surfaces.
        weights *= geometry.magnitude_sqr(
            np.cross(self._mesh.units, self._mesh_occlusal))

        weights -= np.min(weights)
        weights /= np.mean(weights)

        # Fit a quadratic curve to the points with a weighted fitting.
        poly = np.polynomial.Polynomial.fit(x, y, 2, w=weights)

        # If x² coefficient is positive:
        if poly.convert().coef[2] > 0:
            # Flip eY
            self._eY = -self._eY

    def _check_eX_sign(self):
        """Finally eX is just determined so as not to mirror the mesh. This must
        be done after checking eZ and eY because it uses them."""

        # If rotation matrix mirrors then reverse eX
        self._eX[:] *= np.sign(np.linalg.det(self.axes))

    def _adjust_eZ_to_tips(self):
        """Tilt the model forwards/backwards so that the tips of teeth are at
        the same height.

        PCA's vertical is only approximate. This step improves its accuracy
        by fitting a line across the top of the model then adjusting
        :attr:`forwards` and :attr:`occlusal` so that this line is horizontal.
        """
        points = self._mesh.centers
        ys = self.forwards(points)
        heights = self.occlusal(points)

        min_height = heights.min()

        bins = np.arange(ys.min() - 1, ys.max() + 1)
        args = np.digitize(ys, bins)

        # These lines just find the max height in each bin.
        max_heights = np.full_like(bins, min_height)
        np.maximum.at(max_heights, args, heights)

        weights = np.abs((bins - bins[::-1]))
        weights = weights.max() - weights
        weights *= (max_heights - max_heights.min())**6
        yz_line = np.polynomial.Polynomial.fit(bins, max_heights, 1, w=weights)
        yz_forward_tangent = geometry.UnitVector([1, yz_line.deriv()(0)])

        new_forwards_3d = geometry.UnitVector(
            yz_forward_tangent[0] * self.forwards \
            + yz_forward_tangent[1] * self.occlusal)

        eZ = np.cross(self.right, new_forwards_3d)
        self._eZ = geometry.UnitVector(self._eZ.matched_sign(eZ))
        self._eY = new_forwards_3d

    def to_horizontal(self, points) -> np.ndarray:
        """Extract the horizontal components of some **points**.

        More precisely, find the projections in the directions :attr:`right`
        and :attr:`forwards`.

        Args:
            points: A point, array of points, array of arrays of points etc.
        Returns:
            Horizontal components. An array with :py:`.shape[-1] == 2`.

        """
        return geometry.get_components_zipped(points, self.right, self.forwards)

    def from_horizontal(self, points_2d, up=None, occlusal=None):
        """Reconstruct points from their horizontal projections as returned
        by :meth:`to_horizontal`.

        Args:
            points_2d:
                The projections in the directions :attr:`right` and
                :attr:`forwards`. Should be an array with :py:`shape[-1] == 2`.
            up:
                The projection(s) in the :attr:`up` direction, defaults to
                ``0.0``.
            occlusal:
                The projection(s) in the :attr:`occlusal` direction,
                defaults to ``0.0``.
        Returns:
            Remapped points. An array with :py:`shape[-1] == 3`.

        More fine-grained control over what happens to the vertical axis can
        be achieved by feeding the output of this method to the
        :meth:`~motmot.geometry.UnitVector.with_` method of :attr:`up` or
        :attr:`occlusal`.

        """
        out = points_2d @ np.array([self.right, self.forwards])
        if up is not None:
            out += self.up * np.array(up)[..., np.newaxis]
        if occlusal is not None:
            out += self.occlusal * np.array(occlusal)[..., np.newaxis]
        return out

    names: Tuple[str, str, str] = ("right", "forwards", "up")
    """The names of the axes given by :attr:`axes`."""
