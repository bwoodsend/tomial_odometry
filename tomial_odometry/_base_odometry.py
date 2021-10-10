# -*- coding: utf-8 -*-
"""
"""

import abc
import numpy as np
from motmot import geometry


class BaseOdometry(abc.ABC):

    arch_type: str
    """Either of:

    * :py:`'U'` for a maxillary (upper) jaw.
    * :py:`'L'` for a mandibular (lower) jaw.

    """

    def __init__(self, arch_type, axes=None, com=None):
        """Create an odometry explicitly from a center of mass and basis
         vectors.

         Args:
             arch_type:
                 ``'U'`` for upper jaw, ``'L'`` for lower jaw.
             axes:
                 Basis vectors representing each of the directions in
                 :attr:`axes`. Defaults to :py:`numpy.eye(3)`.
             com:
                 center of mass, defaults to the origin :py:`[0, 0, 0]`.

        """
        self.center_of_mass = np.zeros(3) if com is None else com
        axes = np.eye(3) if axes is None else axes

        for (name, axis) in zip(self.names, axes):
            setattr(self, name, geometry.UnitVector(axis))

        assert arch_type in "UL"
        self.arch_type = arch_type

    @abc.abstractmethod
    def names(self):
        """The names of the axes given by :attr:`axes`."""
        pass

    @property
    def axes(self):
        """The core unit-vectors listed in :attr:`names` as rows of a
        3x3 matrix.

        This matrix may be used to normalise and unnormalise a set of points. ::

            # Transform points into a simplified coordinate system.
            normalised = points @ odom.axes.T
            # Direct inspection of (x, y, z) values is now meaningful because it
            # is independent of initial orientation.
            x = normalised[..., 0]  # Latitudinal components.
            y = normalised[..., 1]  # Longitudinal components.
            z = normalised[..., 2]  # Vertical components.

            # Get back to the original coordinate system using:
            points = normalised @ odom.axes

        Because all its rows are perpendicular to each other, this matrix
        satisfies::

            axes @ axes.T == axes.T @ axes == numpy.eye(3)

        And::

            numpy.linalg.det(axes) == 1

        i.e. Its transpose is its own inverse.
        And its determinant will always be 1.

        """
        return np.array([getattr(self, val) for val in self.names])

    def __setattr__(self, key, value):
        if key in self.names:
            value = geometry.UnitVector(value)
        super().__setattr__(key, value)
