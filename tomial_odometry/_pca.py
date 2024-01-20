import numpy as np
from motmot import geometry


class PCA(object):
    """Perform Principle Component Analysis (PCA) on a set of points.

    Optionally weights can be applied to give some points more priority than
    others. Direction vectors are stored as row vectors in
    :attr:`eigenvectors` matrix in order of least to highest covariance.

    """
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray

    def __init__(self, points, weights=None):

        # The floating point error of 32 bit floats accumulates quite severely
        # in this algorithm.
        points = np.asarray(points, dtype=np.float64)

        # PCA is a pretty straight forward recipe.

        # Find the center of mass and eliminate it.
        self.center_of_mass = geometry.center_of_mass(points, weights)
        differences = points - self.center_of_mass

        # Apply weights if any are provided.
        if weights is not None:
            if weights.ndim == 1:
                weights = weights[:, np.newaxis]
            differences *= weights

        # Matrix multiply the `differences` with itself.
        # This is a 3x3 matrix (3 for 3D).
        self.covariance_matrix = differences.T @ differences

        # Perform eigen decomposition on the covariance matrix.
        self.eigenvalues, eigenvectors = \
            np.linalg.eigh(self.covariance_matrix)
        self.eigenvectors = eigenvectors.T
