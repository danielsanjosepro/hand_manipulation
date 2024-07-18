"""Some util functions for grasping."""

import numpy as np

from mujoco import MjContact
from scipy.spatial.transform import Rotation


def get_transposed_grasp_matrix(
    contact: MjContact,
) -> np.ndarray:
    """Get transposed grasp matrix. based on the contact information."""
    position = contact.pos.copy()
    normal = contact.frame[:3].copy()

    R = Rotation.align_vectors(normal, np.array([1, 0, 0]))[0].as_matrix()
    S = skew(position)
    P = np.vstack(
        [
            np.hstack([np.eye(3), np.zeros((3, 3))]),
            np.hstack([S, np.eye(3)]),
        ]
    )
    R_T_stacked = np.vstack(
        [
            np.hstack([R.T, np.zeros((3, 3))]),
            np.hstack([np.zeros((3, 3)), R.T]),
        ]
    )

    return R_T_stacked @ P.T


def skew(vec: np.ndarray) -> np.ndarray:
    """Get skew-symmetric matrix of a vector."""
    vec_flat = vec.flatten()
    return np.array(
        [
            [0, -vec_flat[2], vec_flat[1]],
            [vec_flat[2], 0, -vec_flat[0]],
            [-vec_flat[1], vec_flat[0], 0],
        ]
    )
