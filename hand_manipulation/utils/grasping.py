"""Some util functions for grasping."""

import numpy as np

from scipy.spatial.transform import Rotation


def get_rotation_matrix_from_normal(normal: np.ndarray, stacked: bool = False) -> np.ndarray:
    """Get rotation matrix from normal vector."""
    R = Rotation.align_vectors(normal, np.array([1, 0, 0]))[0].as_matrix()
    return np.vstack([np.hstack([R, np.zeros((3, 3))]), np.hstack([np.zeros((3, 3)), R])]) if stacked else R


def get_transposed_grasp_matrix(
    position: np.ndarray,
    object_position: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Get transposed grasp matrix. based on the contact information."""
    r_obj_c = position - object_position

    S = skew(r_obj_c)
    P = np.vstack(
        [
            np.hstack([np.eye(3), np.zeros((3, 3))]),
            np.hstack([S, np.eye(3)]),
        ]
    )
    R_T_stacked = get_rotation_matrix_from_normal(normal, stacked=True).T
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
