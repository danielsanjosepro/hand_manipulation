"""Test script for grasp_matrix.py."""

import numpy as np
import pytest
import logging
from rich.logging import RichHandler

from hand_manipulation.utils.grasping import get_rotation_matrix_from_normal, get_transposed_grasp_matrix

# Set up logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_first_get_grasp_matrix():
    """Test the get_transposed_grasp_matrix function."""
    position = np.array([0.0, 1.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    object_position = np.array([1.0, 1.0, 0.0])
    grasp_matrix = get_transposed_grasp_matrix(position, object_position, normal).T

    twist_object = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    expected_twist_hand = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(grasp_matrix.T @ twist_object, expected_twist_hand), f"Test failed: expected {expected_twist_hand}, got {grasp_matrix.T @ twist_object}"

def test_second_get_grasp_matrix():
    """Test the get_transposed_grasp_matrix function."""
    position = np.array([0.0, 1.0, 0.0])
    normal = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0])
    object_position = np.array([1.0, 1.0, 0.0])
    grasp_matrix = get_transposed_grasp_matrix(position, object_position, normal).T

    twist_object = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    expected_twist_hand = np.array([-np.cos(np.pi/4), -np.sin(np.pi/4), 0.0, 0.0, 0.0, 1.0])
    twist_contact = grasp_matrix.T @ twist_object
    assert np.allclose(grasp_matrix.T @ twist_object, expected_twist_hand), f"Test failed: expected {expected_twist_hand}, got {grasp_matrix.T @ twist_object}"
    expected_twist_reference = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    R = get_rotation_matrix_from_normal(normal, stacked=True)
    computed_twist = R @ twist_contact
    assert np.allclose(computed_twist, expected_twist_reference), f"Test failed: expected {expected_twist_reference}, got {computed_twist}"


if __name__ == "__main__":
    pytest.main(["-s", "test_grasp_matrix.py"])

