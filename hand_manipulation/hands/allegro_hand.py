"""Class containing functions to control the Allegro hand in the MuJoCo environment."""

import logging
from typing import List, Tuple

import mujoco
from rich.logging import RichHandler
import numpy as np
from transforms3d.quaternions import quat2mat

from utils.grasping import get_transposed_grasp_matrix
from utils.control import PIDController
from utils.mujoco_utils import MujocoModelNames

# Set up the logging
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)


class AllegroLeftHandActuators:
    """Enum class for the different actuator types in the Shadow Hand."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ):
        """Initialize the Shadow Left Hand Actuators."""
        self.model = model
        self.data = data

    def get_actuator_speeds(self, joint_speeds: np.ndarray) -> np.ndarray:
        """Get the actuator speeds from the joint speeds."""
        return joint_speeds[self.j2a_ids[1:] - 1]

    def actuator_names(self, digit: str) -> List[str]:
        """Return the joint names of the digit."""
        if digit == "thumb":
            return ["tha0", "tha1", "tha2", "tha3"]
        elif digit == "index":
            return ["ffa0", "ffa1", "ffa2", "ffa3"]
        elif digit == "middle":
            return ["mfa0", "mfa1", "mfa2", "mfa3"]
        elif digit == "ring":
            return ["rfa0", "rfa1", "rfa2", "rfa3"]
        else:
            raise ValueError(f"Invalid digit: {digit}")

    def joint_names(self, digit: str) -> List[str]:
        """Return the joint names of the digit."""
        if digit == "thumb":
            return ["thj0", "thj1", "thj2", "thj3"]
        elif digit == "index":
            return ["ffj0", "ffj1", "ffj2", "ffj3"]
        elif digit == "middle":
            return ["mfj0", "mfj1", "mfj2", "mfj3"]
        elif digit == "ring":
            return ["rfj0", "rfj1", "rfj2", "rfj3"]
        else:
            raise ValueError(f"Invalid digit: {digit}")

    def actuators(self, digit: str) -> List[int]:
        """Get sorted actuator ids for a given digit."""
        return np.sort([self.model.actuator(actuator).id for actuator in self.actuator_names(digit)])

    def joints(self, digit: str) -> List[int]:
        """Get sorted joint ids for a given digit."""
        return np.sort([self.model.joint(joint).id for joint in self.joint_names(digit)])


class AllegroHand:
    """Class to control the Allegro Hand in the MuJoCo environment."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        hand_type: str = "left",
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.001,
        integral_clip: float = np.inf,
    ) -> None:
        """Initialize the Allegro Hand class."""
        self.model = model
        self.data = data
        self.model_names = MujocoModelNames(model)

        mujoco.mj_kinematics(model, data)
        self.actuators = AllegroLeftHandActuators(model, data) if hand_type == "left" else None

        # PID controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.controllers = {
            "thumb": PIDController(self.kp, self.ki, self.kd),
            "index": PIDController(self.kp, self.ki, self.kd),
            "middle": PIDController(self.kp, self.ki, self.kd),
            "ring": PIDController(self.kp, self.ki, self.kd),
        }

        self.digit_tips = {
            "thumb": "th_tip_top",
            "index": "ff_tip_top",
            "middle": "mf_tip_top",
            "ring": "rf_tip_top",
        }

        self.target_positions = {
            "thumb": np.zeros(3),
            "index": np.zeros(3),
            "middle": np.zeros(3),
            "ring": np.zeros(3),
        }

        self.target_orientations = {
            "thumb": None,
            "index": None,
            "middle": None,
            "ring": None,
        }

        self.grasp_matrix = None

    @classmethod
    def from_xml_path(cls, xml_path: str) -> "AllegroHand":
        """Load the shadow hand from an XML path."""
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return cls(model, data)

    def get_digit_jacobian(
        self,
        digit: str,
        pos: np.ndarray = None,
        id: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the Jacobian of the digit tip."""
        assert digit in self.digit_tips.keys(), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"
        if pos is None:
            pos = self.data.body(self.digit_tips[digit]).xpos.copy()
        if id is None:
            id = self.data.body(self.digit_tips[digit]).id

        jacobian_translational = np.zeros((3, self.model.nv))
        jacobian_rotational = np.zeros((3, self.model.nv))

        mujoco.mj_jac(self.model, self.data, jacobian_translational, jacobian_rotational, pos, id)

        # We only return the relevant columns of the Jacobian corresponding to the digit joints
        return jacobian_translational[:, self.actuators.joints(digit)], jacobian_rotational[
            :, self.actuators.joints(digit)
        ]

    def update_targets(self, target_positions: dict = None, target_orientations: dict = None) -> None:
        """Set the target positions and orientations of the digit tips if provided."""
        if target_positions is not None:
            for digit, target_position in target_positions.items():
                assert (
                    digit in self.digit_tips.keys()
                ), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"
                assert target_position.shape == (
                    3,
                ), f"Invalid target position shape: {target_position.shape}, must be (3,)"
                self.target_positions[digit] = target_position

        if target_orientations is not None:
            for digit, target_orientation in target_orientations.items():
                assert (
                    digit in self.digit_tips.keys()
                ), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"
                assert target_orientation.shape == (
                    3,
                ), f"Invalid target orientation shape: {target_orientation.shape}, must be (3,)"
                self.target_orientations[digit] = target_orientation

    def check_contact(self) -> None:
        """Check if any contact is detected."""
        grasp_tilde_matrix_transposed = None
        grasp_matrix_transposed = None
        B_hard = np.hstack([np.eye(3), np.zeros((3, 3))])

        object_center = self.data.body("target").xpos

        for contact in self.data.contact:
            contact_geom0 = self.model_names.geom_id2name[contact.geom[0]]
            contact_geom1 = self.model_names.geom_id2name[contact.geom[1]]

            if contact_geom0 in ["floor"] or contact_geom1 in ["floor"]:
                continue

            position = contact.pos.copy()
            normal = contact.frame[:3].copy() if "cylinder" in contact_geom0 else -contact.frame[:3].copy()

            grasp_tilde_matrix_contact_transposed = get_transposed_grasp_matrix(position, object_center, normal)
            grasp_tilde_matrix_transposed = (
                grasp_tilde_matrix_contact_transposed
                if grasp_tilde_matrix_transposed is None
                else np.hstack([grasp_tilde_matrix_transposed, grasp_tilde_matrix_contact_transposed])
            )
            grasp_matrix_transposed = (
                B_hard @ grasp_tilde_matrix_contact_transposed
                if grasp_matrix_transposed is None
                else np.hstack([grasp_matrix_transposed, B_hard @ grasp_tilde_matrix_contact_transposed])
            )

        if grasp_tilde_matrix_transposed is not None:
            self.grasp_matrix = grasp_matrix_transposed.T

    def control_step(self) -> None:
        """Apply a control step to the Allegro Hand actuators."""
        for digit in self.digit_tips.keys():
            self.control_digit_tip(digit)

    def control_digit_tip(
        self,
        digit: str,
    ):
        """Move the digit tip to the target position."""
        assert digit in self.digit_tips.keys(), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"

        target_position = self.target_positions[digit]
        target_orientation = self.target_orientations[digit]

        digit_position = self.data.body(self.digit_tips[digit]).xpos.copy()
        digit_orientation = quat2mat(self.data.body(self.digit_tips[digit]).xquat.copy()) @ np.array([0.0, 0.0, 1.0])

        # We first control the position and then the orientation in cartesian space
        # If no target orientation is provided, we only control the position

        cartesian_twist = self.controllers[digit].control(
            np.concatenate([target_position, target_orientation])
            if target_orientation is not None
            else target_position,
            np.concatenate([digit_position, digit_orientation]) if target_orientation is not None else digit_position,
        )

        # From the cartesian control signal, we go the control chain back to the actuators
        # First we need to compute the joint control signal using the Jacobian
        jacobian_translational, jacobian_rotational = self.get_digit_jacobian(digit)
        full_jacobian = (
            np.concatenate(
                [jacobian_translational, jacobian_rotational],
                axis=0,
            )
            if target_orientation is not None
            else jacobian_translational
        )
        joint_speeds = np.linalg.pinv(full_jacobian) @ cartesian_twist

        # Finally, we translate the joint control signal to the proper actuator values
        self.data.ctrl[self.actuators.actuators(digit)] = joint_speeds

        log.debug("===============================================")
        log.debug(f"Digit: {digit}")
        log.debug(f"Target Position: {target_position}")
        log.debug(f"Digit Position: {digit_position}")
        log.debug(f"Target Orientation: {target_orientation}")
        log.debug(f"Digit Orientation: {digit_orientation}")
        log.debug("-----------------------------------------------")
        log.debug(f"Cartesian Control Signal: {cartesian_twist}")
        log.debug(f"Joint Control Signal: {joint_speeds}")
