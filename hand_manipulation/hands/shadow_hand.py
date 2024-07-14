"""Class containing functions to control the Shadow Hand in the MuJoCo environment."""

import logging
from typing import List, Tuple

import mujoco
from rich.logging import RichHandler
import numpy as np
from transforms3d.quaternions import quat2mat

from utils.control import PIDController

# Set up the logging
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)


class ShadowLeftHandActuators:
    """Enum class for the different actuator types in the Shadow Hand."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        include_wrists: bool = True,
    ):
        """Initialize the Shadow Left Hand Actuators."""
        self.model = model
        self.data = data
        self.n_actuators = model.nu
        self.n_joints = model.njnt
        self.include_wrists = include_wrists

        self.actuator_to_joint = {
            "lh_A_WRJ2": "lh_WRJ2",
            "lh_A_WRJ1": "lh_WRJ1",
            "lh_A_THJ5": "lh_THJ5",
            "lh_A_THJ4": "lh_THJ4",
            "lh_A_THJ3": "lh_THJ3",
            "lh_A_THJ2": "lh_THJ2",
            "lh_A_THJ1": "lh_THJ1",
            "lh_A_FFJ4": "lh_FFJ4",
            "lh_A_FFJ3": "lh_FFJ3",
            "lh_A_FFJ0": ["lh_FFJ1", "lh_FFJ2"],
            "lh_A_MFJ4": "lh_MFJ4",
            "lh_A_MFJ3": "lh_MFJ3",
            "lh_A_MFJ0": ["lh_MFJ1", "lh_MFJ2"],
            "lh_A_RFJ4": "lh_RFJ4",
            "lh_A_RFJ3": "lh_RFJ3",
            "lh_A_RFJ0": ["lh_RFJ1", "lh_RFJ2"],
            "lh_A_LFJ5": "lh_LFJ5",
            "lh_A_LFJ4": "lh_LFJ4",
            "lh_A_LFJ3": "lh_LFJ3",
            "lh_A_LFJ0": ["lh_LFJ1", "lh_LFJ2"],
        }

        self.joint_to_actuator = {
            "lh_WRJ2": "lh_A_WRJ2",
            "lh_WRJ1": "lh_A_WRJ1",
            "lh_THJ5": "lh_A_THJ5",
            "lh_THJ4": "lh_A_THJ4",
            "lh_THJ3": "lh_A_THJ3",
            "lh_THJ2": "lh_A_THJ2",
            "lh_THJ1": "lh_A_THJ1",
            "lh_FFJ4": "lh_A_FFJ4",
            "lh_FFJ3": "lh_A_FFJ3",
            "lh_FFJ2": "lh_A_FFJ0",
            "lh_FFJ1": "lh_A_FFJ0",
            "lh_MFJ4": "lh_A_MFJ4",
            "lh_MFJ3": "lh_A_MFJ3",
            "lh_MFJ2": "lh_A_MFJ0",
            "lh_MFJ1": "lh_A_MFJ0",
            "lh_RFJ4": "lh_A_RFJ4",
            "lh_RFJ3": "lh_A_RFJ3",
            "lh_RFJ2": "lh_A_RFJ0",
            "lh_RFJ1": "lh_A_RFJ0",
            "lh_LFJ5": "lh_A_LFJ5",
            "lh_LFJ4": "lh_A_LFJ4",
            "lh_LFJ3": "lh_A_LFJ3",
            "lh_LFJ2": "lh_A_LFJ0",
            "lh_LFJ1": "lh_A_LFJ0",
        }

        self.actuator_to_joint_mat = None
        self.joint_to_actuator_mat = None
        self.initialize_actuator_to_joint_mat()
        self.initialize_joint_to_actuator_mat()

        log.debug(f"Number of actuators: {self.n_actuators}")
        log.debug(f"Number of joints: {self.n_joints}")
        log.debug(f"{self.actuator_to_joint_mat.shape}")
        log.debug(f"{self.actuator_to_joint_mat}")
        log.debug(f"{self.joint_to_actuator_mat.shape}")
        log.debug(f"{self.joint_to_actuator_mat}")

    def initialize_actuator_to_joint_mat(self):
        """Initialize the actuator to joint matrix which maps actuator to joint space such that joint = A2J @ actuator.

        Remark: The matrix is of shape (n_joints, n_actuators).
        """
        self.actuator_to_joint_mat = np.zeros((self.n_joints, self.n_actuators))

        for actuator, joint in self.actuator_to_joint.items():
            if isinstance(joint, list):
                for j in joint:
                    self.actuator_to_joint_mat[self.model.joint(j).id, self.model.actuator(actuator).id] = 1
            else:
                self.actuator_to_joint_mat[self.model.joint(joint).id, self.model.actuator(actuator).id] = 1

    def initialize_joint_to_actuator_mat(self):
        """Initialize the joint to actuator matrix which maps joint to actuator space such that actuator = J2A @ joint.

        Remark: The matrix is of shape (n_actuators, n_joints).
        """
        self.joint_to_actuator_mat = np.linalg.pinv(self.actuator_to_joint_mat)

    def joint_names(self, digit: str) -> List[str]:
        """Return the joint names of the digit."""
        # TODO: actually the wrist joints should be included in the finger motion as well
        wrist_joints = ["lh_WRJ2", "lh_WRJ1"] if self.include_wrists else []
        if digit == "thumb":
            return ["lh_THJ5", "lh_THJ4", "lh_THJ3", "lh_THJ2", "lh_THJ1"] + wrist_joints
        elif digit == "index":
            return ["lh_FFJ4", "lh_FFJ3", "lh_FFJ2", "lh_FFJ1"] + wrist_joints
        elif digit == "middle":
            return ["lh_MFJ4", "lh_MFJ3", "lh_MFJ2", "lh_MFJ1"] + wrist_joints
        elif digit == "ring":
            return ["lh_RFJ4", "lh_RFJ3", "lh_RFJ2", "lh_RFJ1"] + wrist_joints
        elif digit == "little":
            return ["lh_LFJ5", "lh_LFJ4", "lh_LFJ3", "lh_LFJ2", "lh_LFJ1"] + wrist_joints
        else:
            raise ValueError(f"Invalid digit: {digit}")

    def joints(self, digit: str) -> List[int]:
        """Return the joint ids of the digit."""
        return np.sort([self.joint_id(joint) for joint in self.joint_names(digit)])

    def actuator_names(self, digit: str) -> List[str]:
        """Return the joint names of the digit."""
        # TODO: actually the wrist joints should be included in the finger motion as well
        wrist_actuators = ["lh_A_WRJ2", "lh_A_WRJ1"] if self.include_wrists else []
        if digit == "thumb":
            return ["lh_A_THJ5", "lh_A_THJ4", "lh_A_THJ3", "lh_A_THJ2", "lh_A_THJ1"] + wrist_actuators
        elif digit == "index":
            return ["lh_A_FFJ4", "lh_A_FFJ3", "lh_A_FFJ0"] + wrist_actuators
        elif digit == "middle":
            return ["lh_A_MFJ4", "lh_A_MFJ3", "lh_A_MFJ0"] + wrist_actuators
        elif digit == "ring":
            return ["lh_A_RFJ4", "lh_A_RFJ3", "lh_A_RFJ0"] + wrist_actuators
        elif digit == "little":
            return ["lh_A_LFJ5", "lh_A_LFJ4", "lh_A_LFJ3", "lh_A_LFJ0"] + wrist_actuators
        else:
            raise ValueError(f"Invalid digit: {digit}")

    def actuators(self, digit: str) -> List[int]:
        """Get sorted actuator ids for a given digit."""
        return np.sort([self.model.actuator(actuator).id for actuator in self.actuator_names(digit)])


class ShadowRightHandActuators:
    """Enum class for the different actuator types in the Shadow Hand."""

    def __init__(self):
        """Initialize the Shadow Right Hand Actuators."""
        raise NotImplementedError


class ShadowHand:
    """Class to control the Shadow Hand in the MuJoCo environment."""

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
        """Initialize the Shadow Hand class."""
        self.model = model
        self.data = data
        mujoco.mj_kinematics(model, data)

        # PID controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.controllers = {
            "thumb": PIDController(self.kp, self.ki, self.kd),
            "index": PIDController(self.kp, self.ki, self.kd),
            "middle": PIDController(self.kp, self.ki, self.kd),
            "ring": PIDController(self.kp, self.ki, self.kd),
            "little": PIDController(self.kp, self.ki, self.kd),
        }

        # TODO: make it posible to change this to the contact location
        self.digit_tips = (
            {
                "thumb": "lh_th_tip",
                "index": "lh_ff_tip",
                "middle": "lh_mf_tip",
                "ring": "lh_rf_tip",
                "little": "lh_lf_tip",
            }
            if hand_type == "left"
            else {
                "thumb": "rh_th_tip",
                "index": "rh_ff_tip",
                "middle": "rh_mf_tip",
                "ring": "rh_rf_tip",
                "little": "rh_lf_tip",
            }
        )

        self.jacobians_translational = {
            "thumb": np.zeros((3, self.model.nv)),
            "index": np.zeros((3, self.model.nv)),
            "middle": np.zeros((3, self.model.nv)),
            "ring": np.zeros((3, self.model.nv)),
            "little": np.zeros((3, self.model.nv)),
        }

        self.jacobians_rotational = {
            "thumb": np.zeros((3, self.model.nv)),
            "index": np.zeros((3, self.model.nv)),
            "middle": np.zeros((3, self.model.nv)),
            "ring": np.zeros((3, self.model.nv)),
            "little": np.zeros((3, self.model.nv)),
        }

        self.target_positions = {
            "thumb": np.zeros(3),
            "index": np.zeros(3),
            "middle": np.zeros(3),
            "ring": np.zeros(3),
            "little": np.zeros(3),
        }
        self.target_orientations = {
            "thumb": None,
            "index": None,
            "middle": None,
            "ring": None,
            "little": None,
        }

        self.actuators = (
            ShadowLeftHandActuators(model, data) if hand_type == "left" else ShadowRightHandActuators(model, data)
        )

    @classmethod
    def from_xml_path(cls, xml_path: str) -> "ShadowHand":
        """Load the shadow hand from an XML path."""
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return cls(model, data)

    def update_jacobians(self) -> None:
        """Update the Jacobians of the digit tips."""
        for digit in self.digit_tips.keys():
            self.update_digit_jacobian(digit)

    def update_digit_jacobian(self, digit: str) -> np.ndarray:
        """Get the Jacobian of the digit tip."""
        assert digit in self.digit_tips.keys(), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"

        mujoco.mj_jac(
            self.model,
            self.data,
            self.jacobians_translational[digit],
            self.jacobians_rotational[digit],
            self.data.body(self.digit_tips[digit]).xpos,
            self.data.body(self.digit_tips[digit]).id,
        )

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

    def control_step(self) -> None:
        """Apply a control step to the Shadow Hand actuators."""
        self.update_jacobians()

        digit = "index"
        desired_actuation = self.get_digit_tip_control(digit)
        actuator_ids = self.actuators.actuators(digit)
        self.data.ctrl[actuator_ids] = desired_actuation[actuator_ids].copy()

        # TODO: wrist should be dealt differently... no time to do it now
        # for digit in self.digit_tips.keys():
            # desired_actuation = self.get_digit_tip_control(digit)
            # actuator_ids = self.actuators.actuators(digit)
            # self.data.ctrl[actuator_ids] = desired_actuation[actuator_ids].copy()

    def get_digit_tip_control(
        self,
        digit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Move the digit tip to the target position."""
        assert digit in self.digit_tips.keys(), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"

        target_position = self.target_positions[digit]
        target_orientation = self.target_orientations[digit]

        digit_position = self.data.body(self.digit_tips[digit]).xpos.copy()
        digit_orientation = quat2mat(self.data.body(self.digit_tips[digit]).xquat.copy()) @ np.array([0.0, 0.0, 1.0])

        # We first control the position and then the orientation in cartesian space
        # If no target orientation is provided, we only control the position
        # TODO: (optional) handle the case where the target orientation is not provided from the beginning
        cartesian_twist = self.controllers[digit].control(
            np.concatenate([target_position, target_orientation])
            if target_orientation is not None
            else target_position,
            np.concatenate([digit_position, digit_orientation]) if target_orientation is not None else digit_position,
        )

        # From the cartesian control signal, we go the control chain back to the actuators
        # First we need to compute the joint control signal using the Jacobian
        full_jacobian = (
            np.concatenate(
                [self.jacobians_translational[digit], self.jacobians_rotational[digit]],
                axis=0,
            )
            if target_orientation is not None
            else self.jacobians_translational[digit]
        )
        joint_speeds = np.linalg.pinv(full_jacobian) @ cartesian_twist

        # Finally, we translate the joint control signal to the actuator control signal
        actuator_speeds = self.actuators.joint_to_actuator_mat @ joint_speeds

        log.debug("===============================================")
        log.debug(f"Digit: {digit}")
        log.debug(f"Target Position: {target_position}")
        log.debug(f"Digit Position: {digit_position}")
        log.debug(f"Target Orientation: {target_orientation}")
        log.debug(f"Digit Orientation: {digit_orientation}")
        log.debug("-----------------------------------------------")
        log.debug(f"Cartesian Control Signal: {cartesian_twist}")
        log.debug(f"Joint Control Signal: {joint_speeds}")
        log.debug(f"Actuator Control Signal: {actuator_speeds}")

        return actuator_speeds
