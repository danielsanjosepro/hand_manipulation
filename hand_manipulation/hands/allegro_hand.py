"""Class containing functions to control the Allegro hand in the MuJoCo environment."""

import logging
from typing import List, Tuple

import mujoco
from rich.logging import RichHandler
import numpy as np
from transforms3d.quaternions import quat2mat

from utils.grasping import get_rotation_matrix_from_normal, get_transposed_grasp_matrix
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
        only_position_control: bool = False,
    ) -> None:
        """Initialize the Allegro Hand class."""
        self.model = model
        self.data = data
        self.model_names = MujocoModelNames(model)
        log.info(f"Joint names: {self.model_names.joint_names}")
        log.info(f"Joint names length: {len(self.model_names.joint_names)}")
        log.info(f"Body names: {self.model_names.body_names}")
        log.info(f"Body names length: {len(self.model_names.body_names)}")
        log.info(f"Geom names: {self.model_names.geom_names}")
        log.info(f"Geom names length: {len(self.model_names.geom_names)}")
        log.info(f"Actuator names: {self.model_names.actuator_names}")
        log.info(f"Actuator names length: {len(self.model_names.actuator_names)}")


        mujoco.mj_kinematics(model, data)
        self.actuators = AllegroLeftHandActuators(model, data) if hand_type == "left" else None

        # State machine to change between position and torque control
        self.state = "position_control"
        self.time_of_contact = 0
        self.only_position_control = only_position_control

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

        # self.desired_object_twist = np.zeros(6)
        # self.desired_object_twist[3] = 0.0
        # self.target_twists = []

        self.target_contacts = []


    @classmethod
    def from_xml_path(cls, xml_path: str) -> "AllegroHand":
        """Load the shadow hand from an XML path."""
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return cls(model, data)

    def get_full_jacobian(self, pos: np.ndarray, id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full Jacobian of the hand at a given position and body id."""
        jacobian_translational = np.zeros((3, self.model.nv))
        jacobian_rotational = np.zeros((3, self.model.nv))

        mujoco.mj_jac(self.model, self.data, jacobian_translational, jacobian_rotational, pos, id)

        # We remove the 6 last elements which correspond to the object freejoint dof
        return jacobian_translational[:, :17], jacobian_rotational[:, :17]

    def get_digit_jacobian(
        self,
        digit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the Jacobian of the digit tip."""
        assert digit in self.digit_tips.keys(), f"Invalid digit: {digit}, must be one of {self.digit_tips.keys()}"
        pos = self.data.body(self.digit_tips[digit]).xpos.copy()
        id = self.data.body(self.digit_tips[digit]).id

        jacobian_translational, jacobian_rotational = self.get_full_jacobian(pos, id)

        # We only return the relevant columns of the Jacobian corresponding to the digit joints
        return jacobian_translational[:, self.actuators.joints(digit)], jacobian_rotational[
            :, self.actuators.joints(digit)
        ]

    def update(self) -> None:
        """Change the state of the hand based on the current contact points and update the target positions/twist."""
        if self.state == "position_control" and self.data.ncon > 0:
            self.state = "position_control_in_contact"
            self.time_of_contact = self.data.time

        if self.state == "position_control_in_contact" and self.data.ncon == 0:
            self.state = "position_control"

        if self.state == "position_control_in_contact" and self.data.time - self.time_of_contact > 5.0:
            self.state = "object_control"

        # Update the target positions and orientations
        if self.state == "position_control" or self.state == "position_control_in_contact":
            self.target_positions["thumb"] = self.data.body("thumb_target").xpos
            self.target_positions["index"] = self.data.body("index_target").xpos
            self.target_positions["middle"] = self.data.body("middle_target").xpos
            self.target_positions["ring"] = self.data.body("ring_target").xpos
        elif self.state == "object_control":
            for i, contact in enumerate(self.data.contact):
                contact_geom0 = self.model_names.geom_id2name[contact.geom[0]]
                contact_geom1 = self.model_names.geom_id2name[contact.geom[1]]


                if contact_geom0 in ["floor"] or contact_geom1 in ["floor"]:
                    continue

                sign = - 1 if "cylinder" in contact_geom0 else 1

                contact_geom_finger = contact_geom1 if "cylinder" in contact_geom0 else contact_geom0

                position = contact.pos.copy()
                normal = sign * contact.frame[:3].copy()

                contact_wrench_in_contact_frame = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_wrench_in_contact_frame)
                contact_wrench = get_rotation_matrix_from_normal(normal, stacked=True) @ (sign * contact_wrench_in_contact_frame)

                body_name = contact_geom_finger[:-len("_collision")]
                body_id = self.data.body(body_name).id

                # Check if target is already in the list
                for i, target_contact in enumerate(self.target_contacts):
                    if target_contact["body_id"] == body_id:
                        target_contact["contact_wrench"] = contact_wrench
                        target_contact["normal"] = normal
                        target_contact["position"] = position
                        target_contact["mu"] = contact.mu
                        break
                else:
                    log.info(f"Adding new contact: {body_name}")
                    self.target_contacts.append(
                        {
                            "body_id": body_id,
                            "position": position,
                            "normal": normal,
                            "contact_wrench": contact_wrench,
                            "mu": contact.mu,
                            "controller": PIDController(0.001, 0.0, 0.000),  # PID controller for the force control
                            "previous_signal": 0.01, # Previous signal of the PID controller
                        }
                    )

                # grasp_tilde_matrix_transposed = get_transposed_grasp_matrix(position, object_center, normal)
                # # R = get_rotation_matrix_from_normal(normal)
                # # grap_matrix_transposed = B_hard @ grasp_tilde_matrix_transposed

                # R = get_rotation_matrix_from_normal(normal, stacked=True)
                # target_twist_contact = R @ grasp_tilde_matrix_transposed @ self.desired_object_twist

                # # Remove the "collision" string from the contact_geom_finger at the end of the name
                # body_name = contact_geom_finger[:-len("_collision")]
                # body_id = self.data.body(body_name).id

                # self.target_twists.append(
                    # {
                        # "body_id": body_id,
                        # "position": position,
                        # "target_twist": target_twist_contact,
                    # }
                # )

    def compute_grasp_matrix(self) -> None:
        """Used to comute the grasp matrix from the contact points."""
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

        log.info(f"Grasp Matrix: {self.grasp_matrix}")

    def control_step(self) -> None:
        """Apply a control step to the Allegro Hand actuators."""
        if self.state == "position_control" or self.state == "position_control_in_contact":
            for digit in self.digit_tips.keys():
                self.control_digit_tip(digit)
        elif self.state == "object_control":
            self.control_object()


    def control_object(self) -> None:
        """Control the object so that it does not slip."""
        mass = 1.0
        g = 9.81
        safety_factor = 3.0
        load = mass * g * safety_factor
        center_object = self.data.body("target").xpos

        joint_speeds_total = np.zeros(16)
        for target_contact in self.target_contacts:
            body_id = target_contact["body_id"]
            position = target_contact["position"]
            normal = target_contact["normal"]
            contact_wrench = target_contact["contact_wrench"]
            mu = target_contact["mu"]
            pid_controller = target_contact["controller"]
            previous_signal = target_contact["previous_signal"]

            center_object_contact = center_object 
            center_object_contact[2] = position[2]
            direction = center_object_contact - position
            direction /= np.linalg.norm(direction)

            # We press in the direction of the normal until the wrench in the z direction cancels the gravity
            jacobian_translational, jacobian_rotational = self.get_full_jacobian(position, body_id)


            target_force = load * mu / len(self.target_contacts)
            current_force = np.linalg.norm(contact_wrench[:3])
            control_signal = pid_controller.control(target_position=target_force, current_position=current_force)
            current_signal = previous_signal + control_signal
            
            joint_speeds = np.linalg.pinv(jacobian_translational[:,1:]) @ (current_signal * direction)

            joint_speeds_total += joint_speeds

        self.data.ctrl[1:] = joint_speeds_total

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
