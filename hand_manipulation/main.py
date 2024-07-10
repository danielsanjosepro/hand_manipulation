"""Start a Mujoco simulation with the Shadow Hand."""

import time
import logging
from pathlib import Path

import mujoco
import mujoco.viewer

from shadow_hand import ShadowHand


log = logging.getLogger("rich")
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)

project_root = Path(__file__).parents[1]
shadow_hand = ShadowHand.from_xml_path(str(project_root / "assets" / "scene_left.xml"))

model, data = shadow_hand.model, shadow_hand.data

simulation_duration = 1000.0  # seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < simulation_duration:
        step_start = time.time()

        target_pos = data.body("target").xpos.copy()  # Get the position of the target body.

        target_thumb_position = data.body(shadow_hand.digit_tips["thumb"]).xpos.copy()
        target_index_position = data.body(shadow_hand.digit_tips["index"]).xpos.copy()
        target_middle_position = data.body(shadow_hand.digit_tips["middle"]).xpos.copy()
        target_ring_position = data.body(shadow_hand.digit_tips["ring"]).xpos.copy()
        target_little_position = data.body(shadow_hand.digit_tips["little"]).xpos.copy()

        # Set the target positions of the digits that should be controlled.
        target_index_position = target_pos

        target_positions = {
            "thumb": target_thumb_position,
            "index": target_index_position,
            "middle": target_middle_position,
            "ring": target_ring_position,
            "little": target_little_position,
        }

        # TODO: orientation control might not be necessary, but we can add it later
        # ez = np.array([0.0, 0.0, 1.0])
        # target_orientations = {
        # "thumb": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "index": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "middle": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "ring": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "little": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # }
        target_orientations = None
        
        # Our control cycle: we update the targets and control the hand with the new targets.
        shadow_hand.update_targets(target_positions, target_orientations)
        shadow_hand.control_step()

        mujoco.mj_step(model, data)

        # TODO: Maybe get rid of this?
        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
