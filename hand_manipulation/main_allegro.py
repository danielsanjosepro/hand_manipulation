"""Start a Mujoco simulation with a Hand."""

import time
import logging
from pathlib import Path

import mujoco
import mujoco.viewer

from hands.allegro_hand import AllegroHand


log = logging.getLogger("rich")
log.setLevel(logging.DEBUG)
# log.setLevel(logging.INFO)

project_root = Path(__file__).parents[1]
xml_path = str(project_root / "assets" / "allegro" / "scene_left.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

allegro_hand = AllegroHand(
    model=model,
    data=data,
    kp=1.0,
    # ki=0.0,
    # kd=0.0,
    # integral_clip=0.1
)

simulation_duration = 1000.0  # seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    # Activate the contact points in the viewer.
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    while viewer.is_running() and time.time() - start < simulation_duration:
        step_start = time.time()

        # Set the fingers on the cylinder.
        target_thumb_position = data.body("thumb_target").xpos
        target_index_position = data.body("index_target").xpos
        target_middle_position = data.body("middle_target").xpos
        target_ring_position = data.body("ring_target").xpos

        target_positions = {
            "thumb": target_thumb_position,
            "index": target_index_position,
            "middle": target_middle_position,
            "ring": target_ring_position,
        }

        # Our control cycle: we update the targets and control the hand with the new targets.
        allegro_hand.update_targets(target_positions)
        allegro_hand.control_step()

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
