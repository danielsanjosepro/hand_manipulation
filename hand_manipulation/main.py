import time
import os
import logging

import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.quaternions import quat2mat

from shadow_hand import ShadowHand


log = logging.getLogger("rich")
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)

home = os.path.expanduser("~")
shadow_hand = ShadowHand.from_xml_path(os.path.join(home, "hand_manipulation","assets", "scene_left.xml"))
model, data = shadow_hand.model, shadow_hand.data

simulation_duration = 1000.0  # seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < simulation_duration:
    step_start = time.time()

    target_pos = data.body("target").xpos.copy()    # Get the position of the target body.

    target_thumb_position = data.body(shadow_hand.digit_tips["thumb"]).xpos.copy() # Get the position of the thumb tip.
    target_index_position = data.body(shadow_hand.digit_tips["index"]).xpos.copy() # Get the position of the index tip.
    target_middle_position = data.body(shadow_hand.digit_tips["middle"]).xpos.copy()
    target_ring_position = data.body(shadow_hand.digit_tips["ring"]).xpos.copy() # Get the position of the ring tip.
    target_little_position = data.body(shadow_hand.digit_tips["little"]).xpos.copy() # Get the position of the little tip.

    target_index_position = target_pos

    target_positions = {
        "thumb": target_thumb_position,
        "index": target_index_position,
        "middle": target_middle_position,
        "ring": target_ring_position,
        "little": target_little_position,
    }

    # ez = np.array([0.0, 0.0, 1.0])
    # target_orientations = {
        # "thumb": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "index": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "middle": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "ring": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
        # "little": quat2mat(np.array([0.0, 0.0, 0.0, 1.0])) @ ez,
    # }
    target_orientations = None
    shadow_hand.move_digit_tips(target_positions, target_orientations)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(model, data)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

