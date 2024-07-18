"""Start a Mujoco simulation with a Hand."""

import time
import logging
from pathlib import Path
import numpy as np

import mujoco
import mujoco.viewer

from hands.allegro_hand import AllegroHand


log = logging.getLogger("rich")
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)

project_root = Path(__file__).parents[1]
xml_path = str(project_root / "assets" / "allegro" / "scene_left.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

allegro_hand = AllegroHand(
    model=model,
    data=data,
    kp=0.05,
    # ki=0.0,
    ki=0.001,
    kd=0.01,
    integral_clip=0.1,
)


simulation_duration = 1000.0  # seconds
simulation_speed_up = 1  # Speed up the simulation by this factor.


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    # Activate the contact points in the viewer.
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    while viewer.is_running() and time.time() - start < simulation_duration:
        step_start = time.time()

        allegro_hand.update()  # Update the hand state.
        allegro_hand.control_step()  # Control the hand.
    
        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep / simulation_speed_up - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
