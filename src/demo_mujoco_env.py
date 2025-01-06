import numpy as np
import mujoco
import time
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")

# Initialize simulation.
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)



duration = 3.8  # (seconds)
framerate = 60  # (Hz)
timestep = 1 / framerate  # (seconds)

# Simulate and display video.
mujoco.mj_resetData(model, data)  # Reset state and time.

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()
    while viewer.is_running() and time.time() - start < duration:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        # Apply random actions
        random_actions = np.random.uniform(-400, 400, size=model.nu)  # Random control inputs
        data.ctrl[:] = random_actions

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


