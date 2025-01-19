import argparse
import os
import pickle
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs
from pynput import keyboard

# Global variables to store command velocities
lin_x = 0.0
lin_y = 0.0
ang_z = 0.0
base_height = 0.3
jump_height = 0.7
toggle_jump = False

def on_press(key):
    global lin_x, lin_y, ang_z, base_height, toggle_jump, jump_height
    try:
        if key.char == 'w':
            lin_x += 0.1
        elif key.char == 's':
            lin_x -= 0.1
        elif key.char == 'a':
            lin_y += 0.1
        elif key.char == 'd':
            lin_y -= 0.1
        elif key.char == 'q':
            ang_z += 0.1
        elif key.char == 'e':
            ang_z -= 0.1
        elif key.char == 'r':
            base_height += 0.1
        elif key.char == 'f':
            base_height -= 0.1
        elif key.char == 'j':
            toggle_jump = True
        elif key.char == 'u':
            jump_height += 0.1
        elif key.char == 'm':
            jump_height -= 0.1
            
            
        # Clear the console
        os.system('clear')
        
        print(f"lin_x: {lin_x:.2f}, lin_y: {lin_y:.2f}, ang_z: {ang_z:.2f}, base_height: {base_height:.2f}, jump: {toggle_jump*jump_height:.2f}")
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def main():
    global lin_x, lin_y, ang_z, base_height, toggle_jump, jump_height
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=700)
    args = parser.parse_args()

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",

    )

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"genesis/logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    
    env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height]]).to("cuda:0")
    iter = 0

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    reset_jump_toggle_iter = 0
    with torch.no_grad():
        while True:
            actions = policy(obs)
            # print(f"toggle_jump: {toggle_jump}, jump_height: {jump_height}")
            env.commands = torch.tensor([[lin_x, lin_y, ang_z, base_height, toggle_jump*jump_height]]).to("cuda:0")
            obs, _, rews, dones, infos = env.step(actions, is_train=False)
            # print(env.base_pos, env.base_lin_vel)
            if toggle_jump and reset_jump_toggle_iter == 0:
                reset_jump_toggle_iter = iter + 3
            if iter == reset_jump_toggle_iter and toggle_jump:
                toggle_jump = False
                reset_jump_toggle_iter = 0
                    
            iter += 1
            
            if dones.any():
                iter = 0

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""