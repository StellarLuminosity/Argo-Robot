import gymnasium as gym
from gym_custom_envs.o2_env import AntEnv

# env = gym.make('o2-v0', render_mode='human')
env = AntEnv(render_mode='human')
env.reset()

# Test the environment
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        obs = env.reset()
env.close()
