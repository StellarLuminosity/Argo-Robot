from gymnasium.envs.registration import register
from gym_custom_envs.o2_env import AntEnv

register(
    id='o2-v0',
    entry_point='gym_custom_envs.o2_env:AntEnv',
    disable_env_checker = True,
    order_enforce=False
)
