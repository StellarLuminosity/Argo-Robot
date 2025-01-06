from gymnasium.envs.registration import register
from gym_environments.environments import TradingEnv, MultiDatasetTradingEnv

register(
    id='TradingEnv',
    entry_point='gym_environments.environments:TradingEnv',
    disable_env_checker = True,
    order_enforce=False
)
register(
    id='MultiDatasetTradingEnv',
    entry_point='gym_environments.environments:MultiDatasetTradingEnv',
    disable_env_checker = True,
    order_enforce=False
)