from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import OrderEnforcing


class PPOAgent:
    def __init__(self, env, eval_env):
        
        # Wrap the environment with OrderEnforcing
        env = OrderEnforcing(env)
        self.eval_env = OrderEnforcing(eval_env)

        # Wrap the environment with the Monitor wrapper to have nice outputs of the training
        env = Monitor(env, filename="../logs/trading")
        self.eval_env = Monitor(self.eval_env, filename="./logs/trading")
        
        # Wrap the environment with the DummyVecEnv to have parallelized environment
        env = DummyVecEnv([lambda: env])
        # self.eval_env = DummyVecEnv([lambda: self.eval_env]) wtf it automatically call reset at the end of the eppisode
        
        
        # self.model : PPO  = PPO("MlpPolicy", 
        #                         env, 
        #                         n_steps = 2048*8,
        #                         batch_size = 64,
        #                         n_epochs = 10,    
        #                         verbose=1,
        #                         tensorboard_log="./logs/tensorboard_logs/ppp_trading",
        #                         device="cpu"
        #                         )
        
        self.model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard_logs/dqn_trading")
        
        
    def learn(self, total_timesteps):
        
        class SaveForRenderCallback(BaseCallback):
            def __init__(self, eval_env, verbose=0):
                super(SaveForRenderCallback, self).__init__(verbose)
                self.eval_env = eval_env
            def _on_step(self):
                if self.n_calls % 50000 == 0:
                   
                    done, truncated = False, False
                    obs = self.eval_env.reset()[0]
                    states = None
                    while not (done or truncated):
                        action, states = self.model.predict(obs, state=states,deterministic=True)
                        obs, rewards, done, truncated, info, = self.eval_env.step(action)
                        
                    self.eval_env.env.env.save_for_render(dir = "./logs/render_logs")
                return True
            
            
        self.model.learn(total_timesteps=total_timesteps, callback=SaveForRenderCallback(self.eval_env))
        
    def evaluate(self, env, n_eval_episodes=10):
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=n_eval_episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward
        
    def save(self, path):
        self.model.save(path)
        
        
        
        