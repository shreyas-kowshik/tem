import stable_baselines3
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from envs.hangman_simple import *
from utils import *
from cfg import config_dict

# Multiproc env
def make_env(rank: int = 0, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    # def _init() -> gym.Env:
    #     # env = gym.make(env_id)
    #     env = HangmanEnv(word_list, debug=False)
    #     env.reset(seed=seed + rank)
    #     return env

    # set_random_seed(seed)
    # return _init
    return HangmanEnv(word_list, debug=False)

words = read_lines('words_250000_train.txt')
max_word_length = max(len(word) for word in words)

idx=1
while True:
  print('idx : {}'.format(idx))
  print('---')
  idx_max = idx * config_dict["NUM_INC_PER_STEP"]
  if idx_max >= len(words):
    break

  word_list = words[:idx_max]
  # env = HangmanEnv(word_list, debug=False, max_word_length=max_word_length)
  num_cpu = 4  # Number of processes to use
  # Create the vectorized environment
  # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
  env = make_vec_env(make_env, n_envs=num_cpu)

  # check_env(env)

  # model = PPO(MlpPolicy, env, verbose=1, batch_size=64, learning_rate=0.0003, ent_coef=0.0)
  # Use a separate environement for evaluation
  # eval_env = HangmanEnv(words_list_test, debug=False)

  # Random Agent, before training
  # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)

  # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
  # # Train the agent for 10000 steps
  model = PPO(MlpPolicy, env, verbose=1, batch_size=64, learning_rate=0.0003, ent_coef=0.0)
  if idx>1:
    model.load(config_dict["model_path"])
  model.learn(total_timesteps=config_dict["LEARN_STEPS"], log_interval=5)
  model.save(config_dict["model_path"])
  idx+=1
