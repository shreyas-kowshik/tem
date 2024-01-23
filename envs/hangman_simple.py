import stable_baselines3
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces

class HangmanEnv(gym.Env):
    def __init__(self, word_list, debug=False, max_word_length=None):
        super(HangmanEnv, self).__init__()

        # List of words for the hangman game
        self.word_list = word_list

        self.debug=debug
        if max_word_length is None:
          self.max_word_length = self.get_max_word_length()
        else:
          self.max_word_length = max_word_length



        # Initialize the state
        self.reset()

        # Define the action and observation space
        self.action_space = spaces.Discrete(26)  # 'a-z'
        self.observation_space = spaces.MultiDiscrete([28] * self.max_word_length) # 'a-z' + '_' + pad

    def reset(self, seed=None):
        # Reset the environment to a new word
        self.word = np.random.choice(self.word_list).lower()
        self.correct_guesses = set()
        self.incorrect_guesses = set()

        self.cur_obs = self.get_observation()

        if self.debug:
          print('Env is reset')
          print('Word : {}'.format(self.word))

        return (self.cur_obs, {})

    def step(self, action):
        # Take an action and return the next state, reward, done, and info
        # Action will be an integer in 0-26

        # Map action to character
        char = chr(ord('a') + action)

        if self.debug:
          print('Action taken : {}'.format(action))

        # Update state based on the action
        if char in self.word:
            self.correct_guesses.add(char)
        else:
            self.incorrect_guesses.add(char)

        # Define termination condition
        done = self.is_done()

        reward = 0
        if done:
          if self.cur_obs2str() == self.word:
            reward = 1

        # Construct the observation
        self.cur_obs = self.get_observation()

        return self.cur_obs, reward, done, (len(self.incorrect_guesses) >= 6), {}

    def render(self):
        # Optionally, implement a method to visualize the current state
        print("Current State: ", self.cur_obs2str())
        print("Incorrect Guesses: ", sorted(list(self.incorrect_guesses)))

    def calculate_reward(self):
        # Define your own reward logic based on the current state
        correct_letters = set(self.word)
        correct_guesses = self.correct_guesses.intersection(correct_letters)

        # Reward is proportional to the number of correct guesses
        return len(correct_guesses) / len(correct_letters)

    def is_done(self):
        # Define your own termination condition based on the current state
        # For simplicity, the game ends when all letters are guessed or a fixed number of incorrect guesses
        return self.cur_obs2str() == self.word or len(self.incorrect_guesses) >= 6

    def get_observation(self):
        # Encode the current state into an observation
        # observation = [ord('_')] * self.max_word_length
        # for i, char in enumerate(self.word):
        #     if char in self.correct_guesses:
        #         observation[i] = ord(char)
        # return observation
        observation = [27] * self.max_word_length # All pads
        for i, char in enumerate(self.word):
          if char in self.correct_guesses:
            observation[i]=ord(char)-97
          else:
            observation[i]=26
        return np.array(observation)

    def cur_obs2str(self):
      s=''
      for i in self.cur_obs:
        if i>=0 and i<=25:
          s+=chr(ord('a')+i)
        elif i==26:
          s+='_'
      return s


    def get_max_word_length(self):
      # Maximum length of words in the word list
      return max(len(word) for word in self.word_list)