import gym
from gym import spaces
import numpy as np

class HangmanEnv(gym.Env):
    def __init__(self, word_list):
        super(HangmanEnv, self).__init__()

        # List of words for the hangman game
        self.word_list = word_list

        # Initialize the state
        self.reset()

        # Define the action and observation space
        self.action_space = spaces.Discrete(26)  # 'a-z'
        self.observation_space = spaces.MultiDiscrete([27] * self.max_word_length)

    def reset(self):
        # Reset the environment to a new word
        self.word = np.random.choice(self.word_list).lower()
        self.correct_guesses = set()
        self.incorrect_guesses = set()

        return self.get_observation()

    def step(self, action):
        # Take an action and return the next state, reward, done, and info

        # Map action to character
        char = chr(ord('a') + action)

        # Update state based on the action
        if char in self.word:
            self.correct_guesses.add(char)
        else:
            self.incorrect_guesses.add(char)

        # Define reward based on correct and incorrect guesses
        reward = self.calculate_reward()

        # Define termination condition
        done = self.is_done()

        # Construct the observation
        observation = self.get_observation()

        return observation, reward, done, {}

    def render(self):
        # Optionally, implement a method to visualize the current state
        current_state = ''.join([char if char in self.correct_guesses else '_' for char in self.word])
        print("Current State: ", current_state)
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
        return set(self.word) == self.correct_guesses or len(self.incorrect_guesses) >= 6

    def get_observation(self):
        # Encode the current state into an observation
        observation = [ord('_')] * self.max_word_length
        for i, char in enumerate(self.word):
            if char in self.correct_guesses:
                observation[i] = ord(char)
        return observation

    @property
    def max_word_length(self):
        # Maximum length of words in the word list
        return max(len(word) for word in self.word_list)

# Example usage:
word_list = ['hangman', 'python', 'gym', 'openai']
env = HangmanEnv(word_list)
obs = env.reset()
env.render()

for _ in range(10):
    action = env.action_space.sample()  # Replace with your own action selection logic
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()

