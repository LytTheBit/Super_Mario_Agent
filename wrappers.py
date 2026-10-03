# wrappers.py
import numpy as np
import torch
from torchvision import transforms as T
import gym
from gym.spaces import Box


class SkipFrame(gym.Wrapper):
    """Repeat an action for `skip` frames and sum the rewards.
    Consecutive frames barely differ, so this saves computation
    without losing much information."""

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB (240, 256, 3) to grayscale.
    Color is not needed to decide Mario's actions, and dropping it
    cuts the input size by two-thirds."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # [H, W, C] numpy array -> [C, H, W] tensor, as torchvision expects
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """Downsample the frame to a fixed square size (default 84x84),
    matching MarioNet's expected input."""

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class RewardShaping(gym.Wrapper):
    """Adds an optional reward term on top of the environment's default reward.

    mode:
        'none'          -> default reward untouched.
        'flag-bonus'    -> + flag_bonus when info['flag_get'] is True.
        'coin-shaping'  -> + coin_weight * (coin delta) each step, plus the flag bonus.
        'game-score'    -> + score_weight * (in-game score delta) each step, plus the flag bonus.
                           info['score'] already aggregates coins, power-ups and stomped
                           enemies into a single NES counter, so this is a coarser signal
                           than 'coin-shaping' (no way to isolate individual event types),
                           but it is also cheap to compute and captures "something good
                           happened" in one proxy term.
    """

    def __init__(self, env, mode="none", flag_bonus=500, coin_weight=10, score_weight=0.1):
        super().__init__(env)
        self.mode = mode
        self.flag_bonus = flag_bonus
        self.coin_weight = coin_weight
        self.score_weight = score_weight
        self.prev_coins = 0
        self.prev_score = 0

    def reset(self, **kwargs):
        self.prev_coins = 0
        self.prev_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        if self.mode == "flag-bonus":
            if info.get("flag_get", False):
                reward += self.flag_bonus

        elif self.mode == "coin-shaping":
            coin_delta = info.get("coins", 0) - self.prev_coins
            self.prev_coins = info.get("coins", 0)
            reward += coin_delta * self.coin_weight
            if info.get("flag_get", False):
                reward += self.flag_bonus

        elif self.mode == "game-score":
            score_delta = info.get("score", 0) - self.prev_score
            self.prev_score = info.get("score", 0)
            reward += score_delta * self.score_weight
            if info.get("flag_get", False):
                reward += self.flag_bonus

        return obs, reward, done, trunc, info