"""environment wrappers used for training and evaluation"""


from collections import deque

import gymnasium as gym
import numpy as np


class ObservationStackWrapper(gym.Wrapper):

    def __init__(self, env, stack_size = 4):
        super().__init__(env)
        if stack_size < 1:
            raise ValueError("stack_size must be at least 1")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ObservationStackWrapper requires a Box observation space")

        self.stack_size = stack_size
        self._frames = deque(maxlen=stack_size)

        base_space = env.observation_space
        low = np.tile(base_space.low, stack_size).astype(np.float32)
        high = np.tile(base_space.high, stack_size).astype(np.float32)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(base_space.shape[0] * stack_size,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = np.asarray(obs, dtype=np.float32)
        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(frame.copy())
        return self._stacked_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=np.float32))
        return self._stacked_observation(), reward, terminated, truncated, info

    def _stacked_observation(self):
        return np.concatenate(list(self._frames), dtype=np.float32)
