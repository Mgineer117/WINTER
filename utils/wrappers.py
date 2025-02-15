import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


class GridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super(GridWrapper, self).__init__(env)
        self.agent_num = args.agent_num

    def get_agent_pos(self):
        agent_pos = np.full((2 * self.agent_num,), np.nan, dtype=np.float32)
        for i in range(self.agent_num):
            agent_pos[2 * i : 2 * i + 2] = self.env.agents[i].pos
        return agent_pos

    def get_step(self, action):
        action = np.argmax(action)
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        observation = observation["image"]
        return observation, reward, termination, truncation, info

    def reset(self, **kwargs):
        if not "options" in kwargs:
            options = {"random_init_pos": False}
            kwargs["options"] = options

        observation, info = self.env.reset(**kwargs)
        observation = observation["image"]

        obs = {}
        obs["observation"] = observation
        return obs, info

    def step(self, action):
        observation, reward, term, trunc, info = self.get_step(action)
        obs = {}
        obs["observation"] = observation
        return obs, reward, term, trunc, info


class CtFWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super(CtFWrapper, self).__init__(env)
        self.agent_num = args.agent_num

    def get_agent_pos(self):
        agent_pos = np.full((2 * self.agent_num,), np.nan, dtype=np.float32)
        for i in range(self.agent_num):
            agent_pos[2 * i : 2 * i + 2] = self.env.agents[i].pos
        return agent_pos

    def get_step(self, action):
        action = np.argmax(action)
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        return observation, reward, termination, truncation, info

    def reset(self, **kwargs):
        if not "options" in kwargs:
            options = {"random_init_pos": False}
            kwargs["options"] = options

        observation, _ = self.env.reset(**kwargs)

        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, {}

    def step(self, action):
        observation, reward, term, trunc, info = self.get_step(action)
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, reward, term, trunc, info


class NavigationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args: float = 1e-1):
        super(NavigationWrapper, self).__init__(env)
        self.cost_scaler = args.cost_scaler

    def get_agent_pos(self):
        return np.array([0, 0])

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        obs = {}
        obs["observation"] = observation
        return obs, {}

    def step(self, action):
        # Call the original step method
        observation, reward, cost, termination, truncation, info = self.env.step(action)

        obs = {}
        obs["observation"] = observation

        reward -= self.cost_scaler * cost

        if info["goal_met"]:
            truncation = True

        return obs, reward, termination, truncation, info


class GymWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super(GymWrapper, self).__init__(env)

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        obs = {}
        obs["observation"] = observation
        return obs, {}

    def step(self, action):
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)

        obs = {}
        obs["observation"] = observation

        return obs, reward, termination, truncation, info
