from __future__ import division
import gym
import gym_super_mario_bros
from gym import Wrapper, ObservationWrapper
import numpy as np
import torch
from collections import deque
from gym.spaces.box import Box
from cv2 import resize, INTER_AREA, INTER_NEAREST
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import random
from random import randint, randrange
from math import sqrt
import pickle


STAGE_LIST = [
    "1-1",
    "1-2",
    "1-3",
    "1-4",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "3-1",
    "3-2",
    "3-3",
    "3-4",
    "4-1",
    "4-2",
    "4-3",
    "4-4",
    "5-1",
    "5-2",
    "5-3",
    "5-4",
    "6-1",
    "6-2",
    "6-3",
    "6-4",
    "7-1",
    "7-2",
    "7-3",
    "7-4",
    "8-1",
    "8-2",
    "8-3",
    "8-4",
]


CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
]


word_stage_time = {
    1: {1: 400, 2: 400, 3: 300, 4: 300},
    2: {1: 400, 2: 400, 3: 300, 4: 300},
    3: {1: 400, 2: 300, 3: 300, 4: 300},
    4: {1: 400, 2: 400, 3: 300, 4: 400},
    5: {1: 300, 2: 400, 3: 300, 4: 300},
    6: {1: 400, 2: 400, 3: 300, 4: 300},
    7: {1: 400, 2: 400, 3: 300, 4: 400},
    8: {1: 300, 2: 400, 3: 300, 4: 400},
}


def mario_env(env_id, args):
    env = gym.make(env_id)
    env = EpisodicLifeEnv(env, args.time_per_stage)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=args.skip_rate)
    env = RewardWrapper(env)
    env._max_episode_steps = args.max_episode_length
    env = MarioRescale(env)
    env = frame_stack(env)
    env = NormalizedEnv(env)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    return env


def process_frame(frame):
    frame = frame[15:-25, 43:-13]
    frame = resize(frame, (100, 100), INTER_AREA)
    frame = resize(frame, (80, 80), INTER_AREA)
    frame = 0.2989 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    return frame.astype(np.float32)


class MarioRescale(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [4, 80, 80], dtype=np.uint8)
        self.obs_keep = None

    def observation(self, observation):
        return process_frame(observation)


class NoopResetEnv(Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        Wrapper.__init__(self, env)
        self.env = env
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [0, noop_max]."""
        obs = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = randint(0, self.noop_max)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env, tps):
        """Make end-of-life == end-of-episode, make end-of-stage == end-of-episode, and make end-of-time == end-of-episode, but only reset on true game over."""
        super().__init__(env)
        self.env = env
        self.lives = 0
        self.time_per_stage = tps
        self.was_real_done = True
        self.time_limit = False
        self.world_num = self.env.unwrapped._world
        self.stage_num = self.env.unwrapped._stage
        self.singleStage = self.env.unwrapped.is_single_stage_env
        self.world_stage_time = word_stage_time
        self.time_level = max(
            self.world_stage_time[self.world_num][self.stage_num] - self.time_per_stage,
            0,
        )
        self.pen_flag = False
        self.rew_flag = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_last = obs
        self.was_real_done = done
        world = self.env.unwrapped._world
        stage = self.env.unwrapped._stage
        lives = self.env.unwrapped._life
        if lives < self.lives and lives != 255:
            done = True
            self.pen_flag = True
        if world > self.world_num or stage > self.stage_num and not done:
            self.world_num = world
            self.stage_num = stage
            self.time_level = max(
                self.world_stage_time[self.world_num][self.stage_num]
                - self.time_per_stage,
                0,
            )
            done = True
            self.rew_flag = True
        if info["time"] < self.time_level and not done:
            if lives == 0 or self.singleStage:
                self.was_real_done = True

            else:
                self.time_limit = True
            done = True
            self.pen_flag = True
        self.lives = lives
        if self.lives == 255:
            self.pen_flag = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            if self.time_limit:
                self.env.unwrapped._kill_mario()
            obs = self.obs_last
        self.time_limit = False
        self.lives = self.env.unwrapped._life
        self.world_num = self.env.unwrapped._world
        self.stage_num = self.env.unwrapped._stage
        self.singleStage = self.env.unwrapped.is_single_stage_env
        self.time_level = max(
            self.world_stage_time[self.world_num][self.stage_num] - self.time_per_stage,
            0,
        )
        return obs


class RewardWrapper(Wrapper):
    def __init__(self, env=None):
        """Reward Mario for each new point further right he achieves"""
        super().__init__(env)
        self.env = env
        self.old_position = self.env.unwrapped._x_position
        self.singleStage = self.env.unwrapped.is_single_stage_env
        self.buzzer_list = []

    def step(self, action):
        self.old_position = max(self.old_position, self.env.unwrapped._x_position)
        total_reward = 0
        obs, reward, done, info = self.env.step(action)
        if info["x_pos"] > 60000:  # In case of glitch with position on some levels
            info["x_pos"] = 1

        reward = min(max(info["x_pos"] - self.old_position, 0.0), 30.0)

        if (
            info["world"] == 7 and info["stage"] == 4
        ):  # To simulate game reward sound when proper path achieved on level. Agent still needs to figure out what that path is to achieve reward
            if info["x_pos"] < 1700:
                self.buzzer_list = []
            if len(self.buzzer_list) > 0:
                if self.buzzer_list[0] <= (self.env.env.env.env._elapsed_steps - 400):
                    self.buzzer_list = []
            if ((1700 < info["x_pos"] < 1846) and info["y_pos"] >= 191) and len(
                self.buzzer_list
            ) == 0:
                self.buzzer_list.append(self.env.env.env.env._elapsed_steps)
            elif (
                ((1903 <= info["x_pos"] <= 1927) and info["y_pos"] == 191)
                and len(self.buzzer_list) == 1
                and self.buzzer_list[0] > (self.env.env.env.env._elapsed_steps - 400)
            ):
                self.buzzer_list.append("First_Spot")
            elif (1984 <= info["x_pos"] <= 2080) and len(self.buzzer_list) >= 2:
                if info["y_pos"] == 127:
                    if "Main_Spot" not in set(self.buzzer_list):
                        self.buzzer_list.append("Main_Spot")
                else:
                    self.buzzer_list = []
            elif (
                (2114 <= info["x_pos"] <= 2164)
                and info["y_pos"] == 127
                and len(self.buzzer_list) == 3
            ):
                self.buzzer_list.append("Last_Spot")
            elif (
                (2200 < info["x_pos"] <= 2440)
                and info["y_pos"] >= 191
                and len(self.buzzer_list) == 4
                and self.buzzer_list[0] > (self.env.env.env.env._elapsed_steps - 400)
            ):
                if set(["First_Spot", "Main_Spot", "Last_Spot"]) < set(
                    self.buzzer_list
                ):
                    reward = 30
                    self.old_position = info["x_pos"]
                self.buzzer_list = []

        # Below is to deter agent from using warp zone to skip to other worlds. If you do want agent to learn discover warp zones and use them just comment out two lines below
        elif (
            (info["world"] == 1 and info["stage"] == 2)
            or (info["world"] == 4 and info["stage"] == 2)
        ) and info["y_pos"] > 254:
            reward = -1.0

        if info["y_pos"] < 70:  # Mario is falling and about to die so penalize
            reward = -1.0

        total_reward += reward
        if done and not self.singleStage:  # Rewards for passing stage or dying
            if self.env.env.env.pen_flag:
                total_reward = -30.0
            if self.env.env.env.rew_flag:
                total_reward = 30.0
        elif done:  # Rewards for passing stage or dying for single stage environment
            if info["flag_get"]:
                total_reward = 30.0
            else:
                total_reward = -30.0
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.old_position = self.env.unwrapped._x_position
        self.buzzer_list = []
        return obs


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self.env = env
        self._obs_buffer = deque(maxlen=1)
        self._skip = skip
        self.buzzer_list = []

    def step(self, action):
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, reward, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self.buzzer_list = []
        self._obs_buffer.append(obs)
        return obs


class frame_stack(Wrapper):
    def __init__(self, env, stack_frames=4):
        Wrapper.__init__(self, env)
        self.stack_frames = stack_frames
        self.frames = deque([], maxlen=self.stack_frames)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self.observation_stack()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.frames.append(ob)
        return self.observation_stack(), rew, done, info

    def observation_stack(self):
        assert len(self.frames) == self.stack_frames
        return np.stack(self.frames, axis=0).reshape((1, 4, 80, 80))


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float32")
        self.var = np.ones(shape, "float32")
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr)
        batch_var = np.var(arr)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class NormalizedEnv(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.obs_rms = RunningMeanStd(shape=())
        self.is_training = False

    def set_training_on(self):
        self.is_training = True

    def set_training_off(self):
        self.is_training = False

    def observation(self, obs):
        if self.is_training:
            self.obs_rms.update(obs)
        obs = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0
        )
        return obs

    def load(self, path):
        with open(f"{path}/obs_rms.pkl", "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        self.obs_rms = vec_normalize
        return

    def save(self, path):
        with open(f"{path}/obs_rms.pkl", "wb") as file_handler:
            pickle.dump(self.obs_rms, file_handler)
