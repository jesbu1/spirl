import numpy as np
from collections import defaultdict
import random

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv
from spirl.karel_env.karel_gym_env import KarelGymEnv


class KarelEnv(GymEnv):
    """Tiny wrapper around GymEnv for Karel tasks."""
    SUBTASKS = ['cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber_sparse', 'placeSetter', 'shelfStocker', 'chainSmoker', 'topOff']
    def __init__(self, config):
        super().__init__(config)
        args = dict(task_definition='custom_reward',
                env_task=config.subtask,
                max_episode_steps=config.max_episode_steps,
                obv_type=config.obv_type,
                wall_prob=0.25,
                height=config.height,
                width=config.width,
                incorrect_marker_penalty=config.incorrect_marker_penalty,
                delayed_reward=config.delayed_reward,
                seed=random.randint(0, 100000000))
        self.env = KarelGymEnv()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = getattr(self.env, 'spec', None)
        config.subtask

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "karel-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        return super().reset()