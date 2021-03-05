import numpy as np
from collections import defaultdict
import random

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv
from karel_env.karel_gym_env import KarelGymEnv


class KarelEnv(GymEnv):
    """Tiny wrapper around GymEnv for Karel tasks."""
    SUBTASKS = ['cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber_sparse', 'placeSetter', 'shelfStocker', 'chainSmoker', 'topOff']
    def __init__(self, config):
        #super().__init__(config)
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
        env_args = AttrDict()
        env_args.update(args)
        self.env = KarelGymEnv(env_args)
        self.env._max_episode_steps=config.max_episode_steps
        self._env = self.env
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = getattr(self.env, 'spec', None)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "karel-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = self.env.step(*args, **kwargs)
        return self.observation(obs.astype(np.float32)), np.float64(rew), np.array(done), {}#info #self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        return ob

    def observation(self, ob):
        if len(self.observation_space.shape) == 3:
            return np.transpose(ob, (self.op[0], self.op[1], self.op[2]))
        return ob
