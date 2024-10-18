from . import running_objective
from . import rg_env
from ..utils import make_env
import gymnasium


def make_vector_env(env, num_envs):
    return gymnasium.vector.SyncVectorEnv([make_env(env) for i in range(num_envs)])
