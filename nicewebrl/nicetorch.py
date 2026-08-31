from typing import Any

import time
import io
import numpy as np


import torch
from tensordict import TensorDict
import logging


from nicewebrl.logging import get_logger
from nicewebrl import Timestep, StepType


logger = get_logger(__name__)


class Serializer:
  """Serialize and deserialize TensorDict and TorchEnvStageState using torch."""

  def serialize(self, obj: Any) -> bytes:
    """Serialize TensorDict or TorchEnvStageState to bytes."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    bytes_data = buffer.getvalue()
    return bytes_data

  def deserialize(self, serialized: bytes, template: Any = None) -> Any:
    """Deserialize bytes back to TensorDict or TorchEnvStageState.

    Args:
        serialized: Bytes to deserialize
        template: Optional template for type checking (for API compatibility with JAX serializer)

    Returns:
        Deserialized object (TensorDict or TorchEnvStageState)
    """
    buffer = io.BytesIO(serialized)
    obj = torch.load(buffer, weights_only=False)
    return obj


def try_to_get_actions(env):
  if hasattr(env, "num_actions"):
    if callable(getattr(env, "num_actions")):
      num_actions = env.num_actions()
    else:
      num_actions = env.num_actions
  elif hasattr(env, "action_space"):
    if callable(getattr(env, "action_space")):
      num_actions = env.action_space().n
    else:
      num_actions = env.action_space.n
  elif hasattr(env, "action_spec"):
    breakpoint()
    if callable(getattr(env, "action_spec")):
      num_actions = env.action_space().n
    else:
      num_actions = env.action_space.n
  else:
    raise ValueError(
      "Cannot determine number of actions for environment. please provide actions"
    )
  return torch.arange(num_actions, dtype=torch.long)


class TorchRLWebEnv:
  def __init__(self, env, actions=None, compile: bool = True, **kwargs):
    """Precompile env + thin wrapper to match nicewebrl env api"""
    self._env = env
    assert hasattr(env, "reset"), "env needs reset function"
    assert hasattr(env, "step"), "env needs step function"

    if actions is None:
      actions = try_to_get_actions(env)
    self.actions = torch.as_tensor(actions, dtype=torch.int64)

    env_reset, env_step = env.reset, env._step

    if compile:
      logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
      print("Compiling environment reset and step functions.")
      start = time.time()
      env_reset = torch.compile(env_reset, fullgraph=False)
      env_step = torch.compile(env_step, fullgraph=False)
      print(f"\ttime: {time.time() - start}")
    self.env_reset = env_reset
    self.env_step = env_step

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def reset(self, rng=None, env_params=None):
    state = self.env_reset()
    return Timestep(
      state=state,
      observation=None,
      discount=np.ones((), dtype=np.float32),
      reward=np.zeros((), dtype=np.float32),
      step_type=np.full((), StepType.FIRST, dtype=StepType.FIRST.dtype),
    )

  def step(self, rng=None, timestep=None, action=None, env_params=None):
    state = timestep.state
    state = state.clone().set("action", torch.tensor(action))
    next_state = self.env_step(state)
    done = next_state["done"].squeeze()
    return Timestep(
      state=next_state,
      observation=None,
      discount=1.0 - np.asarray(done).astype(np.float32),
      reward=next_state["reward"].squeeze(),
      step_type=np.where(done, StepType.LAST, StepType.MID),
    )
