import io
import inspect
import random
import sys
import time
from base64 import b64encode
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, Union, get_type_hints

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import serialization, struct
from flax.core import FrozenDict
from nicegui import app, ui
from PIL import Image

from nicewebrl.logging import get_logger

# Type definitions
TIMESTEP = Any
RENDER_FN = Callable[[TIMESTEP], jax.Array]

# Module-level variables
logger = get_logger(__name__)


def get_rng():
  """Initializes a jax.random number generator or gets the latest if already initialized."""
  app.storage.user["seed"] = app.storage.user.get("seed", random.getrandbits(32))
  app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0)
  if "rng_key" in app.storage.user:
    rng_key = jnp.array(app.storage.user["rng_key"], dtype=jax.numpy.uint32)
    return rng_key
  else:
    rng_key = jax.random.PRNGKey(app.storage.user["seed"])
    app.storage.user["rng_key"] = rng_key.tolist()
    return rng_key


def new_rng():
  """Return a new jax.random number generator or make a new one if not initialized."""
  app.storage.user["seed"] = app.storage.user.get("seed", random.getrandbits(32))
  if "rng_key" in app.storage.user:
    rng_key = jnp.array(app.storage.user["rng_key"], dtype=jax.numpy.uint32)
    rng_key, rng = jax.random.split(rng_key)
    app.storage.user["rng_key"] = rng_key.tolist()
    app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0) + 1
    return rng
  else:
    rng_key = jax.random.PRNGKey(app.storage.user["seed"])
    app.storage.user["rng_key"] = rng_key.tolist()
    return rng_key


def match_types(example: struct.PyTreeNode, data: struct.PyTreeNode):
  def match_types_(ex, d):
    if d is None:
      return None
    else:
      if hasattr(ex, "dtype"):
        return jax.numpy.array(d, dtype=ex.dtype)
      else:
        return d

  return jax.tree_map(match_types_, example, data)


def make_serializable(obj: Any):
  """Convert nested jax objects to serializable python objects"""
  if isinstance(obj, np.ndarray):
    return obj.tolist()  # Convert JAX array to list
  elif isinstance(obj, jnp.ndarray):
    obj = jax.tree_map(np.array, obj)
    return obj.tolist()  # Convert JAX array to list
  elif isinstance(obj, dict):
    return {k: make_serializable(v) for k, v in obj.items()}
  elif isinstance(obj, (list, tuple)):
    return [make_serializable(v) for v in obj]
  elif isinstance(obj, datetime):
    return obj.isoformat()
  else:
    return obj


def base64_npimage(image: np.ndarray):
  image = np.asarray(image)
  buffer = io.BytesIO()
  # Convert to uint8 if float, keep as is if already uint
  if image.dtype.kind == "f":
    image = (image * 255).clip(0, 255).astype("uint8")
  elif image.dtype != np.uint8:
    image = image.astype("uint8")
  Image.fromarray(image).save(buffer, format="JPEG")
  encoded_image = b64encode(buffer.getvalue()).decode("ascii")
  return "data:image/jpeg;base64," + encoded_image


def get_size(pytree):
  leaves = jax.tree_util.tree_leaves(pytree)
  return sum(
    leaf.nbytes if hasattr(leaf, "nbytes") else sys.getsizeof(leaf) for leaf in leaves
  )


class StepType(jnp.uint8):
  FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
  MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
  LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)

ENV_PARAMS = struct.PyTreeNode

class TimeStep(struct.PyTreeNode):
  state: struct.PyTreeNode

  step_type: StepType
  reward: jax.Array
  discount: jax.Array
  observation: jax.Array

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):
    return self.step_type == StepType.LAST


class TimestepWrapper(object):
  """."""

  def __init__(
    self,
    env,
    autoreset: bool = True,
    use_params: bool = True,
    num_leading_dims: int = 0,
    reset_w_batch_dim: bool = True,
  ):
    self._env = env
    self._autoreset = autoreset
    self._use_params = use_params
    self._num_leading_dims = num_leading_dims
    self._reset_w_batch_dim = reset_w_batch_dim
  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def reset(
    self, key: jax.random.PRNGKey, params: Optional[struct.PyTreeNode] = None
  ) -> Tuple[TimeStep, dict]:
    if self._use_params:
      obs, state = self._env.reset(key, params)
    else:
      obs, state = self._env.reset(key)
    # Get shape from first leaf of obs, assuming it's a batch dimension
    if self._reset_w_batch_dim:
      first_leaf = jax.tree_util.tree_leaves(obs)[0]
      shape = first_leaf.shape[: self._num_leading_dims]
    else:
      shape = ()
    timestep = TimeStep(
      state=state,
      observation=obs,
      discount=jnp.ones(shape, dtype=jnp.float32),
      reward=jnp.zeros(shape, dtype=jnp.float32),
      step_type=jnp.full(shape, StepType.FIRST, dtype=StepType.FIRST.dtype),
    )
    return timestep

  def step(
    self,
    key: jax.random.PRNGKey,
    prior_timestep: TimeStep,
    action: Union[int, float],
    params: Optional[struct.PyTreeNode] = None,
  ) -> Tuple[TimeStep, dict]:
    def env_step(prior_timestep_):
      if self._use_params:
        obs, state, reward, done, info = self._env.step(
          key, prior_timestep_.state, action, params
        )
      else:
        obs, state, reward, done, info = self._env.step(
          key, prior_timestep_.state, action
        )
      del info
      if type(done) == dict:  # multi-agent
        done = done['__all__']
      if type(reward) == dict:   # multi-agent
        reward = reward['agent_0'].astype(jnp.float32)
      return TimeStep(
        state=state,
        observation=obs,
        discount=1.0 - jnp.asarray(done).astype(jnp.float32),
        reward=reward,
        step_type=jnp.where(done, StepType.LAST, StepType.MID),
      )

    if self._autoreset:
      # if prior was last, reset
      # otherwise, do regular step
      timestep = jax.lax.cond(
        prior_timestep.last(),
        lambda: self.reset(key, params),
        lambda: env_step(prior_timestep),
      )
    else:
      timestep = env_step(prior_timestep)
    return timestep


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
  else:
    raise ValueError(
      "Cannot determine number of actions for environment. please provide actions"
    )
  actions = jnp.arange(num_actions)
  return actions


class JaxWebEnv:
  def __init__(self, env, actions=None):
    """The main purpose of this class is to precompile jax functions before experiment starts."""
    self.env = env
    assert hasattr(env, "reset"), "env needs reset function"
    assert hasattr(env, "step"), "env needs step function"

    if actions is None:
      actions = try_to_get_actions(env)

    def next_steps(rng, timestep, env_params):
      # vmap over rngs and actions. re-use timestep
      timesteps = jax.vmap(env.step, in_axes=(None, None, 0, None), out_axes=0)(
        rng, timestep, actions, env_params
      )
      return timesteps

    self.reset = jax.jit(env.reset)
    self.next_steps = jax.jit(next_steps)

  def precompile(self, dummy_env_params: Optional[struct.PyTreeNode] = None) -> None:
    """Call this function to pre-compile jax functions before experiment starts."""
    print("Compiling environment reset and step functions.")
    start = time.time()
    dummy_rng = jax.random.PRNGKey(0)
    self.reset = self.reset.lower(dummy_rng, dummy_env_params).compile()
    print(f"\treset time: {time.time() - start}")
    start = time.time()
    timestep = self.reset(dummy_rng, dummy_env_params)
    self.next_steps = self.next_steps.lower(
      dummy_rng, timestep, dummy_env_params
    ).compile()
    print(f"\tstep time: {time.time() - start}")

  def precompile_vmap_render_fn(
    self, render_fn: RENDER_FN, dummy_env_params: struct.PyTreeNode
  ) -> RENDER_FN:
    """Call this function to pre-compile a multi-render function before experiment starts."""
    print("Compiling multi-render function.")
    start = time.time()
    vmap_render_fn = jax.jit(jax.vmap(render_fn))
    dummy_rng = jax.random.PRNGKey(0)
    timestep = self.reset(dummy_rng, dummy_env_params)
    next_timesteps = self.next_steps(dummy_rng, timestep, dummy_env_params)
    vmap_render_fn = vmap_render_fn.lower(next_timesteps).compile()
    print(f"\ttime: {time.time() - start}")
    return vmap_render_fn


class MultiAgentJaxWebEnv:
    def __init__(self, env, actions=None):
        """The main purpose of this class is to precompile jax functions before experiment starts."""
        self.env = env
        assert hasattr(env, 'reset'), 'env needs reset function'
        assert hasattr(env, 'step'), 'env needs step function'

        if actions is None:
            actions = try_to_get_actions(env)

        def reset(rng, params):
            return env.reset(rng, {'random_reset_fn': 'reset_all'})

        def next_steps(rng, timestep, env_params, other_action=4, h_id=0):
            # vmap over rngs and actions. re-use timestep
            def step_env(rng, timestep, agent_0_action, agent_1_action, env_params):
                action_dict = {
                    'agent_0': agent_0_action,
                    'agent_1': agent_1_action,
                }
                return env.step(rng, timestep, action_dict, env_params)
            agent_0_action = jnp.where(h_id == 0, actions, jnp.repeat(other_action, actions.shape[0]))
            agent_1_action = jnp.where(h_id == 1, actions, jnp.repeat(other_action, actions.shape[0]))

            timesteps = jax.vmap(step_env, in_axes=(None, None, 0, 0, None), out_axes=0)(rng, timestep, agent_0_action, agent_1_action, env_params)
            return timesteps

        self.reset = jax.jit(reset)
        self.next_steps = jax.jit(next_steps)

    def precompile(self, dummy_env_params: struct.PyTreeNode={'random_reset_fn': 'reset_all'}) -> None:
        """Call this function to pre-compile jax functions before experiment starts."""
        logger.info("Compiling jax environment functions.")
        start = time.time()
        dummy_rng = jax.random.PRNGKey(0)
        self.reset = self.reset.lower(dummy_rng, dummy_env_params).compile()
        timestep = self.reset(dummy_rng, dummy_env_params)
        self.next_steps = self.next_steps.lower(
            dummy_rng, timestep, dummy_env_params, other_action=4, h_id=0).compile()
        logger.info(f"\ttime: {time.time()-start}")

    def precompile_vmap_render_fn(
            self,
            render_fn: RENDER_FN,
            dummy_env_params: struct.PyTreeNode={'random_reset_fn': 'reset_all'}) -> RENDER_FN:
        """Call this function to pre-compile a multi-render function before experiment starts."""
        logger.info("Compiling multi-render function.")
        start = time.time()
        vmap_render_fn = jax.jit(jax.vmap(render_fn))
        dummy_rng = jax.random.PRNGKey(0)
        timestep = self.reset(dummy_rng, dummy_env_params)
        next_timesteps = self.next_steps(
            dummy_rng, timestep, dummy_env_params, other_action=4, h_id=0)
        vmap_render_fn = vmap_render_fn.lower(
            next_timesteps).compile()
        logger.info(f"\ttime: {time.time()-start}")
        return vmap_render_fn
