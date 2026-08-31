"""
Implements a synchronous multi-human interaction framework for web-based reinforcement learning environments.

This module provides infrastructure for multiple humans to interact simultaneously with a shared
environment through a leader-follower protocol. The main component is the MultiHumanLeaderFollowerEnvStage
class, which:

1. Designates one client as the leader for state management and synchronization
2. Organizes users into rooms for collaborative interaction
3. Collects actions from all users before advancing the environment
4. Ensures state consistency across all clients via broadcast messages
5. Handles serialization and persistence of environment states

The framework is designed for two-agent environments where each human controls one agent,
with support for action synchronization, state rendering, and experiment data collection.
"""

from typing import List, Callable, Dict, Optional, Any, Union
from functools import partial
import itertools
import asyncio
from asyncio import Lock
import aiofiles
import numpy as np

# import base64
import copy
import dataclasses

import jax

from nicegui import app, ui, Client
from nicewebrl.nicejax import new_rng, base64_npimage, make_serializable
from nicewebrl import Timestep
from nicewebrl.logging import get_logger
from nicewebrl.utils import retry_with_exponential_backoff

from nicewebrl import Stage, EnvStageState
from nicewebrl.stages import (
    safe_save,
    time_diff,
    StageStateModel,
    pytree_replace,
    dataclass_replace,
    save_stage_state,
    get_latest_stage_state,
    EnvStage
)
from nicewebrl import broadcast_event_call
from nicewebrl.utils import write_msgpack_record

try:
  jax_tree_map = jax.tree.map
except AttributeError:
  import jax

  jax_tree_map = jax.tree_map
except:
  raise ImportError("Failed to import jax.tree.map or jax.tree_map")

# Make type hints framework-agnostic
FeedbackFn = Callable[[Any], Dict]  # Can be PyTreeNode or TorchEnvStageState


logger = get_logger(__name__)
Image = Union[np.ndarray, Any]  # Can be jnp.ndarray or torch.Tensor

TimestepCallFn = Callable[[Timestep], None]
RenderFn = Callable[[Timestep], Image]

DisplayFn = Callable[["Stage", ui.element, Timestep], None]


def get_room_users():
  room_id = app.storage.user["room_id"]
  return app.storage.general["rooms"][str(room_id)]


async def broadcast_update_environment(action_key: str):
  # called_by_user_id = str(app.storage.user["seed"])
  nicegui_clients = Client.instances.values()
  for client in nicegui_clients:
    with client:
      called_by_room_id = str(app.storage.user["room_id"])
      # stage = app.storage.user["stage_idx"]
      fn = f"updateEnvironment('{called_by_room_id}','{action_key}')"
      logger.info(fn)
      await ui.run_javascript(fn, timeout=10)



@dataclasses.dataclass
class MultiHumanLeaderFollowerEnvStage(Stage):
  """Multi-human collaborative environment stage with leader-follower protocol.

  Note: this currently assumes environments with only 2 agents.
  TODO: generalize to more agents by allowing class to take in functions that construct multi-agent objects

  Framework Support:
      Supports both JAX (default) and PyTorch environments via the `framework` parameter.

      JAX mode (framework="jax"):
          - Uses precomputed next states for efficient rendering
          - Requires vmap_render_fn for batch rendering
          - Faster performance due to precomputation

      PyTorch mode (framework="torch"):
          - Computes next states on-demand (no precomputation)
          - vmap_render_fn must be None (will raise ValueError if provided)
          - Slower performance but supports PyTorch environments
          - Multi-agent environment wrapper must accept actions as a list:
            `step(rng, timestep, [action0, action1], params)`

  Args:
      instruction (str): Text instructions shown to the user for this stage.
      max_episodes (Optional[int]): Maximum number of episodes allowed before stage completion.
      min_success (Optional[int]): Minimum number of successful episodes required to complete stage.
      web_env (Any): The environment instance that handles state transitions and interactions.
          Can be JaxWebEnv (JAX) or TorchRLWebEnv (PyTorch).
      env_params (Any): Parameters for the environment. Can be PyTreeNode (JAX) or TensorDict (PyTorch).
      render_fn (Callable): Function to render the environment state as an image.
      reset_display_fn (Callable): Function called to reset the display between episodes.
      vmap_render_fn (Optional[Callable]): Vectorized version of render_fn for batch processing.
          Required for JAX mode, must be None for PyTorch mode.
      evaluate_success_fn (Callable): Function that takes a timestep and returns 1 for success, 0 for failure.
      check_finished (Callable): Additional function to check if stage should end (beyond max_episodes/min_success).
      custom_data_fn (Optional[Callable]): Optional function to extract additional data from timesteps for logging.
      state_cls (Optional[Any]): Class used to store the stage's state information.
          Can be EnvStageState (JAX) or TorchEnvStageState (PyTorch).
      action_keys (Optional[Dict[int, str]]): Mapping from action indices to keyboard keys.
      action_to_name (Optional[Dict[int, str]]): Optional mapping from action indices to human-readable names.
      next_button (bool): Whether to show a "next" button (default False).
      notify_success (bool): Whether to show success/failure notifications.
      msg_display_time (int): How long to display notification messages (in milliseconds).
      user_save_file_fn (Optional[Callable[[], str]]): Function that returns the path to save user data.
      verbosity (int): Level of logging verbosity (0 for minimal, higher for more).
      framework (str): Framework to use ("jax" or "torch"). Defaults to "jax".
  """

  instruction: str = "instruction"
  min_success: Optional[int] = 1
  max_episodes: Optional[int] = 10
  web_env: Any = None  # Can be JaxWebEnv or TorchRLWebEnv
  env_params: Any = None  # Can be PyTreeNode or TensorDict
  render_fn: RenderFn = None
  reset_display_fn: Optional[DisplayFn] = None
  vmap_render_fn: Optional[Callable] = None
  evaluate_success_fn: TimestepCallFn = None
  check_finished: Optional[TimestepCallFn] = None
  custom_data_fn: Optional[Callable] = None
  state_cls: Optional[Any] = None  # Can be EnvStageState or TorchEnvStageState
  action_keys: Optional[Dict[int, str]] = None
  action_to_name: Optional[List[str]] = None
  next_button: bool = False
  notify_success: bool = True
  msg_display_time: int = None
  user_save_file_fn: Optional[Callable[[], str]] = None
  verbosity: int = 0
  framework: str = "jax"

  def __post_init__(self):
    super().__post_init__()

    self.nicegui_session_id_fn = lambda: app.storage.user["room_id"]
    # determine if we can precompute next steps
    self.precompute_next_steps = (
        self.vmap_render_fn is not None
        and hasattr(self.web_env, "next_steps")
        and callable(getattr(self.web_env, "next_steps"))
    )

    # Framework-specific setup (similar to EnvStage)
    if self.framework == "jax":
        from nicewebrl.nicejax import Serializer
        from nicewebrl import EnvStageState

        self.serializer = Serializer()
        self.state_cls = partial(EnvStageState, name=self.name)
        self._replace_fn = pytree_replace

    elif self.framework == "torch":
        from nicewebrl.nicetorch import Serializer
        from nicewebrl.stages import TorchEnvStageState

        self.serializer = Serializer()
        self.state_cls = partial(TorchEnvStageState, name=self.name)
        self._replace_fn = dataclass_replace
    else:
        raise ValueError(
            f"Unknown framework '{self.framework}' for MultiHumanLeaderFollowerEnvStage"
        )

    self.key_to_action = {k: a for a, k in enumerate(self.action_keys)}
    if self.action_to_name is None:
        self.action_to_name = dict()
    else:
        self.action_to_name = {k: v for k, v in enumerate(self.action_to_name)}

    if self.user_save_file_fn is None:
        self.user_save_file_fn = (
            lambda: f"data/user={app.storage.user.get('seed')}.msgpack"
        )

    if self.check_finished is None:
        self.check_finished = lambda timestep: False

    self._user_queues = {}  # new: dictionary to store per-user queues
    self.room_data = {}

  def get_room_data(self, key=None, value=None):
    room_id = app.storage.user["room_id"]
    self.room_data[room_id] = self.room_data.get(room_id, {})
    room_data = self.room_data[room_id]
    if key is None:
      return room_data
    return room_data.get(key, value)

  def pop_room_data(self, key, value=None):
    room_id = app.storage.user["room_id"]
    self.room_data[room_id] = self.room_data.get(room_id, {})
    return self.room_data[room_id].pop(key, value)

  async def set_room_data(self, **kwargs):
    room_id = app.storage.user["room_id"]
    async with self._lock:
      self.room_data[room_id] = self.room_data.get(room_id, {})
      self.room_data[room_id].update(kwargs)

  def get_user_queue(self):
    """Get queue for current user, creating if needed"""
    user_seed = app.storage.user["seed"]
    if user_seed not in self._user_queues:
      self._user_queues[user_seed] = asyncio.Queue()
    return self._user_queues[user_seed]

  async def finish_saving_user_data(self):
    await self.get_user_queue().join()

  async def _process_save_queue(self):
    """Process all items currently in the queue for current user"""
    queue = self.get_user_queue()
    while not queue.empty():
      args, timestep, user_stats = await queue.get()
      await self.save_experiment_data(
        args,
        timestep=timestep,
        user_stats=user_stats,
      )
      queue.task_done()

  async def display_timestep(self, container, timestep):
    await self.display_fn(
      stage=self,
      container=container,
      timestep=timestep,
    )

    attempt = 0
    while True:
      attempt += 1
      try:
        # Set the timestamp in the browser
        await ui.run_javascript("window.imageSeenTime = new Date();", timeout=2)
        # If successful, we can return immediately
        return
      except Exception as e:
        if attempt % 10 == 0:  # Log every 10 attempts
          logger.warning(
            f"{self.name}: Error getting imageSeenTime (attempt {attempt}): {e}"
          )
        await asyncio.sleep(0.1)  # Short delay between attempts
        if attempt > 100:
          ui.notify("Please refresh the page", type="negative")
          return

  async def step_and_send_timestep(
    self, container, timestep, update_display: bool = True
  ):
    #############################
    # get next images and store them client-side
    # setup code to display next state
    #############################

    if self.precompute_next_steps:
        # Use precomputed next steps (typically JAX)
        rng = new_rng()
        next_timesteps_ = self.web_env.next_steps(rng, timestep, self.env_params)
        # [action 1, action 2, ...]
        next_images = self.vmap_render_fn(next_timesteps_)

        action_product = itertools.product(
            range(next_images.shape[0]), range(next_images.shape[1])
        )
        next_images_base64 = {}
        next_timesteps = {}
        for a1, a2 in action_product:
            key = f"{self.action_keys[a1]},{self.action_keys[a2]}"
            next_images_base64[key] = base64_npimage(next_images[a1, a2])
            next_timesteps[key] = jax.tree_map(lambda t: t[a1, a2], next_timesteps_)

        js_code = f"window.next_states = {next_images_base64};"
        ui.run_javascript(js_code)

        if app.storage.user["leader"]:
            await self.set_room_data(
                next_timesteps=next_timesteps,
                key_presses={},
            )

    else:
        # Cannot precompute, create dummy next_states (typically PyTorch)
        # Actions will be computed on-demand in update_environment
        dummy_next_states = {}
        for a1 in range(len(self.action_keys)):
            for a2 in range(len(self.action_keys)):
                key = f"{self.action_keys[a1]},{self.action_keys[a2]}"
                dummy_next_states[key] = ""

        js_code = f"window.next_states = {dummy_next_states};"
        ui.run_javascript(js_code)

        if app.storage.user["leader"]:
            await self.set_room_data(
                key_presses={},
            )

    #############################
    # display image
    #############################
    if update_display:
      await self.display_fn(
        stage=self,
        container=container,
        timestep=timestep,
      )

      attempt = 0
      while True:
        attempt += 1
        try:
          # Set the timestamp in the browser
          await ui.run_javascript("window.imageSeenTime = new Date();", timeout=2)
          # If successful, we can return immediately
          return
        except Exception as e:
          if attempt % 10 == 0:  # Log every 10 attempts
            logger.warning(
              f"{self.name}: Error getting imageSeenTime (attempt {attempt}): {e}"
            )
          await asyncio.sleep(0.1)  # Short delay between attempts
          if attempt > 100:
            ui.notify("Please refresh the page", type="negative")
            return

    else:
      ui.run_javascript("window.imageSeenTime = window.next_imageSeenTime;", timeout=10)

  async def wait_for_start(
    self,
    container: ui.element,
    timestep: Any,  # Can be JAX PyTreeNode or PyTorch TensorDict
  ):
    ui.run_javascript("window.accept_keys = false;")
    if self.reset_display_fn is not None:
      await self.reset_display_fn(stage=self, container=container, timestep=timestep)

    ui.run_javascript("window.accept_keys = true;")

  async def reset_stage(self) -> EnvStageState:
    rng = new_rng()

    # NEW EPISODE
    timestep = self.web_env.reset(rng, self.env_params)
    return self.state_cls(timestep=timestep)

  async def display_stage(self, container):
    async with self.get_user_lock():
      stage_state = self.get_room_data("stage_state")
      logger.info("displaying stage state")
      if self.get_room_data("loaded_stage_from_memory"):
        # continue from what's on client screen
        await self.step_and_send_timestep(container, timestep=stage_state.timestep)
      else:
        # DISPLAY NEW EPISODE
        await self.wait_for_start(container, stage_state.timestep)
        await self.step_and_send_timestep(container, timestep=stage_state.timestep)

  async def activate(self, container: ui.element):
    """
    This will follower a leader-follower protocol. Here, the leader is the first person in the room.
    The leader will reset the environment and get a new stage state. They will send then a message to all clients to start loading the stage.

    """
    ui.run_javascript(
        f"window.update_from_next_state = {str(self.precompute_next_steps).lower()};"
    )

    if self.verbosity:
      logger.info("=" * 30)
    if self.verbosity:
      logger.info(self.metadata)

    await broadcast_event_call("display_stage", "false")
    ############################
    # Step 1: leader creates a stage state
    # and sends a load message to all clients
    ############################
    logger.info("leader = " + str(app.storage.user["leader"]))
    if not self.get_room_data("stage_started"):
      if app.storage.user["leader"]:
        logger.info("leader loading stage state")
        # reset environment
        rng = new_rng()
        timestep = self.web_env.reset(rng, self.env_params)
        new_stage_state = self.state_cls(timestep=timestep)

        # (potentially) load stage state from memory
        loaded_stage_state = await get_latest_stage_state(
          example=new_stage_state,
          name=self.name,
          serializer=self.serializer,
          nicegui_session_id_fn=self.nicegui_session_id_fn,
        )
        if loaded_stage_state is None:
          logger.info("No stage state found, starting new stage")

          await self.set_room_data(
            stage_state=new_stage_state,
            loaded_stage_from_memory=False,
            stage_started=True,
          )
          asyncio.create_task(
            save_stage_state(
              new_stage_state,
              serializer=self.serializer,
              nicegui_session_id_fn=self.nicegui_session_id_fn,
              )
          )
        else:
          logger.info("Loading stage state from memory")
          await self.set_room_data(
            stage_state=loaded_stage_state,
            loaded_stage_from_memory=True,
            stage_started=True,
          )
        logger.info("leader sending display_stage signal")
        await broadcast_event_call("display_stage", "true")  # all clients
        await broadcast_update_environment(action_key="starting")
    else:
      await broadcast_event_call("display_stage", "true")  # all clients

  async def handle_key_press(self, event, container):
    ############################################
    # check if stage stated/finished already
    ############################################
    if not self.get_room_data("stage_started", False):
      logger.info("stage.handle_key_press: stage not started")
      return
    if self.get_room_data("stage_finished", False):
      # if already did final save, just return
      logger.info("stage.handle_key_press: stage already finished")
      if self.get_room_data("final_save", False):
        logger.info(
          "stage.handle_key_press: stage already finished and final save done"
        )
        return
      logger.info("stage.handle_key_press: saving")

      # did not do final save, so do so now
      # want stage to end on keypress so that
      # notifications are visible at final timestep
      await self.finish_stage()

      # and dismiss any present notifications
      start_notification = self.pop_user_data("start_notification")
      if start_notification:
        start_notification.dismiss()
      success_notification = self.pop_user_data("success_notification")
      if success_notification:
        success_notification.dismiss()
      return

    ############################################
    # register key press
    ############################################
    key = event.args["key"]
    if self.verbosity:
      logger.info(f"handle_key_press key: {key}")
    # valid key press?
    if key not in self.key_to_action:
      return

    # register key press
    user = str(app.storage.user.get("seed"))
    key_presses = self.get_room_data("key_presses", {})
    async with self.get_user_lock():
      # only handle FIRST key press
      if user not in key_presses:
        key_presses[user] = event.args
        # --------------------------------
        # IMPORTANT
        # asynchonously save experiment data by putting in a save queue
        # save prior timestep + current event information
        # --------------------------------
        user_stats = self.user_stats()
        timestep = self.get_room_data("stage_state").timestep
        await self.get_user_queue().put((event.args, timestep, user_stats))
        asyncio.create_task(self._process_save_queue())

      await self.set_room_data(key_presses=key_presses)
    logger.info(
      f"stage.handle_key_press: {len(key_presses)} key_presses: {key_presses}"
    )

    ############################################
    # check if all room users have pressed a key
    # if so, begin update procedure
    ############################################
    room_users = get_room_users()
    if len(room_users) == len(key_presses):
      logger.info(f"stage.handle_key_press: {len(room_users)} room_users: {room_users}")
      user_to_action_idx = app.storage.general.get("user_to_action_idx")
      actions = [None] * len(room_users)

      # get sorted actions
      for user, action_idx in user_to_action_idx.items():
        actions[action_idx] = key_presses[user]["key"]
      action_key = ",".join(actions)
      await broadcast_update_environment(action_key)
    else:
      logger.info("stage.handle_key_press: not all users have pressed a key")

  def user_stats(self):
    stage_state = self.get_room_data("stage_state")
    if stage_state is None:
      return dict()
    return dict(
      nsteps=int(stage_state.nsteps),
      nepisodes=int(stage_state.nepisodes),
      nsuccesses=int(stage_state.nsuccesses),
    )

  async def save_experiment_data(self, args, timestep, user_stats):
    key = args["key"]
    keydownTime = args.get("keydownTime")
    imageSeenTime = args.get("imageSeenTime")
    action_idx = self.key_to_action.get(key, -1)
    action_name = self.action_to_name.get(action_idx, key)

    timestep_data = {}
    if self.custom_data_fn is not None:
      timestep_data = self.custom_data_fn(timestep)
      timestep_data = jax.tree_map(make_serializable, timestep_data)

    serialized_timestep = self.serializer.serialize(timestep)

    step_metadata = copy.deepcopy(self.metadata)
    step_metadata.update(type="EnvStage", **user_stats)

    user_data = dict(
      user_id=app.storage.user["seed"],
      age=app.storage.user.get("age"),
      sex=app.storage.user.get("sex"),
    )

    save_data = dict(
      stage_idx=app.storage.user.get("stage_idx"),
      session_id=app.storage.user["room_id"],
      data=dict(
        image_seen_time=imageSeenTime,
        action_taken_time=keydownTime,
        computer_interaction=key,
        action_name=action_name,
        action_idx=action_idx,
        timelimit=self.duration,
        timestep=serialized_timestep,
        **timestep_data,
      ),
      user_data=user_data,
      metadata=step_metadata,
      name=self.name,
      body=self.body,
    )

    # Use aiofiles for async file I/O
    save_file = self.user_save_file_fn()
    async with aiofiles.open(save_file, "ab") as f:  # Note: open in binary mode
      # Use msgpack to serialize the data, including bytes
      await write_msgpack_record(f, save_data)
      if imageSeenTime is not None and keydownTime is not None:
        stage_state = self.get_room_data("stage_state")
        if self.verbosity:
          logger.info(f"{self.name} saved file")
          logger.info(f"∆t: {time_diff(imageSeenTime, keydownTime) / 1000.0}")
          logger.info(f"stage state: {self.user_stats()}")
          logger.info(f"env step: {stage_state.nsteps}")

      else:
        logger.error(f"{self.name} saved file")
        logger.error(f"stage state: {self.user_stats()}")
        logger.error(f"imageSeenTime={imageSeenTime}, keydownTime={keydownTime}")
        ui.notification("Error: Stage unexpectedly ending early", type="negative")
        await self.set_user_data(finished=True, final_save=True)

    await self.set_user_data(saved_data=True)

  @retry_with_exponential_backoff(max_retries=5, base_delay=1, max_delay=10)
  async def finish_stage(self):
    if not self.get_room_data("started", False):
      return
    if self.get_room_data("finished", False):
      return

    # Wait for any pending saves to complete
    await self.get_user_queue().join()

    # save experiment data so far (prior time-step + resultant action)
    # if finished, save synchronously (to avoid race condition) with next stage
    await self.set_user_data(finished=True, final_save=True)
    logger.info(f"finish_stage {self.name}. stats: {self.user_stats()}")
    imageSeenTime = await ui.run_javascript("getImageSeenTime()", timeout=10)

    start_notification = self.pop_user_data("start_notification")
    if start_notification:
      start_notification.dismiss()
    success_notification = self.pop_user_data("success_notification")
    if success_notification:
      success_notification.dismiss()

    stage_state = self.get_room_data("stage_state")
    await self.save_experiment_data(
      args=dict(
        key="timer",
        keydownTime=imageSeenTime,
        imageSeenTime=imageSeenTime,
      ),
      timestep=stage_state.timestep,
      user_stats=self.user_stats(),
    )

  async def update_environment(self, action_key, container):
    # Get timestep based on whether we can precompute
    if self.precompute_next_steps:
        # Use precomputed next timesteps
        next_timesteps = self.get_room_data("next_timesteps")
        timestep = next_timesteps[action_key]

    else:
        # Compute timestep on-demand
        stage_state = self.get_room_data("stage_state")
        current_timestep = stage_state.timestep

        # Parse action_key to get both actions
        action_strs = action_key.split(",")
        actions = [self.key_to_action[a] for a in action_strs]

        # Compute next timestep
        rng = new_rng()
        timestep = self.web_env.step(rng, current_timestep, actions, self.env_params)

    episode_reset = timestep.first()
    if episode_reset:
      start_notification = self.pop_user_data("start_notification")
      if start_notification:
        start_notification.dismiss()
      success_notification = self.pop_user_data("success_notification")
      if success_notification:
        success_notification.dismiss()

    success = self.evaluate_success_fn(timestep)

    stage_state = self.get_room_data("stage_state")
    stage_state = self._replace_fn(  # Changed from .replace()
      stage_state,
      timestep=timestep,
      nsteps=stage_state.nsteps + 1,
      nepisodes=stage_state.nepisodes + timestep.first(),
      nsuccesses=stage_state.nsuccesses + success,
    )
    if app.storage.user["leader"]:
      # asynchronously save stage state
      # TODO: change saving of stage state so based on room
      asyncio.create_task(
        save_stage_state(
          stage_state,
          serializer=self.serializer,
          nicegui_session_id_fn=self.nicegui_session_id_fn))
      await self.set_room_data(stage_state=stage_state)

    ################
    # Stage over?
    ################
    achieved_min_success = stage_state.nsuccesses >= self.min_success
    achieved_max_episodes = (
      stage_state.nepisodes >= self.max_episodes and timestep.last()
    )
    finished = achieved_min_success or achieved_max_episodes
    stage_finished = finished or self.check_finished(timestep)

    ################
    # Display new data?
    ################
    if episode_reset:
      await self.wait_for_start(container, timestep)

    if self.precompute_next_steps:
      # image is normally updated client-side
      # when episode resets, update server-side
      await self.step_and_send_timestep(
        container,
        timestep,
        update_display=episode_reset,
      )
    else:
      # non-precomputed: always update display
      await self.step_and_send_timestep(
        container,
        timestep,
        update_display=True,
      )
    ################
    # Episode over?
    ################
    if timestep.last():
      if self.verbosity:
        logger.info("-" * 20)
        logger.info("episode over")
        logger.info("-" * 20)
      start_notification = None
      if not stage_finished:
        start_notification = ui.notification(
          "press any arrow key to start next episode",
          position="center",
          type="info",
          timeout=self.msg_display_time,
        )
      else:
        start_notification = ui.notification(
          "press any arrow key to continue",
          position="center",
          type="info",
          timeout=self.msg_display_time,
        )
      success_notification = None
      if self.notify_success:
        if success:
          success_notification = ui.notification(
            "success",
            type="positive",
            position="center",
            timeout=self.msg_display_time,
          )
        else:
          success_notification = ui.notification(
            "failure",
            type="negative",
            position="center",
            timeout=self.msg_display_time,
          )

      await self.set_user_data(
        start_notification=start_notification,
        success_notification=success_notification,
      )

    await self.set_room_data(stage_finished=stage_finished)

  async def handle_button_press(self, container):
    pass  # do nothing


@dataclasses.dataclass
class MultiHumanSequentialEnvStage(EnvStage):

  """Multi-human collaborative environment stage with leader-follower protocol.

  Note: this currently assumes environments with only 2 agents.
  TODO: generalize to more agents by allowing class to take in functions that construct multi-agent objects

  Framework Support:
      Supports both JAX (default) and PyTorch environments via the `framework` parameter.

      JAX mode (framework="jax"):
          - Uses precomputed next states for efficient rendering
          - Requires vmap_render_fn for batch rendering
          - Faster performance due to precomputation

      PyTorch mode (framework="torch"):
          - Computes next states on-demand (no precomputation)
          - vmap_render_fn must be None (will raise ValueError if provided)
          - Slower performance but supports PyTorch environments
          - Multi-agent environment wrapper must accept actions as a list:
            `step(rng, timestep, [action0, action1], params)`

  Args:
      instruction (str): Text instructions shown to the user for this stage.
      max_episodes (Optional[int]): Maximum number of episodes allowed before stage completion.
      min_success (Optional[int]): Minimum number of successful episodes required to complete stage.
      web_env (Any): The environment instance that handles state transitions and interactions.
          Can be JaxWebEnv (JAX) or TorchRLWebEnv (PyTorch).
      env_params (Any): Parameters for the environment. Can be PyTreeNode (JAX) or TensorDict (PyTorch).
      render_fn (Callable): Function to render the environment state as an image.
      reset_display_fn (Callable): Function called to reset the display between episodes.
      vmap_render_fn (Optional[Callable]): Vectorized version of render_fn for batch processing.
          Required for JAX mode, must be None for PyTorch mode.
      evaluate_success_fn (Callable): Function that takes a timestep and returns 1 for success, 0 for failure.
      check_finished (Callable): Additional function to check if stage should end (beyond max_episodes/min_success).
      custom_data_fn (Optional[Callable]): Optional function to extract additional data from timesteps for logging.
      state_cls (Optional[Any]): Class used to store the stage's state information.
          Can be EnvStageState (JAX) or TorchEnvStageState (PyTorch).
      action_keys (Optional[Dict[int, str]]): Mapping from action indices to keyboard keys.
      action_to_name (Optional[Dict[int, str]]): Optional mapping from action indices to human-readable names.
      next_button (bool): Whether to show a "next" button (default False).
      notify_success (bool): Whether to show success/failure notifications.
      msg_display_time (int): How long to display notification messages (in milliseconds).
      user_save_file_fn (Optional[Callable[[], str]]): Function that returns the path to save user data.
      verbosity (int): Level of logging verbosity (0 for minimal, higher for more).
      framework (str): Framework to use ("jax" or "torch"). Defaults to "jax".
  """

  instruction: str = "instruction"
  min_success: Optional[int] = 1
  max_episodes: Optional[int] = 10
  web_env: Any = None  # Can be JaxWebEnv or TorchRLWebEnv
  env_params: Any = None  # Can be PyTreeNode or TensorDict
  render_fn: RenderFn = None
  reset_display_fn: Optional[DisplayFn] = None
  vmap_render_fn: Optional[Callable] = None
  evaluate_success_fn: TimestepCallFn = None
  check_finished: Optional[TimestepCallFn] = None
  custom_data_fn: Optional[Callable] = None
  state_cls: Optional[Any] = None  # Can be EnvStageState or TorchEnvStageState
  action_keys: Optional[Dict[int, str]] = None
  action_to_name: Optional[List[str]] = None
  next_button: bool = False
  notify_success: bool = True
  msg_display_time: int = None
  user_save_file_fn: Optional[Callable[[], str]] = None
  verbosity: int = 0
  framework: str = "jax"
  precompute_next_steps: bool = False
  nplayers: int = 2

  def __post_init__(self):
    super().__post_init__()
  
    self.nicegui_session_id_fn = lambda: app.storage.user["room_id"]

    if self.precompute_next_steps:
      raise ValueError(
        "MultiHumanSequentialEnvStage does not support precomputing next steps."
      )

  def get_key_for_data(self):
    return app.storage.user["room_id"]

  async def wait_for_start(
      self,
      container: ui.element,
      timestep: Any,  # Can be JAX PyTreeNode or PyTorch TensorDict
  ):
    await ui.run_javascript("window.accept_keys = false;")
    if self.reset_display_fn is not None:
      await self.reset_display_fn(stage=self, container=container, timestep=timestep)

    await ui.run_javascript("window.accept_keys = true;")

  def user_stats(self):
    stage_state = self.get_user_data("stage_state")
    if stage_state is None:
      return dict()
    return dict(
        nsteps=int(stage_state.nsteps),
        nepisodes=int(stage_state.nepisodes),
        nsuccesses=int(stage_state.nsuccesses),
    )

  async def display_stage(self, container):
    async with self.get_user_lock():
      stage_state = self.get_user_data("stage_state")
      if self.get_user_data("loaded_stage_from_memory"):
        logger.info("displaying loaded stage state")
        # continue from what's on client screen
        if stage_state.timestep.first():
          await self.wait_for_start(container)
        await self.step_and_send_timestep(container, timestep=stage_state.timestep)
      else:
        # DISPLAY NEW EPISODE
        logger.info("displaying new stage state")
        await self.wait_for_start(container, stage_state.timestep)
        await self.step_and_send_timestep(container, timestep=stage_state.timestep)

  async def activate(self, container: ui.element):
    """
    This will follower a leader-follower protocol. Here, the leader is the first person in the room.
    The leader will reset the environment and get a new stage state. They will send then a message to all clients to start loading the stage.

    """
    await ui.run_javascript(
        f"window.update_from_next_state = {str(self.precompute_next_steps).lower()};"
    )

    if self.verbosity:
      logger.info("=" * 30)
    if self.verbosity:
      logger.info(self.metadata)

    #await broadcast_event_call("display_stage", "false")
    ############################
    # Step 1: leader creates a stage state
    # and sends a load message to all clients
    ############################
    logger.info("leader = " + str(app.storage.user["leader"]))

    clients_in_room = self.get_user_data("clients_in_room", set())
    await self.set_user_data(
        clients_in_room=clients_in_room.union({app.storage.user["seed"]})
    )

    if len(clients_in_room) == self.nplayers:
      ui.notify("Enough players to start game", type="positive")
      return

    if not self.get_user_data("stage_started"):
      logger.info("starting stage")
      if app.storage.user["leader"]:

        logger.info("leader loading stage state")
        # reset environment
        rng = new_rng()
        timestep = self.web_env.reset(rng, self.env_params)
        new_stage_state = self.state_cls(timestep=timestep)

        # (potentially) load stage state from memory
        loaded_stage_state = await get_latest_stage_state(
            example=new_stage_state,
            name=self.name,
            serializer=self.serializer,
            nicegui_session_id_fn=self.nicegui_session_id_fn,
        )
        if loaded_stage_state is None:
          logger.info("No stage state found, starting new stage")

          await self.set_user_data(
              stage_state=new_stage_state,
              loaded_stage_from_memory=False,
              stage_started=True,
              current_turn=0,
          )
          asyncio.create_task(
              save_stage_state(
                  new_stage_state,
                  serializer=self.serializer,
                  nicegui_session_id_fn=self.nicegui_session_id_fn,
              )
          )
        else:
          logger.info("Loading stage state from memory")
          await self.set_user_data(
              stage_state=loaded_stage_state,
              loaded_stage_from_memory=True,
              stage_started=True,
          )
        logger.info("leader sending display_stage signal")
        await broadcast_event_call("display_stage", "true")  # all clients
      else:
        logger.info("follower waiting for leader to load stage")
    else:
      await broadcast_event_call("display_stage", "true")  # all clients

  async def handle_key_press(self, event, container):
    ############################################
    # check if stage stated/finished already
    ############################################
    if not self.get_user_data("stage_started", False):
      logger.info("stage.handle_key_press: stage not started")
      return

    if self.get_user_data("stage_finished", False):
      # if already did final save, just return
      logger.info("stage.handle_key_press: stage already finished")
      if self.get_user_data("final_save", False):
        logger.info(
            "stage.handle_key_press: stage already finished and final save done"
        )
        return
      logger.info("stage.handle_key_press: saving")

      # did not do final save, so do so now
      # want stage to end on keypress so that
      # notifications are visible at final timestep
      await self.finish_stage()

      # and dismiss any present notifications
      start_notification = self.pop_user_data("start_notification")
      if start_notification:
        start_notification.dismiss()
      success_notification = self.pop_user_data("success_notification")
      if success_notification:
        success_notification.dismiss()
      return

    ############################################
    # register key press
    ############################################
    key = event.args["key"]
    if self.verbosity:
      logger.info(f"handle_key_press key: {key}")
    # valid key press?
    if key not in self.key_to_action:
      return

    clients_in_room = self.get_user_data("clients_in_room", set())
    if len(clients_in_room) < self.nplayers:
      logger.info(
          f"stage.handle_key_press: waiting for all players to join ({len(clients_in_room)}/{self.nplayers})"
      )
      ui.notify("Won't start game until have enough players", type="warning")
      return

    #############################
    # is it this client's turn to act?
    #############################
    client = event.args['client']
    client_idx = app.storage.general["user_to_action_idx"][str(client)]
    current_turn = self.get_user_data("current_turn", 0)
    logger.info(f"requested: {client_idx}, active: {current_turn}")
    if int(current_turn) != int(client_idx):
      logger.info("not this client's turn to act")
      return
    logger.info("client will save data and broadcast action")

    #############################
    # save experiment data
    #############################
    save_data = await self.prepare_save_data(event)
    asyncio.create_task(self.save_key_data(save_data))

    #############################
    # update environment
    #############################
    action_idx = self.key_to_action[key]
    if self.precompute_next_steps:
        # Use precomputed next timesteps
        next_timesteps = self.get_user_data("next_timesteps")
        timestep = next_timesteps[action_idx]

    else:
        # Compute timestep on-demand
        stage_state = self.get_user_data("stage_state")
        current_timestep = stage_state.timestep

        # Parse key to get both actions
        action_idx = self.key_to_action[key]

        # Compute next timestep
        rng = new_rng()
        timestep = self.web_env.step(
            rng, current_timestep, action_idx, self.env_params)

    #############################
    # update stage variables
    #############################
    episode_reset = timestep.first()
    if episode_reset:
      start_notification = self.pop_user_data("start_notification")
      if start_notification:
        start_notification.dismiss()
      success_notification = self.pop_user_data("success_notification")
      if success_notification:
        success_notification.dismiss()

    success = self.evaluate_success_fn(timestep, self.env_params)
    stage_state = self.get_user_data("stage_state")
    stage_state = self._replace_fn(
        stage_state,
        timestep=timestep,
        nsteps=stage_state.nsteps + 1,
        nepisodes=stage_state.nepisodes + timestep.first(),
        nsuccesses=stage_state.nsuccesses + success,
    )

    # asynchronously save stage state
    asyncio.create_task(save_stage_state(
        stage_state,
        serializer=self.serializer,
        nicegui_session_id_fn=self.nicegui_session_id_fn))
    await self.set_user_data(
        stage_state=stage_state,
        current_turn=(current_turn + 1) % self.nplayers,
      )
    
    ################
    # now tell all clients to update environment
    ################
    # NOTE: main difference from envir stage
    await broadcast_event_call("display_stage", "true")


    ################
    # Stage over?
    ################
    achieved_min_success = stage_state.nsuccesses >= self.min_success
    achieved_max_episodes = (
        stage_state.nepisodes >= self.max_episodes and timestep.last()
    )
    finished = achieved_min_success or achieved_max_episodes
    stage_finished = finished or self.check_finished(timestep)

    ################
    # Display new data?
    ################
    if episode_reset:
      await self.wait_for_start(container, timestep)

    ################
    # Episode over?
    ################
    if timestep.last():
      if self.verbosity:
        logger.info("-" * 20)
        logger.info("episode over")
        logger.info("-" * 20)

      start_notification = None
      ################
      # stage finished?
      ################
      if stage_finished:
        if self.verbosity:
          logger.info("stage finished")
        start_notification = ui.notification(
            "press any arrow key to continue",
            position="center",
            type="info",
            timeout=self.msg_display_time,
        )
      else:
        start_notification = ui.notification(
            "press any arrow key to start next episode",
            position="center",
            type="info",
            timeout=self.msg_display_time,
        )
      ################
      # Notify
      ################
      success_notification = None
      if self.notify_success:
        if success:
          success_notification = ui.notification(
              "success",
              type="positive",
              position="center",
              timeout=self.msg_display_time,
          )
        else:
          success_notification = ui.notification(
              "failure",
              type="negative",
              position="center",
              timeout=self.msg_display_time,
          )

      await self.set_user_data(
          start_notification=start_notification,
          success_notification=success_notification,
      )

    await self.set_user_data(stage_finished=stage_finished)


  def current_turn(self):
    return self.get_user_data("current_turn", 0)

  def current_player(self):
    current_room = app.storage.user["room_id"]
    ids_in_room = app.storage.general["rooms"][current_room]
    current_id = str(app.storage.user["seed"])
    return ids_in_room.index(current_id)

  async def prepare_save_data(self, event):
    user_stats = self.user_stats()
    timestep = self.get_user_data("stage_state").timestep
    processed_timestep = self.preprocess_timestep(timestep)
    return (event.args, processed_timestep, user_stats)

