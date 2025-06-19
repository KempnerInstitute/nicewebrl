import asyncio
import copy
import dataclasses
import random
import uuid
from asyncio import Lock
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, struct
from nicegui import app, ui
from tortoise import fields, models

# Local application imports
from nicewebrl import user_data_file
from nicewebrl.container import Container
from nicewebrl.logging import get_logger
from nicewebrl.nicejax import (
    JaxWebEnv,
    MultiAgentJaxWebEnv,
    TimeStep,
    base64_npimage,
    make_serializable,
    new_rng,
)
from nicewebrl.utils import retry_with_exponential_backoff, write_msgpack_record


from jax.tree_util import tree_map as jax_tree_map

# Type definitions
FEEDBACK_FN = Callable[[struct.PyTreeNode], Dict]
IMAGE = jnp.ndarray
PARAMS = struct.PyTreeNode
TIMESTEP_CALL_FN = Callable[[TimeStep], None]
RENDER_FN = Callable[[TimeStep], IMAGE]
DISPLAY_FN = Callable[["Stage", ui.element, TimeStep], None]

logger = get_logger(__name__)


def time_diff(t1, t2) -> float:
  # Convert string timestamps to datetime objects
  t1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%fZ")
  t2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%fZ")

  # Calculate the time difference
  time_difference = t2 - t1

  # Convert the time difference to milliseconds
  return time_difference.total_seconds() * 1000


class StageStateModel(models.Model):
  id = fields.IntField(primary_key=True)
  name = fields.CharField(max_length=255, index=True)
  session_id = fields.CharField(max_length=255, index=True)
  # stage_idx = fields.IntField(index=True)
  data = fields.BinaryField()

  class Meta:
    table = "stage"


class EnvStageState(struct.PyTreeNode):
  timestep: struct.PyTreeNode
  nsteps: int = 1
  nepisodes: int = 1
  nsuccesses: int = 0
  name: str = "stage"


async def get_latest_stage_state(
  example: struct.PyTreeNode, name: str
) -> StageStateModel | None:
  logger.info("Getting latest stage state")
  latest = (
    await StageStateModel.filter(
      session_id=app.storage.browser["id"],
      name=name,
      # stage_idx=app.storage.user["stage_idx"],
    )
    .order_by("-id")
    .first()
  )

  if latest is not None:
    latest = serialization.from_bytes(example, latest.data)

  return latest


async def safe_save(
  model: models.Model,
  max_retries: int = 5,
  base_delay: float = 0.3,
  max_delay: float = 5.0,
  synchronous: bool = True,
):
  """Helper function to safely save model data with retries.

  Args:
      model: Tortoise model instance to save
      max_retries: Maximum number of retry attempts
      base_delay: Initial delay between retries in seconds
      max_delay: Maximum delay between retries in seconds
      synchronous: If True, await the save; if False, create background task
  """

  async def _save():
    from tortoise.exceptions import IntegrityError, OperationalError

    for attempt in range(max_retries):
      try:
        await model.save()
        return
      except (IntegrityError, OperationalError) as e:
        if attempt == max_retries - 1:
          logger.error(
            f"Database conflict while saving {model.__class__.__name__} after {max_retries} attempts: {e}"
          )
          raise

        # Add some random jitter to help prevent repeated collisions
        jitter = random.uniform(0, 0.1)  # 0-100ms random jitter
        delay = min(base_delay * (2**attempt) + jitter, max_delay)
        logger.warning(
          f"Database conflict while saving {model.__class__.__name__} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
        )
        await asyncio.sleep(delay)
      except Exception as e:
        logger.error(f"Unexpected error saving {model.__class__.__name__}: {e}")
        raise

  if synchronous:
    await _save()
  else:
    asyncio.create_task(_save())


async def save_stage_state(
  stage_state, max_retries: int = 5, base_delay: float = 0.3, max_delay: float = 5.0
):
  model = StageStateModel(
    session_id=app.storage.browser["id"],
    # stage_idx=app.storage.user["stage_idx"],
    name=stage_state.name,
    data=serialization.to_bytes(stage_state),
  )
  await safe_save(
    model,
    max_retries=max_retries,
    base_delay=base_delay,
    max_delay=max_delay,
    synchronous=True,
  )


@dataclasses.dataclass
class Stage:
  name: str = "stage"
  title: str = "stage"
  body: str = "text"
  metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
  display_fn: DISPLAY_FN = None
  custom_key_press_fn: Callable = None
  finished: bool = False
  next_button: bool = True
  duration: int = None

  def __post_init__(self):
    self.user_data = {}
    self._lock = Lock()  # Add lock for thread safety
    self._user_locks = {}  # Dictionary to store per-user locks
    self.unique_id = f"stage_{uuid.uuid4().hex}"
    self.metadata.update(
      type="Stage",
      unique_id=self.unique_id,
    )

  def get_user_data(self, key, value=None):
    user_seed = app.storage.user["seed"]
    self.user_data[user_seed] = self.user_data.get(user_seed, {})
    return self.user_data[user_seed].get(key, value)

  def pop_user_data(self, key, value=None):
    user_seed = app.storage.user["seed"]
    self.user_data[user_seed] = self.user_data.get(user_seed, {})
    return self.user_data[user_seed].pop(key, value)

  async def set_user_data(self, **kwargs):
    user_seed = app.storage.user["seed"]
    async with self._lock:
      self.user_data[user_seed] = self.user_data.get(user_seed, {})
      self.user_data[user_seed].update(kwargs)

  def get_user_lock(self):
    user_seed = app.storage.user["seed"]
    if user_seed not in self._user_locks:
      self._user_locks[user_seed] = Lock()
    return self._user_locks[user_seed]

  async def activate(self, container: ui.element):
    await self.display_fn(stage=self, container=container)

  async def finish_stage(self):
    await self.set_user_data(finished=True)

  async def handle_key_press(self, e, container):
    if self.custom_key_press_fn is not None:
      await self.custom_key_press_fn(e, container)
    else:
      await self.finish_stage()

  async def handle_button_press(self, container):
    await self.finish_stage()

  async def finish_saving_user_data(self):
    pass


@dataclasses.dataclass
class FeedbackStage(Stage):
  """A simple feedback stage to collect data from a participant.

  I assume that the display_fn will return once user data is collected and the stage is over. 
  The display_fn should return a dictionary collected data. This is added to the data field of the ExperimentData object.
  """

  next_button: bool = False
  user_save_file_fn: Callable[[], str] = None

  def __post_init__(self):
    super().__post_init__()
    if self.user_save_file_fn is None:
      self.user_save_file_fn = user_data_file
    self.metadata.update(
      type="FeedbackStage",
    )

  async def activate(self, container: ui.element):
    results = await self.display_fn(stage=self, container=container)
    user_data = dict(
      user_id=app.storage.user["seed"],
      age=app.storage.user.get("age"),
      sex=app.storage.user.get("sex"),
    )
    metadata = copy.deepcopy(self.metadata)

    save_data = dict(
      stage_idx=app.storage.user["stage_idx"],
      name=self.name,
      body=self.body,
      session_id=app.storage.browser["id"],
      data=results,
      user_data=user_data,
      metadata=metadata,
    )
    save_file = self.user_save_file_fn()
    async with aiofiles.open(save_file, "ab") as f:
      await write_msgpack_record(f, save_data)
    await self.finish_stage()


@dataclasses.dataclass
class EnvStage(Stage):
  """A stage class for handling interactive environment episodes.

  This class manages the interaction between a user and an environment, handling
  state transitions, user inputs, and data collection.

  Args:
      instruction (str): Text instructions shown to the user for this stage.
      max_episodes (Optional[int]): Maximum number of episodes allowed before stage completion.
      min_success (Optional[int]): Minimum number of successful episodes required to complete stage.
      web_env (Any): The environment instance that handles state transitions and interactions.
      env_params (struct.PyTreeNode): Parameters for the environment.
      render_fn (Callable): Function to render the environment state as an image.
      reset_display_fn (Callable): Function called to reset the display between episodes.
      vmap_render_fn (Callable): Vectorized version of render_fn for batch processing.
      evaluate_success_fn (Callable): Function that takes a timestep and returns 1 for success, 0 for failure.
      check_finished (Callable): Additional function to check if stage should end (beyond max_episodes/min_success).
      custom_data_fn (Callable): Optional function to extract additional data from timesteps for logging.
      state_cls (EnvStageState): Class used to store the stage's state information.
      action_to_name (Dict[int, str]): Optional mapping from action indices to human-readable names.
      next_button (bool): Whether to show a "next" button (default False).
      notify_success (bool): Whether to show success/failure notifications.
      msg_display_time (int): How long to display notification messages (in milliseconds).
      end_on_final_timestep (bool): Whether to end the stage on the final timestep.
      user_save_file_fn (Callable[[], str]): Function that returns the path to save user data.
      verbosity (int): Level of logging verbosity (0 for minimal, higher for more).
      precompile (bool): Whether to precompile the render_fn.
  """

  instruction: str = "instruction"
  min_success: Optional[int] = 1
  max_episodes: Optional[int] = 10
  web_env: JaxWebEnv = None
  env_params: struct.PyTreeNode = None
  render_fn: RENDER_FN = None
  reset_display_fn: Optional[DISPLAY_FN] = None
  vmap_render_fn: Optional[Callable] = None
  evaluate_success_fn: TIMESTEP_CALL_FN = None
  check_finished: Optional[TIMESTEP_CALL_FN] = None
  custom_data_fn: Optional[Callable] = None
  state_cls: Optional[EnvStageState] = None
  action_keys: Optional[Dict[int, str]] = None
  action_to_name: Optional[List[str]] = None
  next_button: bool = False
  notify_success: bool = True
  msg_display_time: int = None
  user_save_file_fn: Optional[Callable[[], str]] = None
  autoreset_on_done: bool = False
  ignore_missing_data: bool = True
  verbosity: int = 0
  preprocess_timestep: Optional[Callable[[TimeStep], TimeStep]] = lambda t: t
  precompile: bool = True

  def __post_init__(self):
    super().__post_init__()
    if self.vmap_render_fn is None:
      if self.precompile:
        self.vmap_render_fn = self.web_env.precompile_vmap_render_fn(
          self.render_fn, self.env_params
        )
      else:
        self.vmap_render_fn = jax.jit(jax.vmap(self.render_fn))

    self.key_to_action = {k: a for a, k in enumerate(self.action_keys)}
    if self.action_to_name is None:
      self.action_to_name = dict()
    else:
      self.action_to_name = {k: v for k, v in enumerate(self.action_to_name)}

    if self.user_save_file_fn is None:
      self.user_save_file_fn = user_data_file


    if self.check_finished is None:
      self.check_finished = lambda timestep: False

    if self.state_cls is None:
      self.state_cls = partial(EnvStageState, name=self.name)

    self._user_queues = {}  # new: dictionary to store per-user queues

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
        await ui.run_javascript("window.imageSeenTime = new Date();", timeout=10)
        # If successful, we can return immediately
        return
      except Exception as e:
        if attempt % 10 == 0:  # Log every 10 attempts
          logger.warning(
            f"'{self.name}': Error getting imageSeenTime (attempt {attempt}): {e}"
          )
        await asyncio.sleep(0.1)  # Short delay between attempts
        if attempt >= 10:
          with container:
            ui.notify("Please refresh the page", type="negative")
          return

  async def step_and_send_timestep(self, container, update_display: bool = True):
    #############################
    # get next images and store them client-side
    # setup code to display next state
    #############################
    rng = new_rng()
    timestep = self.get_user_data("stage_state").timestep
    next_timesteps = self.web_env.next_steps(rng, timestep, self.env_params)
    next_images = self.vmap_render_fn(next_timesteps)

    next_images = {
      self.action_keys[idx]: base64_npimage(image)
      for idx, image in enumerate(next_images)
    }

    js_code = f"window.next_states = {next_images};"

    ui.run_javascript(js_code)

    await self.set_user_data(next_timesteps=next_timesteps)
    #############################
    # display image
    #############################
    if update_display:
      await self.display_timestep(container, timestep)
    else:
      ui.run_javascript("window.imageSeenTime = window.next_imageSeenTime;", timeout=10)

  async def wait_for_start(
    self,
    container: ui.element,
  ):
    ui.run_javascript("window.accept_keys = false;")
    if self.reset_display_fn is not None:
      await self.reset_display_fn(
        stage=self,
        container=container,
        timestep=self.get_user_data("stage_state").timestep,
      )

    ui.run_javascript("window.accept_keys = true;")

  async def activate(self, container: ui.element):
    """

    First reset stage and get a new stage state.
    Then try to load stage state from memory using the stage state to get the right types.
    If no stage state is found, continue with the new stage state.
    """

    if self.verbosity:
      logger.info("=" * 30)
    if self.verbosity:
      logger.info(self.metadata)

    # reset stage
    rng = new_rng()
    timestep = self.web_env.reset(rng, self.env_params)
    new_stage_state = self.state_cls(timestep=timestep)

    # (potentially) load stage state from memory
    loaded_stage_state = await get_latest_stage_state(
      example=new_stage_state,
      name=self.name,
    )

    if loaded_stage_state is None:
      logger.info(f"No stage {self.name} state found, starting new stage")
      # await self.start_stage(container, new_stage_state)
      await self.set_user_data(stage_state=new_stage_state)
      asyncio.create_task(save_stage_state(new_stage_state))

      # DISPLAY NEW EPISODE
      await self.wait_for_start(container)
      await self.step_and_send_timestep(container)

    else:
      logger.info(f"Loading stage {self.name} state from memory")
      await self.set_user_data(stage_state=loaded_stage_state)
      await self.step_and_send_timestep(container)

    await self.set_user_data(started=True)
    ui.run_javascript("window.accept_keys = true;")

  def user_stats(self):
    stage_state = self.get_user_data("stage_state")
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
      timestep_data = jax_tree_map(make_serializable, timestep_data)

    serialized_timestep = serialization.to_bytes(timestep)

    step_metadata = copy.deepcopy(self.metadata)
    step_metadata.update(type="EnvStage", **user_stats)

    user_data = dict(
      user_id=app.storage.user["seed"],
      age=app.storage.user.get("age"),
      sex=app.storage.user.get("sex"),
    )

    save_data = dict(
      stage_idx=app.storage.user.get("stage_idx"),
      session_id=app.storage.browser["id"],
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
    async with aiofiles.open(save_file, "ab") as f:
      await write_msgpack_record(f, save_data)

      name = self.metadata.get("maze", self.name)
      if imageSeenTime is not None and keydownTime is not None:
        stage_state = self.get_user_data("stage_state")
        if self.verbosity:
          logger.info(f"'{name}' saved file")
          logger.info(f"∆t: {time_diff(imageSeenTime, keydownTime) / 1000.0}")
          logger.info(f"stage state: {self.user_stats()}")
          logger.info(f"env step: {stage_state.nsteps}")

      else:
        if not self.ignore_missing_data:
          logger.error(f"'{name}' saved file")
          logger.error(f"stage state: {self.user_stats()}")
          logger.error(f"imageSeenTime={imageSeenTime}, keydownTime={keydownTime}")
          await self.set_user_data(finished=True, final_save=True)
          logger.info("Stage finished due to missing data")

    await self.set_user_data(saved_data=True)
    if self.verbosity:
      logger.info("finished saving")

  @retry_with_exponential_backoff(max_retries=5, base_delay=1, max_delay=10)
  async def finish_stage(self):
    if not self.get_user_data("started", False):
      return
    if self.get_user_data("finished", False):
      return

    # Wait for any pending saves to complete
    await self.get_user_queue().join()

    # save experiment data so far (prior time-step + resultant action)
    # if finished, save synchronously (to avoid race condition) with next stage
    await self.set_user_data(finished=True, final_save=True)
    imageSeenTime = await ui.run_javascript("getImageSeenTime()", timeout=10)

    start_notification = self.pop_user_data("start_notification")
    if start_notification:
      start_notification.dismiss()
    success_notification = self.pop_user_data("success_notification")
    if success_notification:
      success_notification.dismiss()

    stage_state = self.get_user_data("stage_state")
    await self.save_experiment_data(
      args=dict(
        key="timer",
        keydownTime=imageSeenTime,
        imageSeenTime=imageSeenTime,
      ),
      timestep=stage_state.timestep,
      user_stats=self.user_stats(),
    )
    logger.info(f"finished stage '{self.name}'. stats: {self.user_stats()}")

  async def handle_key_press(self, event, container):
    # Get or create lock for this specific user
    # async with self.get_user_lock():
    #  await self._handle_key_press(event, container)

    # async def _handle_key_press(self, event, container):
    key = event.args["key"]
    if self.verbosity:
      logger.info(f"handle_key_press key: {key}")
    if not self.get_user_data("started", False):
      if self.verbosity:
        logger.info(f"pressed '{key}' before started")
      return
    if self.get_user_data("stage_finished", False):
      # if already did final save, just return
      if self.get_user_data("final_save", False):
        if self.verbosity:
          logger.info(f"pressed '{key}' after final save")
        return

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
      logger.info("finishing handle key press")
      return

    # check if valid environment interaction
    if key not in self.key_to_action:
      if self.verbosity:
        logger.info(f"key press '{key}' not in key_to_action")
      return

    #############################
    # get prior timestep information and save experiment data
    # e.g. key presses
    #############################
    # asynchonously save experiment data by putting in a save queue
    # save prior timestep + current event information
    user_stats = self.user_stats()
    timestep = self.get_user_data("stage_state").timestep
    processed_timestep = self.preprocess_timestep(timestep)
    async with self.get_user_lock():
      await self.get_user_queue().put((event.args, processed_timestep, user_stats))
    asyncio.create_task(self._process_save_queue())

    #############################
    # automatically reset on done if flag is set
    #############################
    if self.autoreset_on_done:
      if timestep.last():
        rng = new_rng()
        timestep = self.web_env.reset(rng, self.env_params)
      else:
        action_idx = self.key_to_action[key]
        next_timesteps = self.get_user_data("next_timesteps")
        timestep = jax_tree_map(lambda t: t[action_idx], next_timesteps)
    else:
      # use action to select from avaialble next time-steps
      action_idx = self.key_to_action[key]
      next_timesteps = self.get_user_data("next_timesteps")
      timestep = jax_tree_map(lambda t: t[action_idx], next_timesteps)

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
    stage_state = stage_state.replace(
      timestep=timestep,
      nsteps=stage_state.nsteps + 1,
      nepisodes=stage_state.nepisodes + timestep.first(),
      nsuccesses=stage_state.nsuccesses + success,
    )

    # asynchronously save stage state
    asyncio.create_task(save_stage_state(stage_state))
    await self.set_user_data(stage_state=stage_state)

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
      await self.wait_for_start(container)
    await self.step_and_send_timestep(
      container,
      # image is normally updated client-side
      # when episode resets, update server-side
      update_display=episode_reset,
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

  async def handle_button_press(self, container):
    pass  # do nothing



@dataclasses.dataclass
class MultiAgentEnvStage(EnvStage):
    """A stage class for handling interactive multi-agent environment episodes.
    
    This class extends EnvStage to support environments with multiple agents,
    where one agent is controlled by the user and the other by a model.
    
    Additional Args:
        web_env (MultiAgentJaxWebEnv): The multi-agent environment instance.
        model (Any): Optional model to control the non-human agent.
        model_params (Any): Parameters for the model.
        num_seeds (int): Number of model parameter seeds available.
        init_hidden_state_fn (Any): Function to initialize the model's hidden state.
        human_id (Optional[int]): ID of the human-controlled agent (0 or 1).
        using_param_stack (bool): Whether model_params is a stack of parameters.
    """
    
    web_env: MultiAgentJaxWebEnv = None
    max_timesteps: Optional[int] = 100
    model: Any = None
    model_params: Any = None
    num_seeds: int = 6
    init_hidden_state_fn: Any = None
    human_id: Optional[int] = None  # is agent 0 or agent 1
    using_param_stack: bool = False
    check_finished: Optional[TIMESTEP_CALL_FN] = None

    def __post_init__(self):
        super().__post_init__()
        
        if self.check_finished is None:
            self.check_finished = lambda timestep: timestep.last()

    async def step_and_send_timestep(
            self,
            container,
            update_display: bool = True):
        #############################
        # get next images and store them client-side
        # setup code to display next state
        #############################
        rng = new_rng()
        timestep = self.get_user_data("stage_state").timestep
        
        # Get the other agent's action using the model if available
        if self.model is not None:
            all_obs = self.web_env.env._env.get_obs(timestep.state)
            human_id = self.get_user_data('human_id')
            if human_id == 0:
                other_agent_obs = all_obs['agent_1']
            else:
                other_agent_obs = all_obs['agent_0']
            other_agent_obs = other_agent_obs.flatten()
            other_agent_obs = other_agent_obs[None] # add environment dimension
            other_agent_obs = other_agent_obs[None] # add batch dimension
            agent_pos = timestep.state.agent_pos.astype(jnp.int32)
            agent_pos = agent_pos[None] # add environment dimension
            agent_pos = agent_pos[None] # add batch dimension

            prev_was_last = timestep.last()
            sim_done = prev_was_last.astype(jnp.int32).reshape((1,1))

            ac_input = (other_agent_obs, sim_done, agent_pos)
            hidden_state = self.get_user_data('hidden_state')
            model_index = self.get_user_data('model_index')  # if model param is a stack of params we only want one set
            if self.using_param_stack:
                model_params = jax.tree_map(lambda x: x[model_index], self.model_params)
            else:
                model_params = self.model_params
            action_res = self.model.apply(model_params, hidden_state, ac_input)
            hidden_state, pi = action_res[0], action_res[1]
            await self.set_user_data(hidden_state=hidden_state)
            other_agent_action = jnp.argmax(pi.probs, 2)[0]  # get max prob action
            other_agent_action = other_agent_action.squeeze().astype(jnp.int32)
        else:
            other_agent_action = 4  # Default behavior if no model exists

        # Get next timesteps with the other agent's action
        next_timesteps = self.web_env.next_steps(
            rng, timestep, self.env_params, 
            other_action=other_agent_action, 
            h_id=self.get_user_data('human_id'))
            
        next_images = self.vmap_render_fn(next_timesteps)
        next_images = {
            self.action_keys[idx]: base64_npimage(image) 
            for idx, image in enumerate(next_images)
        }

        js_code = f"window.next_states = {next_images};"
        ui.run_javascript(js_code)

        await self.set_user_data(next_timesteps=next_timesteps)
        
        # Display image
        if update_display:
            await self.display_timestep(container, timestep)
        else:
            ui.run_javascript("window.imageSeenTime = window.next_imageSeenTime;", timeout=10)

    async def activate(self, container: ui.element):
        """Initialize the stage and set up the human agent ID if not specified."""
        async with self.get_user_lock():
            # Randomly assign human ID if not specified
            rng = new_rng()
            if self.human_id is None:
                human_id = jax.random.randint(rng, (), 0, 2)
                human_color = 'Red' if human_id == 0 else 'Blue'
                await self.set_user_data(human_id=human_id)
                await self.set_user_data(human_color=human_color)
                rng = new_rng()
            
            # Initialize model if available
            if self.model is not None:
                hidden_state = self.init_hidden_state_fn()
                await self.set_user_data(hidden_state=hidden_state)
                
                # Sample model index
                model_index = np.random.randint(0, self.num_seeds)
                await self.set_user_data(model_index=model_index)
            
            # Continue with standard activation
            await super().activate(container)


@dataclasses.dataclass
class Block(Container):
  stages: List[Stage] = dataclasses.field(default_factory=list)
  randomize: Union[bool, List[bool]] = False
  metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
  name: str = None

  def __len__(self):
    return len(self.stages)

  def __post_init__(self):
    self._lock = Lock()  # Add lock for thread safety
    self._user_locks = {}  # Dictionary to store per-user locks
    if self.name is None:
      self.name = f"block_{uuid.uuid4().hex}"

    self.unique_id = f"block_{uuid.uuid4().hex}"
    # Broadcast metadata
    self.metadata["name"] = self.name
    for stage in self.stages:
      stage.metadata["block_metadata"] = self.metadata

  # def get_block_data(self):
  #  return app.storage.user.get(f"'{self.name}'_data", {})

  # def get_user_data(self, key, value=None):
  #  return self.get_block_data().get(key, value)

  # async def set_user_data(self, **kwargs):
  #  block_data = self.get_block_data()
  #  block_data.update(kwargs)
  #  async with self._lock:
  #    app.storage.user[f"'{self.name}'_data"] = block_data

  async def get_block_stage_idx(self):
    stage_idx = self.get_user_data("stage_idx")
    if stage_idx is None:
      stage_idx = 0
      await self.set_user_data(stage_idx=stage_idx)
    return stage_idx

  async def get_user_stage_order(self):
    """This function prepares the order of stages in the block."""
    if not self.randomize:
      return list(range(len(self.stages)))
    stage_order = self.get_user_data("stage_order")
    if stage_order is not None:
      return stage_order

    # self.randomize = randomize = [False, False, False, True, True]
    indices = jnp.arange(len(self.stages))
    mask = jnp.array(self.randomize)

    # Get randomizable indices
    random_indices = indices[mask]

    # Permute the randomizable indices
    rng_key = new_rng()
    rng_key, subkey = jax.random.split(rng_key)
    random_indices = jax.random.permutation(subkey, random_indices)

    # Combine back together
    permuted = indices.at[mask].set(random_indices)

    stage_order = [int(i) for i in permuted]
    await self.set_user_data(stage_order=stage_order)
    return stage_order

  async def get_stage(self):
    stage_idx = await self.get_block_stage_idx()
    stage_order = await self.get_user_stage_order()
    if stage_idx >= len(stage_order):
      logger.info("Defaulting to final stage")
      stage_idx = len(stage_order) - 1
    stage = self.stages[stage_order[stage_idx]]
    app.storage.user["stage_name"] = stage.name
    return stage

  async def advance_stage(self):
    stage_idx = await self.get_block_stage_idx()
    await self.set_user_data(stage_idx=stage_idx + 1)

  async def not_finished(self):
    stage_idx = await self.get_block_stage_idx()
    return stage_idx < len(self.stages)


def broadcast_metadata(blocks: List[Block]) -> List[Stage]:
  """This function assigns the block metadata to each stage."""
  # assign block description to each stage description
  for block_idx, block in enumerate(blocks):
    for stage in block.stages:
      block.metadata.update(idx=block_idx)
      stage.metadata["block_metadata"] = block.metadata


def nstages(blocks: List[Block]) -> int:
  return sum(len(block.stages) for block in blocks)


def prepare_blocks(blocks: List[Block]) -> List[Stage]:
  """This function assigns the block metadata to each stage.
  It also flattens all blocks into a single list of stages.
  """
  # assign block description to each stage description
  all_stages = []
  for block_idx, block in enumerate(blocks):
    for stage in block.stages:
      block.metadata.update(idx=block_idx)
      stage.metadata["block_metadata"] = block.metadata
      all_stages.append(stage)
  return all_stages


def generate_stage_order(
  blocks: List[Block], block_order: List[int], rng_key: jnp.ndarray
) -> List[int]:
  """This function generates the order in which the stages should be displayed.
  It takes the blocks and the block order as input and returns the stage order.

  It also randomizes the order of the stages within each block if the block's randomize flag is True.
  """
  # Assign unique indices to each stage in each block
  block_indices = {}
  current_index = 0
  for block_idx, block in enumerate(blocks):
    block_indices[block_idx] = list(
      range(current_index, current_index + len(block.stages))
    )
    current_index += len(block.stages)

  # Generate the final stage order based on block_order
  stage_order = []
  for block_idx in block_order:
    block = blocks[block_idx]
    block_stage_indices = block_indices[block_idx]

    if block.randomize:
      if isinstance(block.randomize, bool):
        randomize = [True] * len(block.stages)
      else:
        randomize = block.randomize

      indices = jnp.arange(len(block.stages))
      mask = jnp.array(randomize)

      # Get randomizable indices
      random_indices = indices[mask]

      # Permute the randomizable indices
      rng_key, subkey = jax.random.split(rng_key)
      random_indices = jax.random.permutation(subkey, random_indices)

      # Combine back together
      permuted = indices.at[mask].set(random_indices)

      block_stage_indices = [block_stage_indices[i] for i in permuted]

    stage_order.extend(block_stage_indices)

  return stage_order
