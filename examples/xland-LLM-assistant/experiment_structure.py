from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui
import asyncio
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl.stages import Stage, EnvStage, FeedbackStage, LLMEnvStage
from nicewebrl import get_logger
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from xminigrid.rendering.text_render import _text_encode_rule, _encode_tile



logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


class PlaygroundTimestepWrapper(TimestepWrapper):
  def reset(self, key: jax.random.PRNGKey, params=None):
    timestep = self._env.reset(key=key, params=params)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)

  def step(self, key, state, action, params=None):
    if isinstance(state, TimeStep):
      state = state.state
    timestep = self._env.step(params=params, timestep=state, action=action)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)

########################################
# Define actions and corresponding keys
########################################
class Actions(IntEnum):
  forward = 0
  right = 1
  left = 2
  pick_up = 3
  put_down = 4
  toggle = 5


# Only first 3 actions are actually used
actions = jnp.array([0, 1, 2, 3, 4, 5])
action_keys = ["ArrowUp", "ArrowRight", "ArrowLeft", "p", "d", "t"]  # Mapping to keys
action_to_name = ["Forward", "Right", "Left", "Pick Up", "Drop", "Toggle"]


########################################
# Create multiple environments with different rulesets
########################################
def text_encode_goal(goal: list[int]) -> str:
  # copied and edited from: https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/rendering/text_render.py#L140
  goal_id = goal[0]
  if goal_id == 1:
    return f"Agent_Hold({_encode_tile(goal[1:3])})"
  elif goal_id == 3:
    return f"Agent_Near({_encode_tile(goal[1:3])})"
  elif goal_id == 4:
    return f"Tile_Near({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 7:
    return f"Tile_Near_Up_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 8:
    return f"Tile_Near_Right_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 9:
    return f"Tile_Near_Down_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 10:
    return f"Tile_Near_Left_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 11:
    return f"Agent_Near_Up_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 12:
    return f"Agent_Near_Right_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 13:
    return f"Agent_Near_Down_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 14:
    return f"Agent_Near_Left_Goal({_encode_tile(goal[1:3])})"
  else:
    raise RuntimeError(f"Rendering: Unknown goal id: {goal_id}")


def describe_ruleset(ruleset) -> str:
  str = "GOAL:" + "\n"
  goal = text_encode_goal(ruleset.goal.tolist())
  goal.split()
  str += text_encode_goal(ruleset.goal.tolist()) + "\n"
  str += "\n"
  str += "RULES:" + "\n"
  for rule in ruleset.rules.tolist():
    if rule[0] != 0:
      str += _text_encode_rule(rule) + "\n"
  str += "\n"
  str += "INIT TILES:" + "\n"
  for tile in ruleset.init_tiles.tolist():
    if tile[0] != 0:
      str += _encode_tile(tile) + "\n"

  return str


def create_env_with_ruleset(ruleset_key):
  env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
  benchmark = xminigrid.load_benchmark(name="trivial-1m")
  rule = benchmark.sample_ruleset(jax.random.key(ruleset_key))
  rule_text = describe_ruleset(rule)

  env_params = env_params.replace(
    ruleset=rule,
    max_steps=50,
    view_size=11,
  )
  env = GymAutoResetWrapper(env)
  env = RGBImgObservationWrapper(env)
  return env, benchmark, env_params, rule_text


num_envs = 3

# Create 5 different environments
env, benchmark, env_params, rule_text = create_env_with_ruleset(0)
jax_env = PlaygroundTimestepWrapper(env, autoreset=True, use_params=True)
jax_web_env = JaxWebEnv(
  env=jax_env,
  actions=actions,
)
jax_web_env.precompile(dummy_env_params=env_params)


def render_fn(timestep: nicewebrl.TimeStep):
  return timestep.observation.astype(jnp.uint8)
render_fn = jax.jit(render_fn)

vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, env_params)


########################################
# Define Instruction Stage
########################################

all_stages = []

async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """Press the arrows keys to move the agent 
      
      Press:
      - p to pick up an object
      - d to drop an object
      - t to transform an object"""
    )

instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)

########################################
# Define Environment Stages
########################################

def make_image_html(src):
  html = f"""
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="width: 400px; height: 400px; object-fit: contain;">
    </div>
    """
  return html


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
  rendered_img = stage.render_fn(timestep)
  new_obs_base64 = base64_npimage(rendered_img)

  stage_state = stage.get_user_data("stage_state")
  rule = benchmark.sample_ruleset(nicewebrl.new_rng())
  current_rule_text = describe_ruleset(rule)
  await stage.set_user_data(rule_text=current_rule_text)

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )
    ui.markdown("""
    You have  50 steps to figure out and complete the task. You can ask the AI for help.
    
    Red exclamations indicate out of bounds.
    """)
    ui.html(make_image_html(src=new_obs_base64))


def evaluate_success_fn(timestep: TimeStep, params: Optional[object] = None):
  return timestep.last() and timestep.reward > 0


# Create 5 different stages
env_stages = []
for i in range(num_envs):
  # Get the rule_text for the current environment
  # envs_and_params_and_ruletext is a list of (env, env_params, rule_text)

  environment_stage = EnvStage(
    name=f"Environment {i + 1}",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    msg_display_time=2,
    metadata=dict(
      desc=f"XLand environment {i + 1}",
      stage_number=i + 1,
    ),
  )
  env_stages.append(environment_stage)

########################################
# Define Feedback Stage
########################################

async def feedback_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown("Please answer the following questions:")

    questions = [
      "How helpful was the AI?",
      "How human-like was the AI?",
    ]

    answers = {}
    completed_all = asyncio.Event()

    # record answer and see if finished
    def recoder_answer(question, answer):
      answers[question] = answer
      if all((i is not None for i in answers.values())):
        completed_all.set()

    # make handler factory for each question
    def make_handler(question):
      return lambda e: recoder_answer(question, e.value)

    # make radio buttons for each question
    for i, q in enumerate(questions):
      with ui.row():
        ui.label(q)
        ui.radio(
          [1, 2, 3, 4, 5], on_change=make_handler(q)
          ).props('inline')
        answers[q] = None
    await completed_all.wait()
    return answers

feedback_stage = FeedbackStage(
  name="Feedback",
  display_fn=feedback_display_fn,
)

llm_stage = LLMEnvStage(
    name="LLM Chat",
    title="Ask the AI",
    body="Ask follow-up questions about the task.",
    # --- required EnvStage params ---
    web_env=jax_web_env,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    action_keys=action_keys,
    action_to_name=action_to_name,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    # --- stage UI / behavior ---
    display_fn=env_stage_display_fn,
    verbosity=VERBOSITY,
    msg_display_time=2,
)

########################################
# Define Experiment
########################################
all_stages = [
  instruction_stage,
  *env_stages,
  feedback_stage,
  llm_stage]
experiment = nicewebrl.SimpleExperiment(
  stages=all_stages,
  randomize=[False] + [True] * (len(all_stages) - 1),
)
