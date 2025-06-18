import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.module import nowrap
import flax.struct as struct
from flax.training.train_state import TrainState
import optax
import flashbax as fbx
import functools
import numpy as np
from typing import Optional, Any
from craftax.craftax_env import make_craftax_env_from_name
from nicewebrl.nicejax import TimestepWrapper
import rlax
import matplotlib.pyplot as plt
import wandb

# Assume gymnax environment imports

# --- Dataclasses ---

# for single env, single timestep
class StepType(jnp.uint8):
  FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
  MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
  LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


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

@struct.dataclass
class Transition:
    # Structure stored in the replay buffer
    timestep: TimeStep      # Information at time t (s_t, r_t, done_t)
    action: jax.Array       # Action taken at time t (a_t)
    agent_state: jax.Array # Agent's RNN state *before* processing timestep t (h_{t-1})

class CustomTrainState(TrainState):
    target_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

@struct.dataclass
class RunnerState():
    train_state: CustomTrainState  # Contains network params, optimizer state, etc.
    env_timestep: TimeStep         # Current environment state
    agent_state: jax.Array        # Current RNN state of the agent
    buffer_state: Any              # State of the replay buffer
    rng: jax.Array                # Random number generator key

@struct.dataclass
class Predictions:
    q_vals: jax.Array
    state: struct.PyTreeNode
    # Optionally store other network outputs if needed

@struct.dataclass
class SimulationOutput:
  actions: jax.Array
  predictions: Optional[Predictions] = None


# --- Helper Functions ---

def make_float(x):
    return x.astype(jnp.float32)

def add_time(v):
    return jax.tree.map(lambda x: x[None], v)

def repeat(x, N: int):
  def identity(y, unused):
    return y

  return jax.vmap(identity, (None, 0), 0)(x, jnp.arange(N))

def is_truncated(timestep):
    non_terminal = timestep.discount

    # either termination or truncation
    is_last = make_float(timestep.last())

    # non_terminal AND is_last confirms truncation
    truncated = (non_terminal + is_last) > 1
    return make_float(1 - truncated)

def simulation_finished_mask(initial_mask, next_timesteps):
  # get mask
  non_terminal = next_timesteps.discount[1:]
  # either termination or truncation
  is_last_t = make_float(next_timesteps.last()[1:])

  # time-step of termination and everything afterwards is masked out
  term_cumsum_t = jnp.cumsum(is_last_t, 0)
  loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)
  return concat_start_sims(initial_mask, loss_mask_t)

def concat_pytrees(tree1, tree2, **kwargs):
    return jax.tree.map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)

def concat_first_rest(first, rest):
    first = add_time(first)  # [N, ...] --> [1, N, ...]
    # rest: [T, N, ...]
    # output: [T+1, N, ...]
    return jax.vmap(concat_pytrees, 1, 1)(first, rest)

def concat_start_sims(start, simulations):
    # concat where vmap over simulation dimension
    # need this since have 1 start path, but multiple simulations
    concat_ = lambda a, b: jnp.concatenate((a, b))
    concat_ = jax.vmap(concat_, (None, 1), 1)
    return jax.tree.map(concat_, start, simulations)

# --- Policy helpers ---
def epsilon_greedy_act(q, eps, key):
    key_a, key_e = jax.random.split(key, 2)
    greedy_actions = jnp.argmax(q, axis=-1)
    random_actions = jax.random.randint(
        key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]
    )
    pick_random = (
        jax.random.uniform(key_e, greedy_actions.shape) < eps
    )
    chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
    return chosen_actions

class FixedEpsilonGreedy:
  """Epsilon Greedy action selection"""

  def __init__(self, epsilons: float):
    self.epsilons = epsilons

  @functools.partial(jax.jit, static_argnums=0)
  def choose_actions(self, q_vals: jnp.ndarray, rng: jax.random.PRNGKey):
    rng = jax.random.split(rng, q_vals.shape[0])
    return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)

# --- Loss Function Helpers ---
def q_learning_lambda_target(
    q_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    is_last_t: jax.Array,
    a_t: jax.Array,
    lambda_: jax.Array,
    stop_target_gradients: bool = True,
) -> jax.Array:
    """MINOR change to rlax.lambda_returns to incorporate is_last_t.

    lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t).
                                            ONLY CHANGE:^
    """
    v_t = rlax.batched_index(q_t, a_t)
    lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)
    target_tm1 = rlax.lambda_returns(
        r_t, discount_t, v_t, lambda_, stop_target_gradients=stop_target_gradients
    )
    return target_tm1


def q_learning_lambda_td(
    q_tm1: jax.Array,
    a_tm1: jax.Array,
    target_q_t: jax.Array,
    a_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    is_last_t: jax.Array,
    lambda_: jax.Array,
    stop_target_gradients: bool = True,
    tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
):
    """Essentially the same as rlax.q_lambda except we use selector actions on q-values, not average. This makes it like Q-learning.

    Other difference is is_last_t is here.
    """

    # Apply signed hyperbolic transform to Q-values
    q_tm1_transformed = tx_pair.apply(q_tm1)
    target_q_t_transformed = tx_pair.apply(target_q_t)

    v_tm1 = rlax.batched_index(q_tm1_transformed, a_tm1)
    target_mt1 = q_learning_lambda_target(
        r_t=r_t,
        q_t=target_q_t_transformed,
        a_t=a_t,
        discount_t=discount_t,
        is_last_t=is_last_t,
        lambda_=lambda_,
        stop_target_gradients=stop_target_gradients,
    )

    v_tm1, target_mt1 = tx_pair.apply_inv(v_tm1), tx_pair.apply_inv(target_mt1)

    return v_tm1, target_mt1

@functools.partial(jax.jit, static_argnums=(1,))
def rolling_window(a, size: int):
    """Create rolling windows of a specified size from an input array.

    Rolls over the first dimension only, preserving other dimensions.

    Args:
        a (array-like): The input array of shape [T, ...]
        size (int): The size of the rolling window

    Returns:
        Array of shape [T-size+1, size, ...]
    """
    # Get shape info
    T = a.shape[0]  # length of first dimension
    remaining_dims = a.shape[1:]  # all other dimensions

    # Create start indices for the first dimension only
    starts = jnp.arange(T - size + 1)

    # Create slice for each start index, preserving other dimensions
    def slice_at(start):
        idx = (start,) + (0,) * len(remaining_dims)  # index tuple for all dims
        size_tuple = (size,) + remaining_dims  # size tuple for all dims
        return jax.lax.dynamic_slice(a, idx, size_tuple)

    return jax.vmap(slice_at)(starts)

# Simulate n trajectories for a given agent and policy
def simulate_n_trajectories(
  h_tm1: jax.Array,
  x_t: TimeStep,
  rng: jax.random.PRNGKey,
  agent: nn.Module,
  params: jax.Array,
  policy_fn: callable = None,
  num_steps: int = 5,
  num_simulations: int = 5,
):
    """
    Simulates multiple trajectories starting from a given state and RNN hidden state.
    
    This function first replicates the initial state and RNN state num_simulations times,
    then applies the agent's policy to generate actions and simulate forward for num_steps.
    Each simulation uses a different epsilon value for exploration (one greedy, others epsilon-greedy).
    
    Args:
        h_tm1 (RnnState): Initial RNN hidden state [D]
        x_t (TimeStep): Initial environment state containing (observation, discount, etc.)
        rng (jax.random.PRNGKey): Random key for simulation
        agent (nn.Module): Agent module containing apply() and apply_world_model() methods
        params (Params): Parameters for the agent
        policy_fn (SimPolicy, optional): Policy function that takes (predictions, rng) and returns actions
        num_steps (int, optional): Number of forward simulation steps. Defaults to 5.
        num_simulations (int, optional): Number of parallel simulations to run. Defaults to 5.
    
    Returns:
        Tuple[TimeStep, SimulationOutput]:
            - all_timesteps: Environment states for all timesteps [T+1, N, ...] where T is num_steps and N is num_simulations
            - sim_outputs: SimulationOutput containing:
                - actions: Selected actions for all timesteps [T+1, N, ...]
                - predictions: Agent predictions for all timesteps [T+1, N, ...]
    """

    def initial_predictions(x, h_tm1):
        # roll through RNN
        # Format inputs for apply fn
        obs, discount = x.observation, x.discount
        obs = obs[np.newaxis, :]
        discount = discount[np.newaxis, :]
        resets = 1.0 - discount
        inputs = (obs, resets)
        h_t, preds = agent.apply(params, h_tm1, inputs)
        
        # remove time dim
        preds = jax.tree.map(lambda p: p.squeeze(0), preds)

        return x, h_t, preds

    # by giving state as input and returning, will
    # return copies. 1 for each sampled action.
    rng, rng_ = jax.random.split(rng)

    # one for each simulation
    # [N, ...]
    # replace (x_t, task) with N-copies
    x_t = jax.tree.map(lambda x: jnp.repeat(x[None], num_simulations, axis=0), x_t)
    h_tm1 = jax.tree.map(lambda x: jnp.repeat(x[None], num_simulations, axis=0), h_tm1)

    x_t, h_t, preds_t = initial_predictions(x_t, h_tm1)

    # choose epsilon-greedy action
    a_t = policy_fn(preds_t.q_vals, rng_)

    def _single_model_step(carry, unused):
        (timestep, agent_state, a, rng) = carry

        ###########################
        # 1. use state + action to predict next state
        ###########################
        rng, rng_ = jax.random.split(rng)

        # print("timestep:", jax.tree.map(lambda x: x.shape, timestep))
        # print("agent_state:", agent_state.shape)
        # print("action:", a.shape)

        # apply world model to get next timestep
        next_timestep = agent.apply_world_model(timestep, a, rng_)

        # Format inputs for apply fn
        obs, discount = next_timestep.observation, next_timestep.discount
        obs = obs[np.newaxis, :]
        discount = discount[np.newaxis, :]
        resets = 1.0 - discount
        x = (obs, resets)

        # get next agent state and actions
        next_agent_state, next_preds = agent.apply(params, agent_state, x)

        # remove time dim
        next_preds = jax.tree.map(lambda x: x.squeeze(0), next_preds)

        # get next actions
        next_a = policy_fn(next_preds.q_vals, rng_)

        # format outputs
        carry = (next_timestep, next_agent_state, next_a, rng)
        sim_output = SimulationOutput(
            predictions=next_preds,
            actions=next_a,
        )
        return carry, (next_timestep, sim_output)

    ################
    # get simulation outputs
    ################
    initial_carry = (x_t, h_t, a_t, rng)
    _, (next_timesteps, sim_outputs) = jax.lax.scan(
        f=_single_model_step, init=initial_carry, xs=None, length=num_steps
    )

    # sim_outputs.predictions: [T, N, ...]
    # concat [1, ...] with [N, T, ...]
    sim_outputs = SimulationOutput(
        predictions=concat_first_rest(preds_t, sim_outputs.predictions),
        actions=concat_first_rest(a_t, sim_outputs.actions),
    )
    all_timesteps = concat_first_rest(x_t, next_timesteps)
    return all_timesteps, sim_outputs

# --- Logger Definition ---
# TODO: Add episode logging with images
class Logger:
    def metrics_logger(self, train_state, metrics):
        """Log scalar metrics (e.g., q_loss, td_error, reward)."""
        key = "metrics"
        def callback(ts, m):
            m.update(
                {
                    f"{key}/num_actor_steps": ts.timesteps,
                    f"{key}/num_learner_updates": ts.n_updates,
                }
            )
            if wandb.run is not None:
                wandb.log(m)
        jax.debug.callback(callback, train_state, metrics)
    
    def gradient_logger(self, train_state, grads):
        key = "gradients"
        gradient_metrics = {
            f"{key}/{k}_norm": optax.global_norm(v) for k, v in grads.items()
        }

        def callback(ts, g):
            g.update(
                {
                f"{key}/num_actor_steps": ts.timesteps,
                f"{key}/num_learner_updates": ts.n_updates,
                }
            )
            if wandb.run is not None:
                wandb.log(g)

        jax.debug.callback(callback, train_state, gradient_metrics)

    def extra_logger(self, log_info):
        """Log diagnostic visuals (e.g., q-curves, trajectories) periodically."""
        def callback(log_info):
            # [B, Env_time, Sim_time, Num_sim] -> [Sim_time]
            data = jax.tree.map(lambda x: x[0, 0, :, 0], log_info["dyna"])
            self._log_trajectory_images(data, tag="dyna")

        jax.debug.callback(callback, log_info)


    def _log_trajectory_images(self, data, tag):
        """Make matplotlib visualizations from per-timestep arrays and log to WandB."""
        rewards = data["timesteps"].reward
        actions = data["actions"]
        td_errors = data["td_errors"]
        q_loss = data["q_loss"]
        q_values = data["q_values"]
        q_target = data["q_target"]
        discounts = data["timesteps"].discount
        is_last = data["timesteps"].last()
        loss_mask = data["loss_mask"]

        width = max(10, int(0.3 * len(rewards)))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(width, 20))

        ax1.plot(rewards, label="Reward")
        ax1.plot(rlax.batched_index(q_values, actions), label="Q")
        ax1.plot(q_target, label="Q Target")
        ax1.set_title("Rewards and Q-Values"); ax1.legend(); ax1.grid()

        ax2.plot(td_errors); ax2.set_title("TD Error"); ax2.grid()
        ax3.plot(q_loss); ax3.set_title("Q Loss"); ax3.grid()
        ax4.plot(discounts, label="γ")
        ax4.plot(loss_mask, label="Mask")
        ax4.plot(is_last, label="is_last")
        ax4.set_title("Episode Markers"); ax4.legend(); ax4.grid()

        if wandb.run is not None:
            wandb.log({f"learner_image/{tag}": wandb.Image(fig)})
        plt.close(fig)

# --- Network Definition ---

# MLP head
class MLP(nn.Module):
    hidden_dim: int
    out_dim: Optional[int] = None
    num_layers: int = 1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(x)
            x = jax.nn.leaky_relu(x)

        x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=self.use_bias)(x)
        return x

# Agent
class DynaAgent(nn.Module):
    config: dict
    env: TimestepWrapper
    env_params: Any

    def setup(self):
        self.encoder = MLP(
            hidden_dim=self.config["ENCODER_HIDDEN_DIM"],
            num_layers=self.config["NUM_ENCODER_LAYERS"],
            use_bias=self.config["USE_BIAS"],
            name="encoder_mlp",
        )
        self.q_head = MLP(
            hidden_dim=self.config["Q_HIDDEN_DIM"],
            out_dim=self.env.action_space(self.env_params).n,
            num_layers=self.config["NUM_Q_LAYERS"],
            use_bias=self.config["USE_BIAS"],
            name="q_head_mlp",
        )
        self.rnn = nn.GRUCell(
            features=self.config["RNN_HIDDEN_DIM"],
            name="gru_cell"
        )

        # Cache config values for use during scan (avoid dict access in traced code)
        self.hidden_size = self.config["RNN_HIDDEN_DIM"]
        self.num_envs = self.config["NUM_ENVS"]

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        """
        carry: GRU hidden state [batch, hidden_size]
        x: tuple of (obs, reset flags)
           obs: [T, B, obs_dim...], resets: [T, B]
        """
        rnn_state = carry
        obs, resets = x  # each [batch, ...]

        # Reinitialize RNN state for environments that have reset
        rnn_state = jnp.where(
            resets[:, None],  # [batch, 1]
            self.initialize_carry(resets.shape[0], self.hidden_size),  # [batch, hidden]
            rnn_state
        )

        embeds = self.encoder(obs)  # [batch, embedding_dim]
        next_rnn_state, rnn_out = self.rnn(rnn_state, embeds)  # both [batch, hidden]
        q_vals = self.q_head(rnn_out)  # [batch, num_actions]

        preds = Predictions(q_vals=q_vals, state=next_rnn_state)
        return next_rnn_state, preds
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))

    def apply_world_model(self, ts: struct.PyTreeNode, action: jax.Array, rng: jax.Array) -> struct.PyTreeNode:
        """
        Simulates one step using the 'world model' (ground truth env).
        This wraps the true `env.step` function.
        """
        
        vmap_step = lambda num_envs: lambda rng, timestep, action: jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng, num_envs), timestep, action, self.env_params)

        next_timestep = vmap_step(action.shape[0])(rng, ts, action)
        return next_timestep


# --- Loss Function ---

@struct.dataclass
class DynaLossFn:
    agent: DynaAgent
    config: dict # Containing GAMMA, TD_LAMBDA, ONLINE_COEFF, DYNA_COEFF, SIM_LENGTH, etc.
    simulation_policy: callable # Function: (q_vals, rng) -> actionp

    def batch_loss(
        self,
        timestep,       # Includes non_terminal etc. [T+1, B, ...]
        online_preds,   # Includes q_vals [T+1, B, A]
        target_preds,   # Includes q_vals [T+1, B, A]
        actions,        # [T+1, B]
        rewards,        # [T+1, B]
        non_terminal,   # [T+1, B]
        is_last,        # [T+1, B]
        loss_mask,      # [T+1, B]
    ):
        # 1. Prepare Inputs (similar to original)
        rewards = make_float(rewards)
        rewards = rewards - self.config["STEP_COST"]      # [T+1, B]
        is_last = make_float(is_last)
        discounts = non_terminal * self.config["GAMMA"] # [T+1, B]
        lambda_ = jnp.ones_like(non_terminal) * self.config["TD_LAMBDA"]
        selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)  # [T+1, B]

        # 2. Align Time Steps
        q_tm1 = online_preds.q_vals[:-1]      # Online Q(s_t, a) [T, B, A]
        a_tm1 = actions[:-1]                  # Action a_t [T, B]
        r_t = rewards[1:]                     # Reward r_{t+1} [T, B]
        discount_t = discounts[1:]            # Discount gamma_{t+1} [T, B]
        q_t_target = target_preds.q_vals[1:]  # Target Q(s_{t+1}, a') [T, B, A]
        selector_actions = selector_actions[1:] # Greedy actions [T, B]

        is_last = is_last[1:]                 # Is last t+1 [T, B]
        loss_mask = loss_mask[:-1]            # Valid transitions mask [T, B]
        non_terminal = non_terminal[1:]       # Non-terminal mask [T, B]
        lambda_ = lambda_[1:]                 # Lambda trimmed [T, B]
        
        # print("q_tm1:", q_tm1.shape)
        # print("a_tm1:", a_tm1.shape)
        # print("q_t_target:", q_t_target.shape)
        # print("selector_actions:", selector_actions.shape)
        # print("r_t:", r_t.shape)
        # print("discount_t:", discount_t.shape)
        # print("is_last:", is_last.shape)
        # print("lambda_:", lambda_.shape)

        # 3. Calculate TD Error
        # We map over the batch dimension (axis 1)
        # We vmap it to handle the batch dimension [T, B, ...] -> [T, B] output
        # Get N-step transformed TD error and loss.
        batch_td_error_fn = jax.vmap(q_learning_lambda_td, in_axes=1, out_axes=1)
        q_t, target_q_t = batch_td_error_fn(
            q_tm1,        # [T, B, A] -> processed as B sequences of [T, A]
            a_tm1,        # [T, B]   -> processed as B sequences of [T]
            q_t_target,   # [T, B, A] -> processed as B sequences of [T, A]
            selector_actions, # [T, B] -> processed as B sequences of [T]
            r_t,          # [T, B]   -> processed as B sequences of [T]
            discount_t,   # [T, B]   -> processed as B sequences of [T]
            is_last,      # [T, B]   -> processed as B sequences of [T]
            lambda_,      # [T, B]   -> processed as B sequences of [T]
        ) # Output shape: [T, B]

        # 4. Apply Mask
        # Zero out errors for invalid transitions
        target_q_t = target_q_t * non_terminal
        batch_td_error = target_q_t - q_t
        batch_td_error = batch_td_error * loss_mask

        # 5. Calculate Loss (Squared Error)
        batch_loss = 0.5 * jnp.square(batch_td_error) # [T, B]

        # 6. Calculate Mean Loss per Batch Item
        # Sum loss over time for each batch item, divide by num valid steps
        batch_loss_mean = batch_loss.sum(0) / loss_mask.sum(0) # [B]

        # 7. Calculate Metrics (similar to original)
        metrics = {
            "0.q_loss": batch_loss.mean(),
            "0.q_td": jnp.abs(batch_td_error).mean(),
            "1.reward": r_t.mean(),
            "z.q_mean": online_preds.q_vals.mean(),
            "z.q_var": online_preds.q_vals.var(),
        }

        log_info = {
            "timesteps": timestep, # [T, B, ...]
            "actions": actions, # [T, B, A]
            "td_errors": batch_td_error,  # [T, B]
            "loss_mask": loss_mask,  # [T, B]
            "q_values": online_preds.q_vals,  # [T, B]
            "q_loss": batch_loss,  # [T, B]
            "q_target": target_q_t, # [T, B]
        }

        return batch_td_error, batch_loss_mean, metrics, log_info

    def total_loss(
        self,
        online_params,
        target_params,
        batch: Transition,
        init_state: jax.Array,
        rng: jax.Array
    ) -> tuple[jax.Array, dict]:
        # Input: batch is sequence [B, T, ...], initial states are [B, ...]

        # --- 1. Online Loss Component ---
        # Unroll online & target Q-networks on the REAL batch data
        # final_online_state: [B, ...]
        # online_preds_real: [B, T, ...]
        # Swap time and batch dimensions
        batch = jax.tree.map(lambda x: x.swapaxes(0, 1), batch) # [T, B, ...]

        # Unpack batch data
        actions = batch.action # [T, B]
        timestep = batch.timestep # [T, B, ...]
        rewards = timestep.reward # [T, B]
        non_terminal = timestep.discount # [T, B]
        is_last = timestep.last() # [T, B]
        loss_mask = is_truncated(timestep) # [T, B]
        obs = timestep.observation # [T, B, ...]
    
        # Unroll online & target Q-networks on the REAL batch data
        resets = 1.0 - non_terminal # [T, B]
        xs = (obs, resets)
        _, online_preds = self.agent.apply(online_params, init_state, xs)
        _, target_preds = self.agent.apply(target_params, init_state, xs)

        # Calculate TD-Lambda loss L_online based on online_preds, target_preds, batch.action, batch.timestep.reward, batch.timestep.discount
        all_metrics = {}
        all_log_info = {}

        T, B = loss_mask.shape[:2]

        td_error, batch_loss, metrics, log_info = self.batch_loss(
            timestep=timestep,
            online_preds=online_preds,
            target_preds=target_preds,
            actions=actions,
            rewards=rewards,
            non_terminal=non_terminal,
            is_last=is_last,
            loss_mask=loss_mask,
        )

        # print("TD ERROR:", td_error.shape)
        # print("BATCH LOSS:", batch_loss.shape)
        # print("METRICS:", jax.tree.map(lambda x: x.shape, metrics))
        # print("LOG INFO:", jax.tree.map(lambda x: x.shape, log_info))

        # update L_online
        L_online = jnp.mean(batch_loss)

        # update metrics
        all_metrics.update({f"{k}/online": v for k, v in metrics.items()})
        all_log_info["online"] = log_info

        # zero pad TD error
        td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
        td_error = jnp.abs(td_error)

        # --- 2. Dyna Loss Component ---
        L_dyna = 0.0
        if self.config["DYNA_COEFF"] > 0:
            # Dyna Loss Calculation
            #   - Select starting points (s_t, h_t) from the real batch sequence (e.g., via windows or sampling)
            #   - For each starting point:
            #       - Call simulate_rollout(world_model, model_params, q_network, online_params, s_t, h_t, ...) -> simulated_trajectory
            #       - Combine real_prefix + simulated_trajectory -> combined_trajectory
            #       - Unroll online & target Q-networks on combined_trajectory (using h from start of real segment) -> Q_comb_on, Q_comb_tar
            #       - Calculate TD-Lambda loss L_sim on combined_trajectory (masking appropriately)
            #   - Average L_sim across starting points/simulations -> L_dyna
            # will use time-step + previous rnn-state to simulate
            # next state at each time-step and compute predictions

            remove_last = lambda x: jax.tree.map(lambda y: y[:-1], x)
            h_tm1_online = concat_first_rest(init_state, remove_last(online_preds.state))
            h_tm1_target = concat_first_rest(init_state, remove_last(target_preds.state))
            x_t = timestep

            dyna_loss_fn = functools.partial(
                self.dyna_loss_fn, online_params=online_params, target_params=target_params
            )

            # vmap over batch
            dyna_loss_fn = jax.vmap(dyna_loss_fn, (1, 1, 1, 1, 1, 0), 0)
            _, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
                x_t,
                actions,
                h_tm1_online,
                h_tm1_target,
                loss_mask,
                jax.random.split(rng, B),
            )
            L_dyna = jnp.mean(dyna_batch_loss)

            # update metrics with dyna metrics
            all_metrics.update({f"{k}/dyna": v for k, v in dyna_metrics.items()})

            all_log_info["dyna"] = dyna_log_info

        # --- 3. Combine Losses ---
        # TODO: Add importance sampling weights if using prioritized replay
        total_loss = self.config["ONLINE_COEFF"] * L_online + self.config["DYNA_COEFF"] * L_dyna

        return total_loss, (td_error, all_metrics, all_log_info)
    
    def dyna_loss_fn(
        self,
        timesteps: TimeStep,
        actions: jax.Array,
        h_online: jax.Array,
        h_target: jax.Array,
        loss_mask: jax.Array,
        rng: jax.random.PRNGKey,
        online_params,
        target_params,
    ):
        """

        Algorithm:
        -----------

        Args:
            x_t (TimeStep): [D], timestep at t
            h_online (jax.Array): [D], rnn-state at t-1
            h_target (jax.Array): [D], rnn-state at t-1 from target network
        """
        window_size = self.config["WINDOW_SIZE"]
        window_size = min(window_size, len(actions))
        window_size = max(window_size, 1)
        roll = functools.partial(rolling_window, size=window_size)
        simulate = functools.partial(
            simulate_n_trajectories,
            agent=self.agent,
            params=online_params,
            num_steps=self.config["SIMULATION_LENGTH"],
            num_simulations=self.config["NUM_SIMULATIONS"],
            policy_fn=self.simulation_policy,
        )

        # first do a rollowing window
        # W = T-window_size+1 = number of windows
        # T' = window_size
        # [T, ...] --> [W, T', ...]
        # actions = jax.tree.map(roll, actions)
        # timesteps = jax.tree.map(roll, timesteps)
        # h_online = jax.tree.map(roll, h_online)
        # h_target = jax.tree.map(roll, h_target)
        # loss_mask = jax.tree.map(roll, loss_mask)

        def _dyna_loss_fn(t, a, h_on, h_tar, l_mask, key):
            """
            Args:
                t (jax.Array): [window_size, ...]
                h_on (jax.Array): [window_size, ...]
                h_tar (jax.Array): [window_size, ...]
                key (jax.random.PRNGKey): [2]
            """
            # get simulations starting from final timestep in window
            key, key_ = jax.random.split(key)
            # [sim_length, num_sim, ...]
            next_t, sim_outputs_t = simulate(
                h_tm1=h_on,
                x_t=t,
                rng=key_,
            )

            # we replace last, because last action from data
            # is different than action from simulation
            # [window_size + sim_length, num_sims, ...]
            # all_but_last = lambda y: jax.tree.map(lambda x: x[:-1], y)
            # all_t = concat_start_sims(all_but_last(t), next_t)
            # all_a = concat_start_sims(all_but_last(a), sim_outputs_t.actions)
            all_t = next_t
            all_a = sim_outputs_t.actions

            # NOTE: we're recomputing RNN but easier to read this way...
            resets = 1.0 - all_t.discount
            xs = (all_t.observation, resets)

            # h_on_init = jax.tree.map(lambda x: x[0], h_on)
            # h_on_init = repeat(h_on_init, self.config["NUM_SIMULATIONS"])
            # h_tar_init = jax.tree.map(lambda x: x[0], h_tar)
            # h_tar_init = repeat(h_tar_init, self.config["NUM_SIMULATIONS"])

            h_on_init = repeat(h_on, self.config["NUM_SIMULATIONS"])
            h_tar_init = repeat(h_tar, self.config["NUM_SIMULATIONS"])

            _, online_preds = self.agent.apply(online_params, h_on_init, xs)
            _, target_preds = self.agent.apply(target_params, h_tar_init, xs)

            all_t_mask = simulation_finished_mask(l_mask[None], next_t)

            batch_td_error, batch_loss_mean, metrics, log_info = self.batch_loss(
                timestep=all_t,
                online_preds=online_preds,
                target_preds=target_preds,
                actions=all_a,
                rewards=all_t.reward,
                is_last=make_float(all_t.last()),
                non_terminal=all_t.discount,
                loss_mask=all_t_mask,
            )

            # print("BATCH TD ERROR:", batch_td_error.shape)
            # print("BATCH LOSS MEAN:", batch_loss_mean.shape)
            # print("METRICS:", jax.tree.map(lambda x: x.shape, metrics))
            # print("LOG INFO:", jax.tree.map(lambda x: x.shape, log_info))

            return batch_td_error, batch_loss_mean, metrics, log_info
        
        # Vmap over each window
        batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(_dyna_loss_fn)(
                timesteps,  # [W, T', ...]
                actions,  # [W, T']
                h_online,  # [W, T', D]
                h_target,  # [W, T', D]
                loss_mask,  # [W, T']
                jax.random.split(rng, len(actions)),  # [W, 2]
            )

        batch_td_error = batch_td_error.mean()  # [W, T', num_sim] -> []
        batch_loss_mean = batch_loss_mean.mean()  # [W, num_sim] -> []

        return batch_td_error, batch_loss_mean, metrics, log_info


# --- Training Loop Structure (Conceptual) ---
def make_train(config):
    # Create env and env_params
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env = TimestepWrapper(env)
    env_params = env.default_params

    # Vmap over environment
    vmap_reset = lambda num_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, num_envs), env_params
    )
    vmap_step = lambda num_envs: lambda rng, timestep, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, num_envs), timestep, action, env_params)

    logger = Logger()

    NUM_UPDATES = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"] // config["TRAINING_INTERVAL"]
    )

    def train(rng):
        # Initialize environment
        rng, _rng = jax.random.split(rng, 2)
        init_timestep = vmap_reset(config["NUM_ENVS"])(_rng)

        # Initialize DynaAgent
        agent = DynaAgent(
            config=config,
            env=env,
            env_params=env_params,
        )
        rng, init_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_carry = DynaAgent.initialize_carry(config["NUM_ENVS"], config["RNN_HIDDEN_DIM"])
        online_params = agent.init(init_rng, init_carry, init_x)
        target_params = jax.tree.map(lambda x: jnp.copy(x), online_params)
        
        # Initialize Optimizer
        lr = config["LR"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=config["EPS_ADAM"]),
        )
        
        # Initialize CustomTrainState
        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=online_params,
            target_params=target_params,
            tx=tx,
            timesteps=0,
            n_updates=0,
        )
        
        # Initialize Trajectory Replay Buffer
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["TOTAL_BATCH_SIZE"] // config["SAMPLE_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_batch_size=config["SAMPLE_BATCH_SIZE"],
            sample_sequence_length=config["TOTAL_BATCH_SIZE"] // config["SAMPLE_BATCH_SIZE"],
            period=config["SAMPLING_PERIOD"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        dummy_timestep = jax.tree.map(lambda x: jnp.zeros_like(x[0]), init_timestep)
        dummy_transition = Transition(
            timestep=dummy_timestep,
            action=jnp.array(0),
            agent_state=jnp.zeros((config["RNN_HIDDEN_DIM"],))
        )
        buffer_state = buffer.init(dummy_transition)
        
        # Define actor single step + policy
        # BELOW is in range of ~(.9,.1)
        rng, eps_rng = jax.random.split(rng, 2)
        vals = np.logspace(
            num=config["NUM_EPSILONS"],
            start=config["EPSILON_MIN"],
            stop=config["EPSILON_MAX"],
            base=config["EPSILON_BASE"],
        )
        act_eps = jax.random.choice(eps_rng, vals, shape=(config["NUM_ENVS"],))
        actor_policy = FixedEpsilonGreedy(
            epsilons=act_eps
        )

        def actor_step(carry, unused):
            agent_state, timestep, rng = carry
            rng, act_rng, env_rng = jax.random.split(rng, 3)

            # Format inputs for apply fn
            obs, discount = timestep.observation, timestep.discount
            obs = obs[np.newaxis, :]
            discount = discount[np.newaxis, :]
            resets = 1.0 - discount
            x = (obs, resets)

            # Get action from agent
            next_agent_state, preds = agent.apply(
                train_state.params,
                agent_state,
                x,
            )

            # Remove time dim
            q_vals = preds.q_vals.squeeze(0)

            # Get next action
            action = actor_policy.choose_actions(q_vals, act_rng)

            # Get next timestep
            next_timestep = vmap_step(config["NUM_ENVS"])(env_rng, timestep, action)

            # Create transition
            transition = Transition(timestep=next_timestep, action=action, agent_state=next_agent_state)

            return (next_agent_state, next_timestep, rng), transition
        
        # Policy for simulation
        rng, eps_rng = jax.random.split(rng)
        vals = np.logspace(num=256, start=1, stop=3, base=0.1) # ACME default values, equivalent to SIM_EPSILON_SETTING=1
        sim_eps = jax.random.choice(eps_rng, vals, shape=(config["NUM_SIMULATIONS"] - 1,))
        sim_eps = jnp.concatenate((jnp.array((0,)), sim_eps))
        simulation_policy = FixedEpsilonGreedy(
            epsilons=sim_eps
        )

        # Initialize DynaLossFn instance
        loss_fn = DynaLossFn(
            agent=agent,
            config=config,
            simulation_policy=simulation_policy.choose_actions
        )
        
        # Define learning step
        def learn_step(train_state, buffer_state, rng):
            # Split RNG for sampling and loss calculation
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)
            
            # Sample batch of sequences from buffer -> sampled_batch
            sampled_batch = buffer.sample(buffer_state, sample_rng).experience

            # Get initial RNN states (online/target) from sampled_batch.experience.agent_state
            init_state = jax.tree.map(
                lambda x: x[:, 0],  # Take first element along time dimension
                sampled_batch.agent_state
            )

            # Call jax.value_and_grad(loss_fn.calculate_loss, has_aux=True)(...)
            #       - Pass online_params, target_params, model_params, learn_batch, initial_states, rng
            #       - Returns: (td_error, loss, metrics, log_info), grads
            (loss, (td_error, metrics, log_info)), grads = jax.value_and_grad(loss_fn.total_loss, has_aux=True)(
                train_state.params,
                train_state.target_params,
                sampled_batch,
                init_state,
                loss_rng
            )

            # print("TD Error:", td_error.shape)
            # print("Loss:", loss.shape)
            # print("Metrics:", jax.tree.map(lambda x: x.shape, metrics))
            # print("Log info:", jax.tree.map(lambda x: x.shape, log_info))
            # print("Grads:", jax.tree.map(lambda x: x.shape, grads))

            # Apply gradients: train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.apply_gradients(grads=grads)

            # TODO: Update priorities in buffer: buffer_state = buffer.set_priorities(buffer_state, indices, aux_data["priorities"])
            # Increment update counter in train_state
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            return train_state, buffer_state, metrics, log_info, grads, rng
        
        # get dummy values for metrics, log_info, grads
        _, _, dummy_metrics, dummy_log_info, dummy_grads, _ = learn_step(
            train_state,
            buffer_state,
            rng
        )

        def _train_step(runner_state, unused):
            train_state, env_timestep, agent_state, buffer_state, rng = runner_state

            # --- 1. Collect Experience ---
            # Use jax.lax.scan with actor_step and env.step (vmap_step) for config["TRAINING_INTERVAL"] steps
            #       - Manage agent_state (RNN state) across steps
            #       - Collect sequence of Transitions (including agent_state) -> traj_batch
            (agent_state, timestep, rng), traj_batch = jax.lax.scan(
                actor_step,
                (agent_state, env_timestep, rng),
                None,
                config["TRAINING_INTERVAL"]
            )

            # --- 2. Add to Buffer ---
            # Transpose traj_batch if needed ([T, B] -> [B, T])
            traj_batch = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_batch)

            # Add traj_batch to buffer: buffer_state = buffer.add(buffer_state, traj_batch)
            buffer_state = buffer.add(buffer_state, traj_batch)

            # Update total timesteps counter in train_state
            train_state = train_state.replace(timesteps=train_state.timesteps + config["TRAINING_INTERVAL"] * config["NUM_ENVS"])

            ##############################
            # 1. Learner update
            ##############################
            # TODO: match up shapes for true and false
            # NOTE: move _learn_step outside of make_train and call it with dummy inputs to get values for False
            # NOTE: Reference vbb
            is_learn_time = (train_state.timesteps > config["LEARNING_STARTS"]) & buffer.can_sample(buffer_state)
            train_state, buffer_state, metrics, log_info, grads, rng = jax.lax.cond(
                is_learn_time,
                lambda ts, bs, r: learn_step(ts, bs, r),
                lambda ts, bs, r: (ts, bs, dummy_metrics, dummy_log_info, dummy_grads, r),
                train_state,
                buffer_state,
                rng
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % config["LEARNER_LOG_PERIOD"] == 0
            )

            jax.lax.cond(
                is_log_time,
                lambda: logger.metrics_logger(train_state, metrics),
                lambda: None,
            )

            # Log extra learner plots
            is_extra_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0
            )
            jax.lax.cond(
                is_extra_log_time,
                lambda d: logger.extra_logger(d),
                lambda d: None,
                log_info
            )

            # Log gradients
            is_grad_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % config["GRADIENT_LOG_PERIOD"] == 0
            )

            jax.lax.cond(
                is_grad_log_time,
                lambda: logger.gradient_logger(train_state, grads),
                lambda: None,
            )

            # --- 6. Return updated states ---
            return (train_state, timestep, agent_state, buffer_state, rng), {}

        # JIT the _train_step function
        _train_step = jax.jit(_train_step)

        # Initialize the initial states
        runner_state = (
            train_state,
            init_timestep,
            init_carry,
            buffer_state,
            rng
        )

        # Run the main loop
        runner_state, _ = jax.lax.scan(_train_step, runner_state, None, NUM_UPDATES)

        # Return results
        return runner_state

    return train # End of make_train

# --- Main Execution ---
if __name__ == "__main__":
    # Define config dictionary
    config = {
        # --- Environment Settings ---
        "ENV_NAME": "Craftax-Symbolic-v1", # Example, adjust as needed
        "NUM_ENVS": 32,  # Number of parallel environments (PureJaxRL DQN used 10, can increase)

        # --- Training Loop Settings ---
        "TOTAL_TIMESTEPS": 5_000_000,    # Total environment steps
        "TRAINING_INTERVAL": 5,          # How many env steps per actor sequence collection
        "LEARNING_STARTS": 10_000,       # Timesteps before learning begins
        "TARGET_UPDATE_INTERVAL": 1_000, # How many LEARNER UPDATES between target network syncs (R2D2 uses ~2500 steps)

        # --- Network Settings ---
        "RNN_HIDDEN_DIM": 256,     # Size of RNN hidden state
        "ENCODER_HIDDEN_DIM": 256, # Hidden dim for observation encoder MLP
        "NUM_ENCODER_LAYERS": 0,   # Hidden layers for observation encoder MLP
        "Q_HIDDEN_DIM": 1024,      # Hidden dim for Q-head MLP
        "NUM_Q_LAYERS": 2,         # Hidden layers for Q-head MLP
        "USE_BIAS": True,          # Whether to use bias in Dense layers

        # --- Optimizer Settings ---
        "LR": 3e-4,
        "LR_LINEAR_DECAY": False,  # Whether to use linear LR decay
        "EPS_ADAM": 1e-5,          # Adam optimizer epsilon
        "MAX_GRAD_NORM": 80,       # Gradient clipping norm
        "TAU": 1.0,

        # --- Buffer Settings ---
        "BUFFER_SIZE": 50_000,     # Total transitions in buffer (R2D2 often uses 1M+, adjust based on memory)
        "TOTAL_BATCH_SIZE": 1280,  # Total transitions sampled from buffer
        "SAMPLE_BATCH_SIZE": 32,   # Batch size sampled from buffer for learning (e.g., 32, 64)
        "SAMPLING_PERIOD": 1,      # Store sequences overlapping by N-1 steps (1 is standard)

        # --- Loss Function Settings ---
        "GAMMA": 0.99,             # Discount factor
        "TD_LAMBDA": 0.9,          # TD-Lambda parameter
        "STEP_COST": 0.0,          # Optional cost added per step (DynaLossFn default 0.0)
        "ONLINE_COEFF": 1.0,       # Weight for the loss on real data
        "DYNA_COEFF": 1.0,         # Weight for the loss on simulated data (DynaLossFn default 1.0)

        # --- Dyna Simulation Settings ---
        "NUM_SIMULATIONS": 2,       # Number of parallel simulations per starting state (DynaLossFn default 2)
        "SIMULATION_LENGTH": 10,    # Length of each simulated rollout (DynaLossFn default 5)
        "WINDOW_SIZE": 1,           # Number of windows to use, must be 1 for DynaLossFn

        # --- Actor Settings (Exploration) ---
        # Choose one exploration strategy
        "NUM_EPSILONS": 256,        # Number of epsilon schedules
        "EPSILON_MIN": 0.05,        # Minimum epsilon
        "EPSILON_MAX": 0.9,         # Maximum epsilon
        "EPSILON_BASE": 0.1,        # Base epsilon

        # --- Logging ---
        "LEARNER_LOG_PERIOD": 500,  # How many LEARNER UPDATES between logging losses/metrics
        "GRADIENT_LOG_PERIOD": 500, # How many GRADIENT UPDATES between logging losses/metrics
        "LEARNER_EXTRA_LOG_PERIOD": 5_000, # How many LEARNER UPDATES between extra logging

        # --- Miscellaneous ---
        "SEED": 1,
        "NUM_SEEDS": 1,
        "ENTITY": "hoonshin",
        "PROJECT": "dyna-crafter",
        "WANDB_MODE": "online",
    }

    # Initialize wandb
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["Dyna", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'dyna_crafter',
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Call make_train(config, env, env_params)
    rng = jax.random.PRNGKey(config["SEED"])

    train_jit = jax.jit(make_train(config))
    outs = jax.block_until_ready(train_jit(rng))