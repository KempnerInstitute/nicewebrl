import pytest
import jax
import jax.numpy as jnp
from nicegui import app
from nicewebrl.stages import Stage, Block, EnvStage, MultiAgentEnvStage, EnvStageState, StageStateModel
from nicewebrl.nicejax import TimeStep, JaxWebEnv, MultiAgentJaxWebEnv
from flax import struct
import dataclasses
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import tempfile
import os
import msgpack
from tortoise import fields, models
import pytest_asyncio
import random

@struct.dataclass
class MockTimeStep:
    first: bool = False
    last: bool = False
    state: dict = None

class MockEnv:
    def __init__(self):
        self.num_actions = 4
        self.action_space = type('ActionSpace', (), {'n': 4})()

    def reset(self, rng, env_params):
        return {}

    def step(self, rng, timestep, action, env_params):
        return {}

class MockWebEnv(JaxWebEnv):
    def __init__(self):
        super().__init__(env=MockEnv())
    
    def reset(self, rng, env_params):
        return MockTimeStep()
    
    def next_steps(self, rng, timestep, action, env_params):
        return MockTimeStep()

class MockMultiAgentWebEnv(MultiAgentJaxWebEnv):
    def __init__(self):
        super().__init__(env=MockEnv())
    
    def reset(self, rng, env_params):
        return MockTimeStep()
    
    def next_steps(self, rng, timestep, action, env_params):
        return MockTimeStep()

@pytest.fixture
def mock_container():
    return Mock()

@pytest.fixture
def mock_app_storage(monkeypatch):
    mock_storage = Mock()
    mock_storage.user = {"seed": 0}  # Use an integer seed!
    app.storage = mock_storage
    return mock_storage

@pytest.fixture
def basic_stage():
    return Stage(
        name="test_stage",
        title="Test Stage",
        body="Test body",
        display_fn=AsyncMock(),
        next_button=True
    )

def dummy_render_fn(*args, **kwargs):
    return 0

def dummy_vmap_render_fn(*args, **kwargs):
    return 0

@pytest.fixture
def env_stage():
    return EnvStage(
        name="test_env_stage",
        web_env=MockWebEnv(),
        action_keys={0: "up", 1: "down", 2: "left", 3: "right"},
        evaluate_success_fn=lambda x, y: True,
        render_fn=dummy_render_fn,
        vmap_render_fn=dummy_vmap_render_fn,
    )

@pytest.fixture
def multi_agent_env_stage():
    return MultiAgentEnvStage(
        name="test_multi_agent_stage",
        web_env=MockMultiAgentWebEnv(),
        action_keys={0: "up", 1: "down", 2: "left", 3: "right"},
        evaluate_success_fn=lambda x, y: True,
        render_fn=dummy_render_fn,
        vmap_render_fn=dummy_vmap_render_fn,
    )

@pytest.fixture
def block():
    s1 = Stage(name="stage1", display_fn=AsyncMock())
    s2 = Stage(name="stage2", display_fn=AsyncMock())
    return Block(name="test_block", stages=[s1, s2], randomize=False, metadata={"name": "test_block"})

def test_stage_initialization(basic_stage):
    """Test stage initialization"""
    assert basic_stage.name == "test_stage"
    assert basic_stage.title == "Test Stage"
    assert basic_stage.body == "Test body"
    assert basic_stage.next_button is True
    assert basic_stage.finished is False

@pytest.mark.asyncio
async def test_stage_user_data(basic_stage, mock_app_storage):
    """Test stage user data management"""
    # Test setting user data
    await basic_stage.set_user_data(test_key="test_value")
    assert basic_stage.get_user_data("test_key") == "test_value"

    # Test popping user data
    value = basic_stage.pop_user_data("test_key")
    assert value == "test_value"
    assert basic_stage.get_user_data("test_key") is None

@pytest.mark.asyncio
async def test_stage_activation(basic_stage):
    """Test stage activation"""
    container = Mock()
    await basic_stage.activate(container)
    basic_stage.display_fn.assert_called_once_with(stage=basic_stage, container=container)

@pytest.mark.asyncio
async def test_env_stage_initialization(env_stage):
    """Test environment stage initialization"""
    assert env_stage.name == "test_env_stage"
    assert isinstance(env_stage.web_env, MockWebEnv)
    assert env_stage.action_keys == {0: "up", 1: "down", 2: "left", 3: "right"}
    assert env_stage.next_button is False

@pytest.mark.asyncio
async def test_multi_agent_env_stage_initialization(multi_agent_env_stage):
    """Test multi-agent environment stage initialization"""
    assert multi_agent_env_stage.name == "test_multi_agent_stage"
    assert isinstance(multi_agent_env_stage.web_env, MockMultiAgentWebEnv)
    assert multi_agent_env_stage.action_keys == {0: "up", 1: "down", 2: "left", 3: "right"}
    assert multi_agent_env_stage.next_button is False

def test_block_initialization(block):
    """Test block initialization"""
    assert block.name == "test_block"
    assert len(block.stages) == 2
    assert block.randomize is False
    assert block.metadata["name"] == "test_block"

@pytest.mark.asyncio
async def test_block_stage_management(block, mock_app_storage):
    """Test block stage management"""

    container = Mock()
    stage = await block.get_stage()
    await stage.activate(container)
    stage.display_fn.assert_called_once_with(stage=stage, container=container)


    await block.advance_stage()
    new_stage = await block.get_stage()
    assert new_stage != stage

@pytest.mark.asyncio
async def test_block_randomization(block):
    """Test block randomization"""
    block.randomize = True
    original_order = [stage.name for stage in block.stages]
    stage_order = await block.get_user_stage_order()
    assert len(stage_order) == len(block.stages)
    assert set(stage_order) == set(range(len(block.stages)))

@pytest.mark.asyncio
async def test_env_stage_state_management(env_stage, mock_app_storage):
    """Test environment stage state management"""
    # Test initial state
    assert env_stage.get_user_data("stage_state") is None
    
    # Test state update
    state = EnvStageState(name="test_stage", timestep=MockTimeStep())
    await env_stage.set_user_data(stage_state=state)
    assert env_stage.get_user_data("stage_state") == state

@pytest.mark.asyncio
async def test_stage_finish(basic_stage, mock_app_storage):
    """Test stage finishing"""
    # Patch finish_stage to set finished directly for test
    with patch.object(basic_stage, 'finish_stage', wraps=basic_stage.finish_stage) as patched_finish:
        await basic_stage.finish_stage()
        # After finish_stage, finished should be True
        assert basic_stage.finished is True or basic_stage.get_user_data('finished') is True

@pytest.mark.asyncio
async def test_env_stage_key_press(env_stage, mock_app_storage):
    """Test environment stage key press handling"""
    # Test key press handling
    container = Mock()
    event = Mock()
    event.args = {"key": "up"}
    await env_stage.handle_key_press(event, container)
    # Add assertions based on expected behavior

@pytest.mark.asyncio
async def test_block_stage_order(block, mock_app_storage):
    """Test block stage ordering with randomization"""
    # Test stage order
    stage_idx = await block.get_block_stage_idx()
    assert stage_idx == 0

    # Test stage advancement
    await block.advance_stage()
    new_stage_idx = await block.get_block_stage_idx()
    assert new_stage_idx == 1

@pytest.mark.asyncio
async def test_env_stage_save_data(env_stage, mock_app_storage):
    """Test environment stage data saving"""
    # Test data saving
    state = EnvStageState(name="test_stage", timestep=MockTimeStep())
    await env_stage.set_user_data(stage_state=state)
    assert env_stage.get_user_data("stage_state") == state

@pytest.mark.asyncio
async def test_block_metadata_broadcast(block, mock_app_storage):
    """Test block metadata broadcasting to stages"""
    block.metadata = {"test_key": "test_value"}
    
    stage = Stage(name="test_stage")
    block.stages.append(stage)
    
    block.__post_init__()
    
    for stage in block.stages:
        assert "block_metadata" in stage.metadata
        assert stage.metadata["block_metadata"] == block.metadata

def test_stage_state_model():
    """Test StageStateModel"""
    model = StageStateModel(
        id=1,
        name="test_stage",
        session_id="test_session",
        data=b"test_data"
    )
    assert model.id == 1
    assert model.name == "test_stage"
    assert model.session_id == "test_session"
    assert model.data == b"test_data"

from nicewebrl.stages import StageStateModel



@pytest.mark.asyncio
async def test_concurrent_safe_save():
    """Test that safe_save handles concurrent saves correctly."""
    from nicewebrl.stages import safe_save
    from tortoise import Tortoise
    
    # Initialize database and register models directly
    await Tortoise.init(
    db_url='sqlite://:memory:',
    modules={'models': ['nicewebrl.stages']}
)
    await Tortoise.generate_schemas()

    try:
        num_users = 20
        num_saves = 5  # Number of saves per user
        
        async def simulate_user(user_id):
            for i in range(num_saves):
                model = StageStateModel(
                    name=f"user_{user_id}_save_{i}",
                    session_id=f"session_{user_id}",
                    data=b"test_data"
                )
                
                await safe_save(model)
                await asyncio.sleep(random.uniform(0.01, 0.05))
        
        await asyncio.gather(*[simulate_user(i) for i in range(num_users)])
        
        all_records = await StageStateModel.all()
        records = list(all_records)
        
        expected_records = num_users * num_saves
        assert len(records) == expected_records, f"Expected {expected_records} records, got {len(records)}"
        
        for user_id in range(num_users):
            user_records = [r for r in records if r.session_id == f"session_{user_id}"]
            assert len(user_records) == num_saves, f"User {user_id} should have {num_saves} records"
            
            save_numbers = sorted([int(r.name.split('_')[-1]) for r in user_records])
            assert save_numbers == list(range(num_saves)), f"User {user_id} records are not in sequence"
            
    finally:
        await Tortoise.close_connections()



