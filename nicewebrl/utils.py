import asyncio
import functools
import json
import os.path
import random
import struct
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import aiofiles
import jax
import jax.numpy as jnp
import msgpack
from fastapi import Request
from nicegui import Client, app, ui

from nicewebrl.logging import get_logger

# Type definitions
T = TypeVar('T')

# Module-level variables
logger = get_logger(__name__)
_user_locks: Dict[str, asyncio.Lock] = {}

def get_user_lock() -> asyncio.Lock:
    """Get or create an asyncio Lock for the current user.
    
    Returns:
        asyncio.Lock: A lock instance specific to the current user.
    """
    user_seed = app.storage.user["seed"]
    if user_seed not in _user_locks:
        _user_locks[user_seed] = asyncio.Lock()
    return _user_locks[user_seed]


async def toggle_fullscreen() -> bool:
    """Toggle the browser's fullscreen state.
    
    Returns:
        bool: True if the operation was successful.
    """
    logger.info("Toggling fullscreen")
    return await ui.run_javascript(
        """
        return (async () => {
            if (!document.fullscreenElement) {
                await document.documentElement.requestFullscreen();
            } else {
                if (document.exitFullscreen) {
                    await document.exitFullscreen();
                }
            }
            return true;
        })();
        """,
        timeout=10,
    )


async def prevent_default_spacebar_behavior(should_prevent: bool) -> None:
    """Set whether the spacebar's default behavior (fullscreen toggle) is prevented.

    Args:
        should_prevent: Whether to prevent the spacebar's default behavior.
    """
    logger.info(f"Setting spacebar behavior: should_prevent={should_prevent}")
    await ui.run_javascript(
        f"""
        return preventDefaultSpacebarBehavior({str(should_prevent).lower()});
        """,
        timeout=10,
    )


async def check_fullscreen() -> Optional[bool]:
    """Check if the browser is currently in fullscreen mode.
    
    Returns:
        Optional[bool]: True if in fullscreen mode, False if not, None if check failed.
    """
    result = None
    try:
        result = await ui.run_javascript(
            "return document.fullscreenElement !== null;", timeout=10
        )
    except TimeoutError as e:
        logger.error(f"JavaScript execution timed out: {e}")
    return result


def clear_element(element: Any) -> None:
    """Clear the contents of a UI element.
    
    Args:
        element: The UI element to clear.
    """
    try:
        element.clear()
    except Exception as e:
        logger.error(f"Error clearing element: {e}")


async def wait_for_button_or_keypress(
    button: Any, 
    ignore_recent_press: bool = False
) -> Any:
    """Wait for either a button click or keyboard press.
    
    Args:
        button: The button to monitor for clicks.
        ignore_recent_press: If True, ignore keypresses that occur too soon after
            the previous one.
            
    Returns:
        The event that triggered the wait (either button click or keypress).
    """
    attempt = 0
    while True:
        try:
            key_pressed_future = asyncio.get_event_loop().create_future()
            last_key_press_time = asyncio.get_event_loop().time()

            def on_keypress(event: Any) -> None:
                nonlocal last_key_press_time
                current_time = asyncio.get_event_loop().time()

                if ignore_recent_press:
                    if (current_time - last_key_press_time) > 0.5 and not key_pressed_future.done():
                        key_pressed_future.set_result(event)
                else:
                    if not key_pressed_future.done():
                        key_pressed_future.set_result(event)

                last_key_press_time = current_time

            keyboard = ui.keyboard(on_key=on_keypress)

            try:
                tasks = [button.clicked(), key_pressed_future]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    if not task.done():
                        task.cancel()

                for task in done:
                    return task.result()

            except asyncio.CancelledError as e:
                logger.error(f"{attempt}. Task was cancelled. Cleaning up and retrying...")
                logger.error(f"Error: '{e}'")
                continue

            finally:
                try:
                    keyboard.delete()
                except Exception as e:
                    logger.error(f"{attempt}. Error deleting keyboard: '{str(e)}'")

        except Exception as e:
            logger.error(
                f"{attempt}. Waiting for button or keypress. Error occurred: "
                f"'{str(e)}'. Retrying..."
            )
            attempt += 1
            if attempt > 10:
                return
            await asyncio.sleep(1)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1,
    max_delay: float = 10
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        
    Returns:
        A decorator function that adds retry logic to the decorated function.
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(
                            f"All {max_retries} attempts failed. Last error: {str(e)}"
                        )
                        raise

                    delay = min(
                        base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1),
                        max_delay,
                    )
                    logger.error(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)

            raise Exception("Unexpected error in retry logic")

        return wrapper

    return decorator


def basic_javascript_file() -> str:
    """Get the path to the basic JavaScript file.
    
    Returns:
        str: Absolute path to the basics.js file.
    """
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    return f"{current_directory}/basics.js"


def multihuman_javascript_file() -> str:
    """Get the path to the multi-human JavaScript file.
    
    Returns:
        str: Absolute path to the multihuman_basics.js file.
    """
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    return f"{current_directory}/multihuman_basics.js"


def initialize_user(*, seed: int = 0, request: Optional[Request] = None) -> None:
    """Initialize user-specific data and settings.

    Args:
        seed: The seed for random number generation. Defaults to 0.
        request: Optional FastAPI request object.
    """
    if seed:
        app.storage.user["seed"] = seed
    else:
        app.storage.user["seed"] = app.storage.user.get(
            "seed", random.getrandbits(32)
        )

    app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0)
    if "rng_key" not in app.storage.user:
        rng_key = jax.random.PRNGKey(app.storage.user["seed"])
        app.storage.user["rng_key"] = rng_key.tolist()
        app.storage.user["init_rng_key"] = app.storage.user["rng_key"]
    app.storage.user["session_start"] = app.storage.user.get(
        "session_start", datetime.now().isoformat()
    )
    app.storage.user["session_duration"] = 0

    ############################################################
    # Log worker information from Mturk
    ############################################################
    if request is not None:
        app.storage.user["worker_id"] = request.query_params.get("workerId", None)
        app.storage.user["hit_id"] = request.query_params.get("hitId", None)
        app.storage.user["assignment_id"] = request.query_params.get("assignmentId", None)

        app.storage.user["user_id"] = (
            app.storage.user["worker_id"] or app.storage.user["seed"]
        )

    ############################################################
    # needed to maintain connection to client
    ############################################################
    def print_ping(e):
        logger.info(str(e.args))

    ui.on("ping", print_ping)

def get_progress():
    progress = float(
        f"{(app.storage.user['stage_idx'] + 1) / app.storage.user['num_stages']:.2f}"
    )
    app.storage.user["stage_progress"] = progress
    return progress

def user_data_file():
    return f"data/user_data_{app.storage.user['seed']}.msgpack"


def user_metadata_file():
    return f"data/user_data_{app.storage.user['seed']}.json"


def save_metadata(metadata: Dict, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(metadata))


def get_user_session_minutes():
    start_time = datetime.fromisoformat(app.storage.user["session_start"])
    current_time = datetime.now()
    duration = current_time - start_time
    minutes_passed = duration.total_seconds() / 60
    app.storage.user["session_duration"] = minutes_passed
    return minutes_passed


def broadcast_message(event: str, message: str):
    called_by_user_id = str(app.storage.user["seed"])
    called_by_room_id = str(app.storage.user["room_id"])
    stage = app.storage.user["stage_idx"]
    fn = f"userMessage('{called_by_room_id}', '{called_by_user_id}', '{event}', '{stage}', '{message}')"
    logger.info(fn)
    for client in Client.instances.values():
        with client:
            ui.run_javascript(fn)


async def write_msgpack_record(f, data):
    """Write a length-prefixed msgpack record to a file.

    Args:
        f: An aiofiles file object opened in binary mode
        data: The data to write
    """
    packed_data = msgpack.packb(data)
    length = len(packed_data)
    await f.write(length.to_bytes(4, byteorder="big"))
    await f.write(packed_data)


async def read_msgpack_records(filepath: str):
    """Read length-prefixed msgpack records from a file.

    Args:
        filepath: Path to the file containing the records

    Yields:
        Decoded msgpack records one at a time
    """
    async with aiofiles.open(filepath, "rb") as f:
        while True:
            # Read length prefix (4 bytes)
            length_bytes = await f.read(4)
            if not length_bytes:  # End of file
                break

            # Convert bytes to integer
            length = int.from_bytes(length_bytes, byteorder="big")

            # Read the record data
            data = await f.read(length)
            if len(data) < length:  # Incomplete record
                logger.error(
                    f"Corrupt data in {filepath}: Expected {length} bytes but got {len(data)}"
                )
                # break

            # Unpack and yield the record
            try:
                record = msgpack.unpackb(data, strict_map_key=False)
                yield record
            except Exception as e:
                logger.error(f"Failed to unpack record in {filepath}: {e}")
                break


def read_msgpack_records_sync(filepath: str):
    """Synchronous version of read_msgpack_records that reads msgpack records from a file."""
    try:
        with open(filepath, "rb") as f:
            # Read the file content
            content = f.read()

            # Initialize position
            pos = 0

            # Read records until we reach the end of the file
            while pos < len(content):
                # Read the size of the next record
                size_bytes = content[pos : pos + 4]
                if len(size_bytes) < 4:
                    break

                size = struct.unpack(">I", size_bytes)[0]
                pos += 4

                # Read the record data
                if pos + size > len(content):
                    logger.error(f"Incomplete record in {filepath}")
                    break

                data = content[pos : pos + size]
                pos += size

                # Unpack and yield the record
                try:
                    record = msgpack.unpackb(data, strict_map_key=False)
                    yield record
                except Exception as e:
                    logger.error(f"Failed to unpack record in {filepath}: {e}")
                    break
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")


async def read_all_records(filepath: str) -> List[Dict]:
    """Helper function to read all records into a list."""
    return [record async for record in read_msgpack_records(filepath)]


def read_all_records_sync(filepath: str) -> List[Dict]:
    """Synchronous version that reads all msgpack records from a file and returns them as a list."""
    return list(read_msgpack_records_sync(filepath))

load_data = read_all_records_sync