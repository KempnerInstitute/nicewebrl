"""

A simple utility for logging to files and the console. Key features:
- prepend user_id to console logs
- create user-specific log files with all stdout and stderr (useful for user-specific debugging)

NOTE: you only need to call setup_logging once at the beginning of your program.
  the settings will propogate to every other usage.
  afterwards, you just need to call `logger = logging.get_logger(__name__)` and `logger.info('some message')`, etc as usual.

log format:
- which user index, total number of users, and user_id
- timestamp
- logger name
- message

Usage:

from nicewebrl import logging
logger = logging.get_logger(__name__)

def log_filename_fn(log_dir, user_id):
  return os.path.join(log_dir, f'log_{user_id}.log')

logging.setup_logging(
    LOG_DIR,
    log_filename_fn=log_filename_fn,  # optional, defaults to above
    # use whatever key you use to identify users in nicegui's app.storage.user
    nicegui_storage_user_key='user_id'
)
logger = logging.get_logger('some_name')

logger.info('some message')
> (2/31) 3266458590: 2024/10/26 20:24:47 - some_name - INFO - some message
logger.error('some error message')
> (2/31) 3266458590: 2024/10/26 20:24:47 - some_name - ERROR - some error message
logger.warning('some warning message')
> (2/31) 3266458590: 2024/10/26 20:24:47 - some_name - WARNING - some warning message
"""

import io
import logging
import os
import sys
import traceback
from functools import lru_cache
from typing import Callable, Optional

from nicegui import app


class UserAwareOutput(io.TextIOBase):
  """
  A custom output stream that handles user-specific and global logging.

  This class extends io.TextIOBase to provide a flexible logging solution
  that can differentiate between user-specific and global log messages.
  It manages separate buffers for each user and a global buffer.

  Attributes:
      console_stream (io.TextIOBase): The underlying console stream for output.
      user_key (str): The key used to identify users in the app storage.
      log_dir (str): The directory where log files are stored.
      error_log (str): The path to the error log file.
      buffers (dict): A dictionary to store user-specific StringIO buffers.
  """

  def __init__(
    self,
    console_stream: io.TextIOBase,
    log_dir: str,
    log_filename_fn: Optional[Callable[[str, str], str]] = None,
    nicegui_storage_user_key: str = "user_id",
    colored: bool = True,
  ):
    """
    Initialize the UserAwareOutput instance.

    Args:
        console_stream (io.TextIOBase): The underlying console stream for output.
        log_dir (str): The directory where log files will be stored.
        log_filename_fn (Optional[Callable[[str, str], str]], optional): Function to generate log filenames.
            Takes log_dir and user_id as arguments and returns the full file path.
            Defaults to lambda log_dir, user_id: os.path.join(log_dir, f'log_{user_id}.log')
        nicegui_storage_user_key (str, optional): The key used to identify users in the app storage.
            Defaults to 'user_id'.
    """
    self.console_stream = console_stream
    self.user_key = nicegui_storage_user_key
    self.log_dir = log_dir
    self.error_log = os.path.join(log_dir, "error.log")
    self.joint_log = os.path.join(log_dir, "joint.log")
    self.buffers = {}  # Dictionary to store user-specific buffers
    self.user_to_idx = {}
    self.user_to_color = {}  # New: store color assignments
    self.colored = colored
    if log_filename_fn is None:
      log_filename_fn = lambda log_dir, user_id: os.path.join(
        log_dir, f"log_{user_id}.log"
      )
    self.log_filename_fn = log_filename_fn

  def _get_buffer(self, user_id):
    """
    Get or create a buffer for a specific user.
    """
    if user_id not in self.buffers:
      self.buffers[user_id] = io.StringIO()
      self.user_to_idx[user_id] = len(self.buffers) - 1
    return self.buffers[user_id]

  @property
  def nusers(self):
    if "global" in self.buffers:
      return len(self.buffers) - 1  # exclude global
    else:
      return len(self.buffers)

  def write(self, s: str, get_user_id: bool = True) -> int:
    """
    Write a string to the appropriate output stream.

    Args:
        s (str): The string to write.
        get_user_id (bool, optional): Whether to attempt to get a user ID. Defaults to True.

    Returns:
        int: The number of characters written.
    """
    try:
      if get_user_id:
        return self.write_user(s)
      else:
        return self.write_global(s)
    except Exception as e:
      self.log_error(
        f"Error in UserAwareOutput.write: {str(e)}\n{traceback.format_exc()}"
      )
      return 0

  def write_user(self, s: str) -> int:
    """
    Write a string to a user-specific log file and buffer.
    """
    try:
      user_id = app.storage.user.get(self.user_key, None)
      log_file = self.log_filename_fn(self.log_dir, user_id)

      # Format the message once
      formatted_msg = self._format_message(s, user_id)

      # Write to user-specific log file
      with open(log_file, "a") as file_stream:
        file_stream.write(formatted_msg)
        file_stream.flush()

      # Write to joint log
      with open(self.joint_log, "a") as file_stream:
        file_stream.write(formatted_msg)
        file_stream.flush()

      # Write to console buffer
      buffer = self._get_buffer(user_id)
      buffer.write(formatted_msg)
      if s.endswith("\n"):
        output = buffer.getvalue()
        self.console_stream.write(output)
        buffer.truncate(0)
        buffer.seek(0)
      return len(formatted_msg)
    except Exception as e:
      if "app.storage.user" in str(e):
        # Silently ignore this specific error and fall back to global write
        return self.write_global(s)
      else:
        # Log all other exceptions
        self.log_error(f"Error in write_user: {str(e)}\n{traceback.format_exc()}")
        return 0

  def write_global(self, s: str) -> int:
    """
    Write a string to the global buffer.
    """
    buffer = self._get_buffer("global")
    buffer.write(s)
    if s.endswith("\n"):
      output = buffer.getvalue()
      self.console_stream.write(output)
      buffer.truncate(0)
      buffer.seek(0)
    return len(s)

  def _get_color(self, user_id: str) -> str:
    """
    Assign or retrieve a color for a user.
    """
    if user_id not in self.user_to_color and user_id != "global":
      # List of ANSI color codes (foreground colors)
      colors = [
        "\033[31m",  # red
        "\033[32m",  # green
        "\033[33m",  # yellow
        "\033[34m",  # blue
        "\033[35m",  # magenta
        "\033[36m",  # cyan
        "\033[91m",  # bright red
        "\033[92m",  # bright green
        "\033[93m",  # bright yellow
        "\033[94m",  # bright blue
        "\033[95m",  # bright magenta
        "\033[96m",  # bright cyan
      ]
      # Assign next available color
      color_idx = len(self.user_to_color) % len(colors)
      self.user_to_color[user_id] = colors[color_idx]
    return self.user_to_color.get(user_id, "")

  def _get_caller_info(self) -> str:
    """
    Get information about the calling function using inspect.
    Returns a formatted string with caller name in brackets, or empty string if caller info not found.
    """
    import inspect

    # Get caller info - go back 4 frames to get the actual caller
    # (skipping logging framework frames)
    caller_frame = None
    try:
      for i in range(4, 10):  # Look through several frames to find non-logging frame
        frame = inspect.currentframe()
        for _ in range(i):
          if frame is not None:
            frame = frame.f_back
        if frame and "logging" not in frame.f_code.co_filename:
          caller_frame = frame
          break
    except Exception:
      pass

    caller_info = ""
    if caller_frame:
      try:
        caller_name = caller_frame.f_code.co_name
        caller_info = f"[{caller_name}] "
      except Exception:
        pass
      finally:
        del caller_frame  # Avoid reference cycles

    return caller_info

  def _format_message(self, s: str, user_id: str) -> str:
    """
    Format a message with user information and color if applicable.
    """
    self._get_buffer(user_id)
    if user_id != "global" and s.strip():
      idx = self.user_to_idx[user_id]
      stage_idx = app.storage.user.get("stage_idx")
      block_idx = app.storage.user.get("block_idx")
      stage_name = app.storage.user.get("stage_name")
      block_name = app.storage.user.get("block_name")

      msg = f"user {idx}/{self.nusers}:  {user_id} "
      if block_idx is not None:
        if block_name is not None:
          msg += f"|block {block_idx}: {block_name}| "
        else:
          msg += f"|block {block_idx}| "
      if stage_idx is not None:
        if stage_name is not None:
          msg += f"|stage {stage_idx}: {stage_name}| "
        else:
          msg += f"|stage {stage_idx}| "
      msg += self._get_caller_info() + s

      if self.colored:
        color = self._get_color(user_id)
        reset = "\033[0m"  # Reset color
        return f"{color}{msg}{reset}"
      return msg
    return s

  def flush(self):
    """
    Flush all buffers, writing their contents to the console stream.
    """
    try:
      for user_id, buffer in self.buffers.items():
        output = buffer.getvalue()
        if output:
          output = self._format_message(output, user_id)
          self.console_stream.write(output)
          buffer.truncate(0)
          buffer.seek(0)
      self.console_stream.flush()
    except Exception as e:
      self.log_error(
        f"Error in UserAwareOutput.flush: {str(e)}\n{traceback.format_exc()}"
      )

  def log_error(self, error_message):
    """
    Log an error message to the error
    """
    with open(self.error_log, "a") as error_file:
      error_file.write(f"{error_message}\n")


class UserAwareHandler(logging.Handler):
  """
  A custom logging handler that writes log messages to a UserAwareOutput.

  This handler is designed to work with the UserAwareOutput class, which
  manages user-specific logging streams.

  Attributes:
      user_aware_output (UserAwareOutput): The output stream to write log messages to.
  """

  def __init__(self, user_aware_output):
    """
    Initialize the UserAwareHandler.

    Args:
        user_aware_output (UserAwareOutput): The output stream to write log messages to.
    """
    super().__init__()
    self.user_aware_output = user_aware_output

  def emit(self, record):
    """
    Emit a log record.

    This method formats the log record and writes it to the UserAwareOutput.
    If an error occurs during emission, it logs the error to a separate error log file.

    Args:
        record (logging.LogRecord): The log record to emit.
    """
    try:
      msg = self.format(record)
      self.user_aware_output.write(msg + "\n")
    except Exception as e:
      # Log the error to a separate error log file
      error_msg = f"Error in UserAwareHandler.emit: {str(e)}\n{traceback.format_exc()}"
      self.user_aware_output.log_error(error_msg)
      # Optionally, you can also print to stderr for immediate visibility
      print(error_msg, file=sys.stderr)

  def flush(self):
    """
    Flush the handler.

    This method ensures that any buffered log records are written out.
    """
    self.user_aware_output.flush()


class WatchfilesFilter(logging.Filter):
  def __init__(self, ignored_dirs=("data/", ".nicegui/")):
    super().__init__()
    self.ignored_dirs = ignored_dirs

  def filter(self, record):
    if hasattr(record, "msg") and isinstance(record.msg, str):
      if "changes detected:" in record.msg:
        changes = eval(record.msg.split(":", 1)[1])
        filtered_changes = [
          change for change in changes if not change[1].startswith(self.ignored_dirs)
        ]
        if not filtered_changes:
          return False
        record.msg = (
          f"{len(filtered_changes)} changes detected: {set(filtered_changes)}"
        )
    return True


@lru_cache(maxsize=None)
def get_logger(name: str):
  return logging.getLogger(name)


def setup_logging(
  log_dir: str,
  nicegui_storage_user_key: str = "user_id",
  watchfiles_logging: bool = False,
  log_filename_fn: Optional[Callable[[str, str], str]] = None,
  ignored_watchfiles_dirs: tuple = ("data/", ".nicegui/"),
  display_time: bool = False,
):
  """
  Set up logging configuration.

  Args:
  log_dir (str): Directory for log files.
  nicegui_storage_user_key (str): Key for user storage in NiceGUI.
  watchfiles_logging (bool): Flag to enable detailed watchfiles logging.
  ignored_watchfiles_dirs (tuple): Directories to ignore in watchfiles logging.

  Returns:
  logging.Logger: Configured root logger.
  """
  # Ensure log directory exists
  os.makedirs(log_dir, exist_ok=True)

  # Create UserAwareOutput instances
  user_aware_stdout = UserAwareOutput(
    sys.stdout,
    log_dir=log_dir,
    log_filename_fn=log_filename_fn,
    nicegui_storage_user_key=nicegui_storage_user_key,
  )
  user_aware_stderr = UserAwareOutput(
    sys.stderr,
    log_dir=log_dir,
    log_filename_fn=log_filename_fn,
    nicegui_storage_user_key=nicegui_storage_user_key,
  )

  # Redirect sys.stdout and sys.stderr
  sys.stdout = user_aware_stdout
  # sys.stderr = user_aware_stderr

  # Create and configure the root logger
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.INFO)

  # Remove any existing handlers
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

  # Create handlers using UserAwareOutput
  stdout_handler = UserAwareHandler(user_aware_stdout)

  # Set formatting for the handlers
  format = "%(name)s - %(levelname)s - %(message)s"
  formatter = logging.Formatter(
    format if not display_time else f"%(asctime)s - {format}",
  )
  stdout_handler.setFormatter(formatter)
  root_logger.addHandler(stdout_handler)

  # Configure watchfiles logger
  watchfiles_logger = logging.getLogger("watchfiles")
  watchfiles_logger.addFilter(WatchfilesFilter(ignored_dirs=ignored_watchfiles_dirs))

  if watchfiles_logging:
    watchfiles_logger.setLevel(logging.DEBUG)
    watchfiles_handler = logging.StreamHandler()
    watchfiles_handler.setFormatter(formatter)
    watchfiles_logger.addHandler(watchfiles_handler)
  else:
    watchfiles_logger.setLevel(logging.WARNING)

  return root_logger
