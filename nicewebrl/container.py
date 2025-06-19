import dataclasses
import uuid
from asyncio import Lock

from nicegui import app, ui

from nicewebrl.utils import get_user_lock


@dataclasses.dataclass
class Container:
  name: str = None

  def __post_init__(self):
    self._lock = Lock()  # Add lock for thread safety
    self._user_locks = {}  # Dictionary to store per-user locks
    if self.name is None:
      self.name = f"container_{uuid.uuid4().hex}"

  def get_data(self):
    return app.storage.user.get(f"{self.name}_data", {})

  def get_user_data(self, key, value=None):
    return self.get_data().get(key, value)

  async def set_user_data(self, **kwargs):
    data = self.get_data()
    data.update(kwargs)
    async with get_user_lock():
      app.storage.user[f"{self.name}_data"] = data
