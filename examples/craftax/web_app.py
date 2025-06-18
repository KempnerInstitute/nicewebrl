import nicewebrl
import os
from importlib.util import find_spec
import shutil

def restore_texture_cache_if_needed():
  """Restore texture cache files from local cache if they don't exist in the package directory."""
  # Get paths for texture cache files
  original_constants_directory = os.path.join(
    os.path.dirname(find_spec("craftax.craftax.constants").origin), "assets"
  )
  TEXTURE_CACHE_FILE = os.path.join(original_constants_directory, "texture_cache.pbz2")

  # Local cache paths
  cache_dir = "craftax_cache"
  source_cache = os.path.join(cache_dir, "texture_cache.pbz2")

  # Create the destination directories if they don't exist
  os.makedirs(os.path.dirname(TEXTURE_CACHE_FILE), exist_ok=True)

  # Copy texture cache files if needed
  if not os.path.exists(TEXTURE_CACHE_FILE) and os.path.exists(source_cache):
    print(
      f"Restoring texture cache from {source_cache} to {TEXTURE_CACHE_FILE}"
    )
    shutil.copy2(source_cache, TEXTURE_CACHE_FILE)
    print("Regular cache file restored successfully!")
  else:
    print(f"{TEXTURE_CACHE_FILE} already exists.")

nicewebrl.run(
    storage_secret="a_very_secret_key_for_testing_only_12345",
    experiment_file="examples/craftax/experiment_structure.py",
    title="NiceWebRL Craftax Experiment",
    reload=False, 
    on_startup_fn=restore_texture_cache_if_needed, # e.g. restore cache
    on_termination_fn=None, # e.g. saving and uploading to cloud storage
)