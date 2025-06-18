import os
import asyncio
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
import importlib.util

from nicegui import app, ui, Client
from fastapi import APIRouter
from tortoise import Tortoise

import nicewebrl
from nicewebrl.experiment import SimpleExperiment
from nicewebrl import stages
from nicewebrl.utils import get_user_lock
from nicewebrl.logging import setup_logging, get_logger

### --- Globals ---
module_logger: Optional[Any] = None
experiment_obj: Optional[SimpleExperiment] = None

env_loaded: asyncio.Event = asyncio.Event()
load_start_time: Optional[datetime] = None
load_error: Optional[str] = None

DATA_DIR_DEFAULT = "data"
DATABASE_FILE_DEFAULT = "db.sqlite"

### --- Status Router ---
router = APIRouter()

@router.get("/status")
async def get_status():
    current_time = datetime.now()
    load_duration = (current_time - load_start_time).total_seconds() if load_start_time else None
    return {
        "loaded": env_loaded.is_set(),
        "load_duration": load_duration,
        "load_error": load_error,
        "load_start_time": load_start_time.isoformat() if load_start_time else None,
    }

app.include_router(router)

### --- Loading Screen ---
def show_loading_screen():
    if not env_loaded.is_set():
        with ui.card().classes("fixed-center shadow-lg").style("width: 60vw; max-height: 80vh; padding: 2rem;"):
            ui.label("🚀 Loading Your Experiment...").classes("text-h4 text-center q-mb-md")
            ui.label("This may take up to 5 minutes. Please don’t refresh or close this tab.").classes("text-subtitle1 text-grey q-mb-xl")

            # Animated spinner
            ui.spinner(size="lg", color="primary").classes("q-mb-md")

            # Dynamic status info
            elapsed_label = ui.label("Time elapsed: 0 seconds").classes("text-subtitle1 text-grey-8")
            status_label = ui.label("Current status: Initializing...").classes("text-subtitle1 text-grey-8")
            error_label = ui.label().classes("text-negative text-subtitle1")

            start_time = time.time()

            def update():
                seconds = int(time.time() - start_time)
                elapsed_label.text = f"⏳ Time elapsed: {seconds} seconds"
                if load_error:
                    error_label.text = f"⚠️ Error: {load_error}"
                    status_label.text = "❌ Status: Failed to load"
            ui.timer(1.0, update)

            # Motivational quote
            ui.separator()
            ui.label("“Great things take time — thanks for your patience!”").classes("italic text-center text-grey")

            # JS to auto-poll /status
            ui.add_body_html("""
            <script>
            async function checkStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    if (data.loaded) window.location.reload();
                    else setTimeout(checkStatus, 1000);
                } catch {
                    setTimeout(checkStatus, 1000);
                }
            }
            checkStatus();
            </script>
            """)

### --- Footer ---
async def footer(footer_container):
    """Add user information and progress bar to the footer"""
    with footer_container:
        # Info row
        with ui.row().classes("w-full justify-center gap-6 items-center"):
            ui.label().bind_text_from(app.storage.user, "user_id", lambda v: f"User ID: {v}").classes("text-sm text-gray-500")

            def text_display(v):
                stage_idx = max(experiment_obj.num_stages, int(v) + 1)
                return f"Stage: {stage_idx}/{experiment_obj.num_stages}"

            ui.label().bind_text_from(app.storage.user, "stage_idx", text_display).classes("text-lg font-bold text-indigo-600")

            ui.label().bind_text_from(
                app.storage.user, "session_duration",
                lambda v: f"Minutes passed: {int(v)}"
            ).classes("text-lg font-medium text-green-700")

        # Fullscreen button row
        with ui.row().classes("w-full justify-center mt-3"):
            ui.button(
                "Toggle fullscreen",
                icon="fullscreen",
                on_click=nicewebrl.utils.toggle_fullscreen,
            ).props("flat").classes("text-blue-600 hover:text-blue-800")

### --- Experiment Loader ---
async def load_experiment(file: str):
    global experiment_obj, load_error
    try:
        loop = asyncio.get_running_loop()
        module_name = os.path.splitext(os.path.basename(file))[0]

        def _sync_load():
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec or loader from {file}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # Blocking call
            experiment = getattr(mod, "experiment", None)
            if not isinstance(experiment, SimpleExperiment):
                raise TypeError("Expected a SimpleExperiment instance named 'experiment'")
            return experiment

        experiment = await loop.run_in_executor(None, _sync_load)
        experiment_obj = experiment
        module_logger.info(f"Loaded experiment: {experiment_obj.name}")

    except Exception as e:
        load_error = str(e)
        raise
    finally:
        env_loaded.set()

### --- Lifecycle Hooks ---
async def init_db(db_url: str):
    await Tortoise.init(db_url=db_url, modules={"models": ["nicewebrl.stages"]})
    await Tortoise.generate_schemas()

async def close_db():
    await Tortoise.close_connections()

### --- Check If Over ---
async def check_if_over(container, episode_limit=60):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    # define custom behavior on time-out
    pass

### --- Experiment Runner ---
async def run_stage(stage: stages.Stage, container: ui.element):
    event = asyncio.Event()

    async def signal():
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                event.set()

    await stage.set_user_data(stage_completion_signal_from_event=signal)
    await stage.activate(container)

    if stage.get_user_data("finished", False):
        event.set()

    if stage.next_button:
        with container:
            ui.button("Next", on_click=lambda: asyncio.create_task(stage.handle_button_press(container))).on("click", signal)

    await event.wait()
    await stage.set_user_data(stage_completion_signal_from_event=None)

### --- Flow ---
async def start_experiment(meta_container: ui.element, stage_container: ui.element, on_termination_fn: Optional[Callable] = None):
    user_id = app.storage.user.get("seed", "unknown")
    await experiment_obj.initialize()

    # Register keydown handler
    ui.on("keydown", lambda e: global_handle_key_press(e, stage_container))

    while experiment_obj.not_finished():
        stage = await experiment_obj.get_stage()
        nicewebrl.clear_element(stage_container)
        await run_stage(stage, stage_container)
        if isinstance(stage, stages.EnvStage):
            await stage.finish_saving_user_data()
        await experiment_obj.advance_stage()

    if on_termination_fn:
        result = on_termination_fn()
        if asyncio.iscoroutine(result): await result

    nicewebrl.clear_element(meta_container)
    with meta_container:
        ui.markdown("# Experiment Over\nThank you for your participation.")

async def global_handle_key_press(e, container):
    module_logger.info(f"Key press detected: {e}")
    if experiment_obj.finished(): return
    stage = await experiment_obj.get_stage()
    if stage.get_user_data("finished", False): return
    await stage.handle_key_press(e, container)
    fn = stage.get_user_data("stage_completion_signal_from_event")
    if fn: await fn()

### --- Main Entrypoint ---
def run(
    storage_secret: str,
    experiment_file: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    title: str = "NiceWebRL",
    data_dir: str = DATA_DIR_DEFAULT,
    database_file: str = DATABASE_FILE_DEFAULT,
    reload: bool = False,
    on_startup_fn: Optional[Callable] = None,
    on_termination_fn: Optional[Callable] = None,
):
    global module_logger, load_start_time

    setup_logging(data_dir, nicegui_storage_user_key="seed")
    module_logger = get_logger("run_experiment")
    db_path = os.path.abspath(os.path.join(data_dir, database_file))

    async def _on_startup():
        await init_db(f"sqlite:///{db_path}")
        if on_startup_fn:
            result = on_startup_fn()
            if asyncio.iscoroutine(result): await result
        global load_start_time
        load_start_time = datetime.now()
        asyncio.create_task(load_experiment(experiment_file))

    app.on_startup(_on_startup)
    app.on_shutdown(close_db)

    @ui.page("/")
    async def index(client: Client):
        nicewebrl.initialize_user()

        if not env_loaded.is_set():
            show_loading_screen()
            return

        await env_loaded.wait()  # Ensure experiment_obj is ready

        if load_error:
            with ui.card().classes("fixed-center"):
                ui.label("Failed to load experiment.")
                ui.markdown(f"Error: `{load_error}`")
            return

        # Setup container and key handler
        basic_javascript_file = nicewebrl.basic_javascript_file()
        with open(basic_javascript_file) as f:
            ui.add_body_html("<script>" + f.read() + "</script>")

        card = (
            ui.card(align_items=["center"])
            .classes("fixed-center")
            .style(
            "max-width: 90vw;"  # Set the max width of the card
            "max-height: 90vh;"  # Ensure the max height is 90% of the viewport height
            "overflow: auto;"  # Allow scrolling inside the card if content overflows
            "display: flex;"  # Use flexbox for centering
            "flex-direction: column;"  # Stack content vertically
            "justify-content: flex-start;"
            "align-items: center;"
            )
        )

        with card:
            meta_container = ui.column()
            # --- Layout containers ---
            with meta_container.style("align-items: center;"):
                display_container = ui.row()
                with display_container.style("align-items: center;"):
                    # Stage and LLM experiment areas
                    stage_container = ui.column()

                    # Timer to check if experiment timed out
                    ui.timer(
                        interval=1,
                        callback=lambda: check_if_over(
                            episode_limit=1,
                            container=stage_container,
                        ),
                    )

            footer_container = ui.row().style("margin-top: 20px; justify-content: center;")

            # --- Footer and experiment logic ---
            with meta_container.style("align-items: center;"):
                await footer(footer_container)
                with display_container.style("align-items: center;"):
                    await start_experiment(display_container, stage_container, on_termination_fn)

    ui.run(storage_secret=storage_secret, host=host, port=port, reload=reload, title=title)