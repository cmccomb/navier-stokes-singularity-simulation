"""Fixed-resolution native slices with bounded, cancellable playback buffering."""

import asyncio
import json
import os
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import monotonic

import gradio as gr
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.figure import Figure

try:
    from .reader import PLANES, NativeDataset
except ImportError:
    from reader import PLANES, NativeDataset

DISPLAY_SAMPLES = 256
PLAYBACK_INTERVAL = 0.2
PLAYBACK_BUFFER = 3
PLOT_LOCK = Lock()
CMAP = LinearSegmentedColormap.from_list(
    "native-cyan", ["#06111d", "#0e415c", "#2199bd", "#a9e1df", "#fff3b0"]
)
CSS = """
.gradio-container {background:#06111d !important; color:#e9f1f5 !important; max-width:1400px !important; padding:0 !important; min-height:0 !important}
.prose {color:#e9f1f5 !important} footer {display:none !important}
"""
HEIGHT_JS = """() => {
    if (window.parent === window) return;
    const container = document.querySelector('.gradio-container main > .column');
    if (!container) return;
    const report = () => window.parent.postMessage({
        type: 'native-explorer-height',
        height: Math.ceil(container.getBoundingClientRect().bottom + window.scrollY + 16)
    }, '*');
    const observer = new ResizeObserver(report);
    observer.observe(container);
    window.addEventListener('pagehide', () => observer.disconnect(), {once: true});
    report();
}"""


def dark_theme():
    """Use the same Gradio dark palette in standalone and embedded views."""
    theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")
    return theme.set(
        **{
            key.removesuffix("_dark"): value
            for key, value in theme.to_dict()["theme"].items()
            if key.endswith("_dark")
        }
    )


@lru_cache(maxsize=12)
def _plot_template(plane, coordinate, vmax):
    """Retain the static axes and color scale; called only under PLOT_LOCK."""
    h, v, fixed = PLANES[plane]
    figure = Figure(figsize=(4.4, 4.4), dpi=120, facecolor="#06111d")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, facecolor="#06111d")
    picture = axes.imshow(
        np.zeros((DISPLAY_SAMPLES, DISPLAY_SAMPLES)),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=CMAP,
        norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=0, vmax=vmax),
        interpolation="nearest",
        animated=True,
    )
    axes.set(
        xlabel="xyz"[h],
        ylabel="xyz"[v],
        title=f"{plane[0]}–{plane[1]} · {'xyz'[fixed]} = {coordinate:.5f}",
    )
    axes.tick_params(colors="#9fb3c2")
    for label in (axes.xaxis.label, axes.yaxis.label, axes.title):
        label.set_color("#e9f1f5")
    colorbar = figure.colorbar(picture, ax=axes, fraction=0.045, pad=0.035)
    colorbar.ax.tick_params(colors="#9fb3c2")
    figure.tight_layout()
    canvas.draw()
    return figure, canvas, axes, picture, canvas.copy_from_bbox(figure.bbox)


def render_slice(dataset, frame, plane, coordinate, quantity):
    vectors, _ = dataset.sample_plane(
        frame, plane, coordinate, quantity, DISPLAY_SAMPLES
    )
    values = np.linalg.norm(vectors, axis=-1)
    vmax = max(dataset.manifest["color_max"][quantity], 1e-15)
    with PLOT_LOCK:
        figure, canvas, axes, picture, background = _plot_template(
            plane, coordinate, vmax
        )
        canvas.restore_region(background)
        picture.set_data(values.T)
        axes.draw_artist(picture)
        canvas.blit(figure.bbox)
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def render_state(dataset, frame, x, y, z, quantity):
    if (
        not isinstance(frame, (int, float, np.integer, np.floating))
        or isinstance(frame, bool)
        or not np.isfinite(frame)
        or int(frame) != frame
        or not 0 <= frame < len(dataset.times)
    ):
        raise ValueError("Choose a saved frame")
    frame = int(frame)
    images = [
        render_slice(dataset, frame, plane, coord, quantity)
        for plane, coord in zip(("yz", "xz", "xy"), (x, y, z))
    ]
    label = (
        f"Saved frame {frame + 1}/{len(dataset.times)} · t = {dataset.times[frame]:.8f}"
    )
    return (*images, label)


async def buffered_frames(render, start, stop, args):
    """Prefill three frames, then produce sequentially while the client displays."""
    if start >= stop:
        return
    queue = asyncio.Queue(maxsize=PLAYBACK_BUFFER)
    ready = asyncio.Event()

    async def produce():
        try:
            for index in range(start, stop):
                result = await asyncio.to_thread(render, index, *args)
                await queue.put((index, result, None))
                if queue.qsize() >= min(PLAYBACK_BUFFER, stop - start):
                    ready.set()
        except Exception as error:  # noqa: BLE001 - forward worker errors to the client
            await queue.put((None, None, error))
            ready.set()

    producer = asyncio.create_task(produce())
    try:
        await ready.wait()
        for _ in range(start, stop):
            index, result, error = await queue.get()
            if error is not None:
                raise error
            started = monotonic()
            yield index, result
            await asyncio.sleep(max(0, PLAYBACK_INTERVAL - (monotonic() - started)))
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


def make_app(dataset):
    # Bounded across this app's visitors; revisits reuse already-rendered planes.
    render = lru_cache(maxsize=12)(lambda *args: render_state(dataset, *args))

    def update(frame, result):
        *images, label = result
        return (*images, gr.update(value=int(frame), label=label))

    def render_controls(frame, *args):
        return update(frame, render(frame, *args))

    with gr.Blocks(
        title="Native Navier–Stokes explorer", delete_cache=(3600, 3600)
    ) as app:
        quantity = gr.Radio(["velocity", "force"], value="velocity", label="Quantity")
        step = min(min(level["spacing"]) for level in dataset.levels)
        with gr.Row():
            x = gr.Slider(-1, 1, value=0, step=step / 2, label="x · YZ slice")
            y = gr.Slider(-1, 1, value=0, step=step / 2, label="y · XZ slice")
            z = gr.Slider(-1, 1, value=0, step=step / 2, label="z · XY slice")
        with gr.Row():
            images = [
                gr.Image(
                    label=name,
                    interactive=False,
                    format="png",
                    type="numpy",
                    buttons=["download", "fullscreen"],
                    alt_text=f"Native {name} magnitude slice",
                )
                for name in ("YZ", "XZ", "XY")
            ]
        time = gr.Slider(
            0,
            len(dataset.times) - 1,
            value=0,
            step=1,
            label=f"Saved frame 1/{len(dataset.times)} · t = {dataset.times[0]:.8f}",
        )
        with gr.Row():
            previous = gr.Button("Previous frame")
            play = gr.Button("Play", variant="primary")
            pause = gr.Button("Pause")
            following = gr.Button("Next frame")
        inputs = [time, x, y, z, quantity]
        outputs = [*images, time]

        async def animate(frame, *args):
            start = 0 if int(frame) == len(dataset.times) - 1 else int(frame)
            async for index, result in buffered_frames(
                render, start, len(dataset.times), args
            ):
                yield update(index, result)

        playback = play.click(
            animate,
            inputs,
            outputs,
            api_name=False,
            concurrency_id="native-playback",
            concurrency_limit=4,
            show_progress="minimal",
        )
        pause.click(fn=None, cancels=[playback], queue=False)
        # Only user input triggers a render, so updating the time control during
        # playback cannot enqueue a duplicate request or cancel the animation.
        gr.on(
            triggers=[x.input, y.input, z.input, time.input, quantity.change],
            fn=render_controls,
            inputs=inputs,
            outputs=outputs,
            cancels=[playback],
            trigger_mode="always_last",
            concurrency_id="native-manual",
            concurrency_limit=1,
            show_progress="hidden",
        )
        for button, increment in ((previous, -1), (following, 1)):

            def advance(frame, *args, amount=increment):
                index = max(0, min(len(dataset.times) - 1, int(frame) + amount))
                return render_controls(index, *args)

            button.click(
                advance,
                inputs,
                outputs,
                cancels=[playback],
                concurrency_id="native-manual",
                concurrency_limit=1,
                show_progress="hidden",
            )
        app.load(
            render_controls,
            inputs,
            outputs,
            api_name="slice",
            concurrency_id="native-manual",
            concurrency_limit=1,
        )
        app.load(fn=None, js=HEIGHT_JS, queue=False)
    return app.queue(max_size=16, default_concurrency_limit=4)


if __name__ == "__main__":
    config_path = Path(__file__).with_name("dataset.json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    dataset = NativeDataset(
        folder=os.environ.get("NS_DATASET_LOCAL"),
        repo_id=os.environ.get("NS_DATASET_REPO", config.get("repo_id")),
        revision=os.environ.get("NS_DATASET_REVISION", config.get("revision")),
        prefix=os.environ.get("NS_DATASET_PREFIX", config.get("prefix", "")),
    )
    if os.environ.get("SPACE_ID"):
        dataset.warm_archive_cache()
    make_app(dataset).launch(
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        server_port=int(
            os.environ.get("PORT", "7860" if os.environ.get("SPACE_ID") else "8877")
        ),
        share=False,
        max_threads=4,
        theme=dark_theme(),
        css=CSS,
        footer_links=[],
        show_error=False,
    )
