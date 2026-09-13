"""Three native-cell slice controls and cancellable, saved-frame playback."""

import asyncio
import json
import os
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

PLOT_LOCK = Lock()
CMAP = LinearSegmentedColormap.from_list(
    "native-cyan", ["#06111d", "#0e415c", "#2199bd", "#a9e1df", "#fff3b0"]
)
CSS = """
.gradio-container {background:#06111d !important; color:#e9f1f5 !important; max-width:1400px !important}
.prose {color:#e9f1f5 !important} footer {display:none !important}
"""


def render_slice(dataset, frame, plane, coordinate, quantity, n):
    vectors, _ = dataset.sample_plane(frame, plane, coordinate, quantity, n)
    values = np.linalg.norm(vectors, axis=-1)
    vmax = max(dataset.manifest["color_max"][quantity], 1e-15)
    h, v, fixed = PLANES[plane]
    with PLOT_LOCK:
        figure = Figure(figsize=(4.4, 4.4), dpi=120, facecolor="#06111d")
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, facecolor="#06111d")
        picture = axes.imshow(
            values.T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap=CMAP,
            norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=0, vmax=vmax),
            interpolation="nearest",
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
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def render_state(dataset, frame, x, y, z, quantity, n):
    if not np.isfinite(frame) or int(frame) != frame or int(n) != n:
        raise ValueError("Choose a saved frame and supported display resolution")
    frame, n = int(frame), int(n)
    images = [
        render_slice(dataset, frame, plane, coord, quantity, n)
        for plane, coord in zip(("yz", "xz", "xy"), (x, y, z))
    ]
    status = f"Frame {frame + 1}/{len(dataset.times)} · t = {dataset.times[frame]:.8f} · {quantity} · {n}² display samples per plane\n\nx = {x:.5f} · y = {y:.5f} · z = {z:.5f}\n\n{dataset.cache_status}"
    return (*images, status)


def make_app(dataset):
    with gr.Blocks(
        title="Native Navier–Stokes explorer", delete_cache=(3600, 3600)
    ) as app:
        gr.Markdown(
            "## Explore the saved simulation\nMove the three fixed-coordinate planes, then play through saved time. Velocity and forcing are in model units."
        )
        if not dataset.manifest["complete_history"]:
            gr.Markdown(
                f"**Prototype subset:** {len(dataset.times)} of {dataset.manifest['source_frames']} saved states. This is not the complete animation."
            )
        with gr.Row():
            quantity = gr.Radio(
                ["velocity", "force"], value="velocity", label="Quantity"
            )
            resolution = gr.Dropdown(
                [128, 256, 512, 1024], value=256, label="Display samples per axis"
            )
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
            label="t · saved frame",
            info="Actual simulation time is shown below. Saved times are not uniformly spaced.",
        )
        with gr.Row():
            previous = gr.Button("Previous frame")
            play = gr.Button("Play", variant="primary")
            pause = gr.Button("Pause")
            following = gr.Button("Next frame")
        status = gr.Markdown()
        gr.Markdown(
            "The viewer samples the finest native cell available at each location. Display resolution does not change solver resolution. No spatial or temporal interpolation, automatic peak tracking, or vector arrows. Playback targets five saved states per second and waits for data; it never skips a state.\n\nFinite manufactured-force experiment—not a singularity or convergence certificate."
        )
        inputs = [time, x, y, z, quantity, resolution]
        outputs = [*images, status]

        def render(*args):
            return render_state(dataset, *args)

        async def animate(frame, *args):
            start = 0 if int(frame) == len(dataset.times) - 1 else int(frame)
            for index in range(start, len(dataset.times)):
                started = monotonic()
                result = await asyncio.to_thread(render, index, *args)
                yield index, *result
                await asyncio.sleep(max(0, 0.2 - (monotonic() - started)))

        playback = play.click(
            animate,
            inputs,
            [time, *outputs],
            api_name=False,
            concurrency_id="native-playback",
            concurrency_limit=4,
            show_progress="hidden",
        )
        pause.click(fn=None, cancels=[playback], queue=False)
        # One ordered event for all controls: a slower old request must not
        # overwrite a newer quantity/coordinate choice. Playback uses a separate
        # bounded group so one long animation cannot monopolize manual controls.
        gr.on(
            triggers=[
                x.input,
                y.input,
                z.input,
                time.input,
                quantity.change,
                resolution.change,
            ],
            fn=render,
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
                return index, *render(index, *args)

            button.click(
                advance,
                inputs,
                [time, *outputs],
                cancels=[playback],
                concurrency_id="native-manual",
                concurrency_limit=1,
                show_progress="hidden",
            )
        app.load(
            render,
            inputs,
            outputs,
            api_name="slice",
            concurrency_id="native-manual",
            concurrency_limit=1,
        )
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
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
        css=CSS,
        footer_links=[],
        show_error=False,
    )
