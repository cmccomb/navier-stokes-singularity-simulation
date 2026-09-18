const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function asset(record) {
  const url = new URL(record.path, window.location.href);
  if (url.origin !== window.location.origin || !record.path.startsWith("media/kay-")) {
    throw new Error("Expected a Kay media asset");
  }
  url.searchParams.set("v", record.sha256.slice(0, 12));
  return url.href;
}

const selections = {};
const descriptions = {
  flow: "Cutaway speed surfaces + native-core streamlines. Cyan rises; gold descends. Color shows direction, not speed. These are instantaneous streamlines, not particle tracks.",
  force: "Cutaway force surfaces + native-core volume. Color and opacity use a fixed logarithmic mapping; opacity is not material density.",
};

function showMovie(kind, views, detail = false) {
  const media = views.media[kind].files;
  const video = document.querySelector(`#${kind}-video`);
  const previous = selections[kind];
  const seek = previous ? (previous.pendingTime ?? video.currentTime) : 0;
  const playing = previous ? (previous.pendingTime !== null ? previous.playing : !video.paused) : !reducedMotion.matches;
  const selection = { detail, pendingTime: seek, playing };
  selections[kind] = selection;
  video.pause();
  document.querySelector(`#${kind}-pending`).textContent = "Loading the selected movie at the same saved time.";
  document.querySelector(`#${kind}-pending`).hidden = false;
  document.querySelector(`#${kind}-figure`).classList.toggle("detail-view", detail);
  document.querySelector(`#${kind}-view-note`).textContent = detail ? descriptions[kind] : "Fixed planes: y = 0 and z = 0. No vector arrows.";
  document.querySelector(`#${kind}-scroll-hint`).textContent = detail ? "Scroll sideways: cutaway → native core · fullscreen for detail" : "Scroll sideways: x–y → x–z → isometric";
  for (const view of ["views", "detail"]) {
    document.querySelector(`#${kind}-${view}`).setAttribute("aria-pressed", String(detail === (view === "detail")));
  }
  video.setAttribute("aria-label", detail ? descriptions[kind] : `${kind === "flow" ? "Velocity" : "Force"} magnitude: x-y slice, x-z slice, then isometric surfaces, synchronized through all saved times`);
  document.querySelector(`#${kind}-gif`).href = asset(media.gif);
  document.querySelector(`#${kind}-mp4`).href = asset(media.mp4);
  video.poster = asset(media.png);
  video.onloadedmetadata = () => {
    if (selections[kind] !== selection) return;
    video.currentTime = Math.min(seek, Math.max(0, video.duration - 0.01));
    selection.pendingTime = null;
    if (playing) video.play().catch(() => {});
  };
  video.onloadeddata = () => { document.querySelector(`#${kind}-pending`).hidden = true; };
  video.onerror = () => {
    const note = document.querySelector(`#${kind}-pending`);
    note.textContent = "Video unavailable. The GIF download is available below.";
    note.hidden = false;
  };
  video.src = asset(media.mp4);
  video.muted = true;
  document.querySelector(`#${kind}-figure`).hidden = false;
}

function enableDetails(run, views) {
  fetch("data/detail-view.json", { cache: "no-store" })
    .then((response) => { if (!response.ok) throw new Error("Detail record unavailable"); return response.json(); })
    .then((detail) => {
      if (detail.kind !== "native-detail-3d" || !detail.validated || !detail.complete_history
          || detail.saved_frames !== run.saved_frames
          || detail.source_record_sha256 !== run.source_record_sha256
          || JSON.stringify(detail.frame_times) !== JSON.stringify(run.frame_times)
          || ["flow", "force"].some((kind) => detail.media[kind].saved_states !== run.saved_frames
            || detail.media[kind].duration_seconds !== views.media[kind].duration_seconds)) {
        throw new Error("3D detail media must match the complete native history and playback clock");
      }
      for (const kind of ["flow", "force"]) {
        document.querySelector(`#${kind}-switch`).hidden = false;
        for (const mode of ["views", "detail"]) {
          document.querySelector(`#${kind}-${mode}`).onclick = () => {
            const isDetail = mode === "detail";
            if (selections[kind].detail !== isDetail) showMovie(kind, isDetail ? detail : views, isDetail);
          };
        }
      }
    })
    .catch((error) => console.warn("3D detail unavailable; complete three-view movies remain available.", error));
}

Promise.all(["data/best.json", "data/three-view.json"].map((path) =>
  fetch(path, { cache: "no-store" })
    .then((response) => { if (!response.ok) throw new Error("Completion record unavailable"); return response.json(); })
))
  .then(([run, views]) => {
    if (run.kind !== "completed-native-best" || run.status !== "completed" || !run.validated
        || run.saved_frames !== run.frame_times.length || run.frame_times[0] !== 0
        || run.frame_times.at(-1) !== run.parameters.end || run.vector_arrows !== false) {
      throw new Error("Invalid completed native result");
    }
    if (views.kind !== "native-three-view" || !views.validated || !views.complete_history
        || views.saved_frames !== run.saved_frames
        || views.source_record_sha256 !== run.source_record_sha256
        || JSON.stringify(views.frame_times) !== JSON.stringify(run.frame_times)
        || JSON.stringify(views.view_order) !== '["xy","xz","isometric"]'
        || ["flow", "force"].some((kind) => views.media[kind].saved_states !== run.saved_frames)) {
      throw new Error("Three-view media must match the complete native history");
    }
    document.querySelector("#cell-count").textContent = run.active_cells.toLocaleString();
    document.querySelector("#frame-count").textContent = run.saved_frames.toLocaleString();
    document.querySelector("#peak-speed").textContent = run.diagnostics.peak_speed.toFixed(3);
    document.querySelector("#speed-time").textContent = `Model units · full-field diagnostic at t = ${run.diagnostics.time}`;
    document.querySelectorAll("[data-clip-range]").forEach((node) => {
      node.textContent = `all ${run.saved_frames} saved frames · t = 0 to ${run.parameters.end}`;
    });
    showMovie("flow", views);
    showMovie("force", views);
    enableDetails(run, views);
  })
  .catch((error) => {
    for (const kind of ["flow", "force"]) {
      const note = document.querySelector(`#${kind}-pending`);
      note.textContent = "Movie metadata unavailable. Use the direct GIF or MP4 download below.";
      note.hidden = false;
      document.querySelector(`#${kind}-figure`).hidden = false;
    }
    console.error(error);
  });
