let completedRun;
const selections = { flow: "midplane", force: "midplane" };
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function asset(record) {
  const url = new URL(record.path, window.location.href);
  if (url.origin !== window.location.origin || !record.path.startsWith("media/oliver-")) {
    throw new Error("Expected an Oliver media asset");
  }
  url.searchParams.set("v", record.sha256.slice(0, 12));
  return url.href;
}

function showView(kind, view) {
  if (!completedRun) return;
  selections[kind] = view;
  const media = completedRun.media[`${kind}_${view}`];
  const video = document.querySelector(`#${kind}-video`);
  const firstLoad = !video.hasAttribute("src");
  const position = firstLoad ? 0 : video.currentTime;
  const playing = firstLoad || !video.paused;
  document.querySelectorAll(`[data-field="${kind}"]`).forEach((tab) => {
    const selected = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected) document.querySelector(`#${kind}-panel-2d`).setAttribute("aria-labelledby", tab.id);
  });
  document.querySelector(`#${kind}-view-note`).textContent = view === "peak"
    ? "The x–z plane stays at y = 0. The x–y plane follows the height of the full 3D peak speed at each saved time, labeled z. This is a moving slice, not a tracked fluid parcel."
    : "Fixed planes: y = 0 and z = 0. No vector arrows.";
  document.querySelector(`#${kind}-gif`).href = asset(media.gif);
  document.querySelector(`#${kind}-mp4`).href = asset(media.mp4);
  video.poster = asset(media.png);
  video.onloadedmetadata = () => {
    if (Number.isFinite(position) && position > 0) video.currentTime = Math.min(position, video.duration - 0.02);
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
  if (playing && !reducedMotion.matches) video.play().catch(() => {});
}

for (const tab of document.querySelectorAll("[role=tab]")) {
  tab.addEventListener("click", () => showView(tab.dataset.field, tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const view = event.key === "Home" ? "midplane" : event.key === "End" ? "peak"
      : selections[tab.dataset.field] === "midplane" ? "peak" : "midplane";
    const next = document.querySelector(`[data-field="${tab.dataset.field}"][data-view="${view}"]`);
    showView(tab.dataset.field, view);
    next.focus();
  });
}

fetch("data/best.json", { cache: "no-store" })
  .then((response) => { if (!response.ok) throw new Error("Completion record unavailable"); return response.json(); })
  .then((run) => {
    if (run.kind !== "completed-native-best" || run.status !== "completed" || !run.validated
        || run.saved_frames !== run.frame_times.length || run.frame_times[0] !== 0
        || run.frame_times.at(-1) !== run.parameters.end || run.vector_arrows !== false) {
      throw new Error("Invalid completed native result");
    }
    completedRun = run;
    document.querySelector("#cell-count").textContent = run.active_cells.toLocaleString();
    document.querySelector("#frame-count").textContent = run.saved_frames.toLocaleString();
    document.querySelector("#peak-speed").textContent = run.diagnostics.peak_speed.toFixed(3);
    document.querySelector("#speed-time").textContent = `Model units · full-field diagnostic at t = ${run.diagnostics.time}`;
    document.querySelectorAll("[data-clip-range]").forEach((node) => {
      node.textContent = `all ${run.saved_frames} saved frames · t = 0 to ${run.parameters.end}`;
    });
    showView("flow", "midplane");
    showView("force", "midplane");
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
