const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function asset(record) {
  const url = new URL(record.path, window.location.href);
  if (url.origin !== window.location.origin || !/^media\/oliver-(flow|force)-views\.(gif|mp4|png)$/.test(record.path)) {
    throw new Error("Expected a current Oliver media asset");
  }
  url.searchParams.set("v", record.sha256.slice(0, 12));
  return url.href;
}

function showMovie(kind, views) {
  const media = views.media[kind].files;
  const video = document.querySelector(`#${kind}-video`);
  const note = document.querySelector(`#${kind}-pending`);
  document.querySelector(`#${kind}-gif`).href = asset(media.gif);
  document.querySelector(`#${kind}-mp4`).href = asset(media.mp4);
  video.poster = asset(media.png);
  video.muted = true;
  video.onloadedmetadata = () => {
    if (!reducedMotion.matches) video.play().catch(() => {});
  };
  video.onloadeddata = () => { note.hidden = true; };
  video.onerror = () => {
    note.textContent = "Video unavailable. The GIF download is available below.";
    note.hidden = false;
  };
  video.src = asset(media.mp4);
  document.querySelector(`#${kind}-figure`).hidden = false;
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
    document.querySelectorAll("[data-clip-range]").forEach((node) => {
      node.textContent = `all ${run.saved_frames} saved frames · t = 0 to ${run.parameters.end}`;
    });
    showMovie("flow", views);
    showMovie("force", views);
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
