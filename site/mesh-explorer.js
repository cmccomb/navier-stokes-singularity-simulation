"use strict";
(() => {
  const wholeEmbed = new URLSearchParams(location.search).get("embed") === "whole";
  if (wholeEmbed) document.documentElement.classList.add("whole-embed");
  const canvas = document.getElementById("mesh-canvas"),
    ctx = canvas.getContext("2d");
  const caption = document.getElementById("mesh-caption"),
    status = document.getElementById("mesh-status");
  let mesh,
    segments = [],
    mode = "whole",
    yaw = -0.65,
    pitch = 0.48,
    zoom = 1,
    center = [0, 0, 0],
    extent = 1;
  let scheduled = false,
    drag = null,
    width = 1,
    height = 1;
  const edgePairs = [
    [0, 1],
    [0, 2],
    [0, 4],
    [1, 3],
    [1, 5],
    [2, 3],
    [2, 6],
    [3, 7],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
  ];
  function cube(lo, hi, color, weight = 1) {
    const corners = Array.from({ length: 8 }, (_, i) =>
      lo.map((v, a) => (i & (1 << a) ? hi[a] : v)),
    );
    return edgePairs.map(([a, b]) => ({
      a: corners[a],
      b: corners[b],
      color,
      weight,
    }));
  }
  function project(p) {
    const [x, y, z] = p.map((v, i) => (v - center[i]) / extent);
    const u = x * Math.cos(yaw) - y * Math.sin(yaw),
      depth = x * Math.sin(yaw) + y * Math.cos(yaw);
    const v = z * Math.cos(pitch) - depth * Math.sin(pitch),
      w = z * Math.sin(pitch) + depth * Math.cos(pitch);
    const scale = Math.min(width, height) * (wholeEmbed ? 0.275 : 0.29) * zoom;
    return [width / 2 + u * scale, height / 2 - v * scale, w];
  }
  function draw() {
    scheduled = false;
    ctx.clearRect(0, 0, width, height);
    const projected = segments
      .map((s) => ({ ...s, p: project(s.a), q: project(s.b) }))
      .sort((a, b) => a.p[2] + a.q[2] - b.p[2] - b.q[2]);
    for (const s of projected) {
      if (
        Math.max(s.p[0], s.q[0]) < 0 ||
        Math.min(s.p[0], s.q[0]) > width ||
        Math.max(s.p[1], s.q[1]) < 0 ||
        Math.min(s.p[1], s.q[1]) > height
      )
        continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.weight;
      ctx.globalAlpha = mode === "cells" ? 0.7 : 0.85;
      ctx.beginPath();
      ctx.moveTo(s.p[0], s.p[1]);
      ctx.lineTo(s.q[0], s.q[1]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = "14px Arial";
    ctx.fillStyle = "#d1e4ee";
    if (mode !== "cells") {
      "xyz".split("").forEach((axis, i) => {
        const p = [0, 0, 0];
        p[i] = extent * 1.14;
        const v = project(p);
        ctx.fillText(axis, v[0], v[1]);
      });
    }
  }
  function redraw() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(draw);
    }
  }
  function resize() {
    const box = canvas.getBoundingClientRect(),
      dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = box.width;
    height = box.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  }
  function select(next) {
    if (!mesh || (wholeEmbed && next !== "whole")) return;
    mode = next;
    zoom = 1;
    segments = [];
    center = [0, 0, 0];
    extent = mesh.levels[0].half;
    for (const name of ["whole", "core", "cells"])
      document
        .getElementById(name)
        .setAttribute("aria-pressed", String(name === mode));
    if (mode === "cells") {
      center = mesh.detail_bounds.map(([lo, hi]) => (lo + hi) / 2);
      extent = Math.max(...mesh.detail_bounds.map(([lo, hi]) => (hi - lo) / 2));
      // Shared cell edges are deduplicated, preserving the finer interface edges.
      const unique = new Map();
      for (const [level, x, y, z, dx] of mesh.cells) {
        for (const s of cube(
          [x, y, z],
          [x + dx, y + dx, z + dx],
          mesh.levels[level].color,
        ))
          unique.set(JSON.stringify([s.a, s.b]), s);
      }
      segments = [...unique.values()];
      const n = mesh.cells.length;
      caption.textContent = `${n} active cells across the innermost refinement boundary. Fine cells have half the spacing of their coarse neighbors.`;
    } else {
      if (mode === "core") extent = mesh.levels.at(-1).half * 1.5;
      for (const [level, ...points] of mesh.planes3d) {
        const a = points.slice(0, 3), b = points.slice(3);
        if (a.some((v, axis) => v === b[axis] && Math.abs(v) > extent)) continue;
        const clip = (v) => Math.max(-extent, Math.min(extent, v));
        const lo = a.map(clip), hi = b.map(clip);
        if (lo.every((v, axis) => v === hi[axis])) continue;
        segments.push({ a: lo, b: hi, color: mesh.levels[level].color, weight: 0.65 });
      }
      for (const level of mesh.levels)
        if (level.half <= extent)
          segments.push(
            ...cube(
              Array(3).fill(-level.half),
              Array(3).fill(level.half),
              level.color,
              1.35,
            ),
          );
      caption.textContent =
        mode === "core"
          ? "Three central sections at the core. Every line follows an actual cell face; covered coarse cells are excluded."
          : "Three central sections through the active cells. Outer bands are included; cube outlines mark the central refinement regions.";
    }
    redraw();
  }
  function magnify(factor) {
    zoom = Math.min(8, Math.max(0.45, zoom * factor));
    redraw();
  }
  canvas.addEventListener("pointerdown", (e) => {
    drag = [e.pointerId, e.clientX, e.clientY];
    canvas.setPointerCapture(e.pointerId);
    canvas.focus({ preventScroll: true });
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag[0]) return;
    yaw += (e.clientX - drag[1]) * 0.007;
    pitch = Math.max(
      -1.4,
      Math.min(1.4, pitch + (e.clientY - drag[2]) * 0.007),
    );
    drag = [e.pointerId, e.clientX, e.clientY];
    redraw();
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"])
    canvas.addEventListener(event, () => {
      drag = null;
    });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      magnify(Math.exp(-e.deltaY * 0.001));
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (e) => {
    if (
      ![
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "+",
        "=",
        "-",
        "0",
      ].includes(e.key)
    )
      return;
    e.preventDefault();
    if (e.key === "ArrowLeft") yaw -= 0.1;
    if (e.key === "ArrowRight") yaw += 0.1;
    if (e.key === "ArrowUp") pitch -= 0.1;
    if (e.key === "ArrowDown") pitch += 0.1;
    pitch = Math.max(-1.4, Math.min(1.4, pitch));
    if (e.key === "+" || e.key === "=") magnify(1.2);
    if (e.key === "-") magnify(1 / 1.2);
    if (e.key === "0") reset();
    redraw();
  });
  function reset() {
    yaw = -0.65;
    pitch = 0.48;
    zoom = 1;
    redraw();
  }
  for (const view of ["whole", "core", "cells"])
    document.getElementById(view).addEventListener("click", () => select(view));
  document.getElementById("reset").addEventListener("click", reset);
  document.getElementById("reset-whole").addEventListener("click", reset);
  document
    .getElementById("zoom-in")
    .addEventListener("click", () => magnify(1.25));
  document
    .getElementById("zoom-out")
    .addEventListener("click", () => magnify(0.8));
  new ResizeObserver(resize).observe(canvas);
  fetch("data/native-mesh.json", { cache: "no-store" })
    .then((r) => {
      if (!r.ok) throw Error("Mesh data unavailable");
      return r.json();
    })
    .then((data) => {
      mesh = data;
      for (const level of mesh.levels) {
        const item = document.createElement("span");
        item.style.setProperty("--level", level.color);
        item.textContent = `L${level.level} · Δ = ${level.dx}`;
        document.getElementById("mesh-legend").append(item);
      }
      status.textContent = "";
      select("whole");
    })
    .catch(() => {
      status.textContent =
        "The mesh could not load. Reload this view to retry.";
    });
})();
