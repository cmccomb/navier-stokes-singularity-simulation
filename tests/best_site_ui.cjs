// Unit-level page-controller checks; no browser or live DOM required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const site = path.join(__dirname, "../site");
const html = fs.readFileSync(path.join(site, "index.html"), "utf8");
const run = JSON.parse(fs.readFileSync(path.join(site, "data/best.json")));
const views = JSON.parse(fs.readFileSync(path.join(site, "data/three-view.json")));
// Use a shape-correct detail fixture before the full rendering is available.
const detailPath = path.join(site, "data/detail-view.json");
const detail = fs.existsSync(detailPath) ? JSON.parse(fs.readFileSync(detailPath)) : {
  ...views, kind: "native-detail-3d",
  media: Object.fromEntries(["flow", "force"].map((kind) => [kind, {
    ...views.media[kind], files: Object.fromEntries(Object.entries(views.media[kind].files).map(([ext, file]) => [ext, { ...file, path: file.path.replace("-views.", "-detail.") }]))
  }]))
};

async function exercise(reduced, invalidRecord = false, invalidViews = false, invalidDetail = false) {
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => [m[1], {
    id: m[1], currentTime: 0, duration: 57.1,
    paused: true, plays: 0, hidden: m[1].endsWith("-switch"), attributes: {},
    classes: new Set(),
    get classList() { return { toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name) }; },
    setAttribute(name, value) { this.attributes[name] = value; },
    pause() { this.paused = true; },
    play() { this.plays += 1; this.paused = false; return Promise.resolve(); },
  }]));
  assert(!html.includes('role="tab'));
  assert(!nodes.has("flow-tab-peak") && !nodes.has("force-tab-peak"));
  const captions = [{}, {}];
  const errors = [];
  const warnings = [];
  const document = {
    querySelector(selector) {
      assert(selector.startsWith("#") && nodes.has(selector.slice(1)), selector);
      return nodes.get(selector.slice(1));
    },
    querySelectorAll(selector) {
      assert.equal(selector, "[data-clip-range]");
      return captions;
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(site, "best.js"), "utf8"), {
    document, URL, window: { location: new URL("http://localhost/"), matchMedia: () => ({ matches: reduced }) },
    fetch: async (url) => ({ ok: true, json: async () => url.includes("detail-view") ? (invalidDetail ? { ...detail, source_record_sha256: "wrong" } : detail) : url.includes("three-view") ? (invalidViews ? { ...views, view_order: ["xz", "xy", "isometric"] } : views) : invalidRecord ? { ...run, validated: false } : run }),
    console: { error: (error) => errors.push(error), warn: (...message) => warnings.push(message) },
  });
  await new Promise(setImmediate);
  if (invalidRecord || invalidViews) {
    assert.equal(errors.length, 1);
    for (const kind of ["flow", "force"]) {
      assert.equal(nodes.get(`${kind}-video`).src, undefined);
      assert.equal(nodes.get(`${kind}-pending`).hidden, false);
      assert(nodes.get(`${kind}-pending`).textContent.includes("metadata unavailable"));
      assert.equal(nodes.get(`${kind}-figure`).hidden, false);
      assert(html.includes(`href="media/kay-${kind}-views.gif"`));
    }
    return;
  }
  assert.equal(errors.length, 0);
  for (const kind of ["flow", "force"]) {
    const video = nodes.get(`${kind}-video`);
    const media = views.media[kind].files;
    const expected = (ext) => new URL(`${media[ext].path}?v=${media[ext].sha256.slice(0, 12)}`, "http://localhost/").href;
    assert.equal(video.src, expected("mp4"));
    assert.equal(video.poster, expected("png"));
    assert.equal(nodes.get(`${kind}-gif`).href, expected("gif"));
    assert.equal(nodes.get(`${kind}-mp4`).href, expected("mp4"));
    assert.equal(video.plays, 0);
    video.onloadedmetadata();
    assert.equal(video.plays, reduced ? 0 : 1);
    assert.equal(video.muted, true);
    assert.equal(nodes.get(`${kind}-figure`).hidden, false);
    video.onloadeddata();
    assert.equal(nodes.get(`${kind}-pending`).hidden, true);
    video.onerror();
    assert.equal(nodes.get(`${kind}-pending`).hidden, false);
    assert(nodes.get(`${kind}-pending`).textContent.includes("GIF download"));
    if (invalidDetail) {
      assert.equal(nodes.get(`${kind}-switch`).hidden, true);
      assert.equal(warnings.length, 1);
      continue;
    }
    assert.equal(nodes.get(`${kind}-switch`).hidden, false);
    video.pause();
    video.currentTime = 27.4;
    const plays = video.plays;
    nodes.get(`${kind}-detail`).onclick();
    assert(video.src.includes(`kay-${kind}-detail.mp4`));
    assert(nodes.get(`${kind}-gif`).href.includes(`kay-${kind}-detail.gif`));
    assert.equal(nodes.get(`${kind}-detail`).attributes["aria-pressed"], "true");
    assert(nodes.get(`${kind}-figure`).classes.has("detail-view"));
    video.onloadedmetadata();
    assert.equal(video.currentTime, 27.4);
    assert.equal(video.plays, plays); // Switching a paused video cannot autoplay.
    assert.equal(video.paused, true);
    video.play();
    video.currentTime = 33.6;
    nodes.get(`${kind}-views`).onclick();
    const staleHandler = video.onloadedmetadata;
    // Rapidly switching again while metadata loads retains the pending time.
    video.currentTime = 0;
    nodes.get(`${kind}-detail`).onclick();
    staleHandler();
    assert.equal(video.currentTime, 0);
    video.onloadedmetadata();
    assert.equal(video.currentTime, 33.6);
    assert.equal(video.paused, false);
    nodes.get(`${kind}-views`).onclick();
    video.onloadedmetadata();
    assert.equal(video.currentTime, 33.6);
    assert(!nodes.get(`${kind}-figure`).classes.has("detail-view"));
    assert.equal(nodes.get(`${kind}-view-note`).textContent, "Fixed planes: y = 0 and z = 0. No vector arrows.");
  }
  assert(captions.every((c) => c.textContent.includes("280 saved frames")));
}
exercise(false).then(() => exercise(true)).then(() => exercise(false, true))
  .then(() => exercise(false, false, true))
  .then(() => exercise(false, false, false, true))
  .then(() => console.log("Media, detail toggles, stable clocks, rapid switching, fallback and reduced motion passed."));
