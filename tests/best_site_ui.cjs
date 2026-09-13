// Unit-level page-controller checks; no browser or live DOM required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const site = path.join(__dirname, "../site");
const html = fs.readFileSync(path.join(site, "index.html"), "utf8");
const run = JSON.parse(fs.readFileSync(path.join(site, "data/best.json")));
const views = JSON.parse(fs.readFileSync(path.join(site, "data/three-view.json")));

async function exercise(reduced, invalidRecord = false, invalidViews = false) {
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => [m[1], {
    id: m[1], currentTime: 0, duration: 57.1,
    paused: true, plays: 0, hidden: false,
    play() { this.plays += 1; this.paused = false; return Promise.resolve(); },
  }]));
  assert(!html.includes('role="tab'));
  assert(!nodes.has("flow-tab-peak") && !nodes.has("force-tab-peak"));
  const captions = [{}, {}];
  const errors = [];
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
    fetch: async (url) => ({ ok: true, json: async () => url.includes("three-view") ? (invalidViews ? { ...views, view_order: ["xz", "xy", "isometric"] } : views) : invalidRecord ? { ...run, validated: false } : run }),
    console: { error: (error) => errors.push(error) },
  });
  await new Promise(setImmediate);
  if (invalidRecord || invalidViews) {
    assert.equal(errors.length, 1);
    for (const kind of ["flow", "force"]) {
      assert.equal(nodes.get(`${kind}-video`).src, undefined);
      assert.equal(nodes.get(`${kind}-pending`).hidden, false);
      assert(nodes.get(`${kind}-pending`).textContent.includes("metadata unavailable"));
      assert.equal(nodes.get(`${kind}-figure`).hidden, false);
      assert(html.includes(`href="media/oliver-${kind}-views.gif"`));
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
    assert.equal(video.plays, reduced ? 0 : 1);
    assert.equal(video.muted, true);
    assert.equal(nodes.get(`${kind}-figure`).hidden, false);
    video.onloadeddata();
    assert.equal(nodes.get(`${kind}-pending`).hidden, true);
    video.onerror();
    assert.equal(nodes.get(`${kind}-pending`).hidden, false);
    assert(nodes.get(`${kind}-pending`).textContent.includes("GIF download"));
  }
  assert(captions.every((c) => c.textContent.includes("280 saved frames")));
}
exercise(false).then(() => exercise(true)).then(() => exercise(false, true))
  .then(() => exercise(false, false, true))
  .then(() => console.log("Fixed-plane media, downloads, loading errors and reduced motion passed."));
