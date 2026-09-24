// Completed history, media identity, loading failures, and reduced motion.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const site = path.join(__dirname, "../site");
const html = fs.readFileSync(path.join(site, "index.html"), "utf8");
const run = JSON.parse(fs.readFileSync(path.join(site, "data/best.json")));
const views = JSON.parse(fs.readFileSync(path.join(site, "data/three-view.json")));

async function exercise(reduced, invalidRecord = false, invalidViews = false, wrongMedia = false) {
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => [m[1], {
    id: m[1], plays: 0, hidden: true,
    play() { this.plays += 1; return Promise.resolve(); },
  }]));
  assert(!nodes.has("flow-switch") && !nodes.has("force-switch"));
  const captions = [{}, {}], errors = [], fetched = [];
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
  const candidate = structuredClone(views);
  if (invalidViews) candidate.source_record_sha256 = "wrong";
  if (wrongMedia) candidate.media.flow.files.mp4.path = "media/kay-flow-views.mp4";
  vm.runInNewContext(fs.readFileSync(path.join(site, "best.js"), "utf8"), {
    document, URL,
    window: { location: new URL("http://localhost/"), matchMedia: () => ({ matches: reduced }) },
    fetch: async (url) => {
      fetched.push(url);
      return { ok: true, json: async () => url.includes("three-view") ? candidate : invalidRecord ? { ...run, validated: false } : run };
    },
    console: { error: (error) => errors.push(error) },
  });
  await new Promise(setImmediate);
  assert.deepEqual(fetched, ["data/best.json", "data/three-view.json"]);
  if (invalidRecord || invalidViews || wrongMedia) {
    assert.equal(errors.length, 1);
    for (const kind of ["flow", "force"]) {
      assert.equal(nodes.get(`${kind}-video`).src, undefined);
      assert.equal(nodes.get(`${kind}-pending`).hidden, false);
      assert(nodes.get(`${kind}-pending`).textContent.includes("metadata unavailable"));
      assert.equal(nodes.get(`${kind}-figure`).hidden, false);
      assert(html.includes(`href="media/oliver-${kind}-views.gif?v=outer128"`));
    }
    return;
  }
  assert.equal(errors.length, 0);
  for (const kind of ["flow", "force"]) {
    const video = nodes.get(`${kind}-video`), media = views.media[kind].files;
    const expected = (ext) => new URL(`${media[ext].path}?v=${media[ext].sha256.slice(0, 12)}`, "http://localhost/").href;
    assert.equal(video.src, expected("mp4"));
    assert.equal(video.poster, expected("png"));
    assert.equal(nodes.get(`${kind}-gif`).href, expected("gif"));
    assert.equal(nodes.get(`${kind}-mp4`).href, expected("mp4"));
    assert.equal(video.plays, 0);
    video.onloadedmetadata();
    assert.equal(video.plays, reduced ? 0 : 1);
    assert.equal(video.muted, true);
    video.onloadeddata();
    assert.equal(nodes.get(`${kind}-pending`).hidden, true);
    video.onerror();
    assert.equal(nodes.get(`${kind}-pending`).hidden, false);
    assert(nodes.get(`${kind}-pending`).textContent.includes("GIF download"));
  }
  assert(captions.every((c) => c.textContent.includes("280 saved frames")));
}
exercise(false).then(() => exercise(true)).then(() => exercise(false, true))
  .then(() => exercise(false, false, true)).then(() => exercise(false, false, false, true))
  .then(() => console.log("Current media identity, complete clock, failure fallback and reduced motion passed."));
