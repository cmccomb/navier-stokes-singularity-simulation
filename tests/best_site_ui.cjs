// Unit-level page-controller checks; no browser or live DOM required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const site = path.join(__dirname, "../site");
const html = fs.readFileSync(path.join(site, "index.html"), "utf8");
const run = JSON.parse(fs.readFileSync(path.join(site, "data/best.json")));

async function exercise(reduced) {
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => [m[1], {
    id: m[1], attributes: {}, listeners: {}, currentTime: 0, duration: 57.1,
    paused: true, plays: 0, hidden: false,
    hasAttribute(name) { return name === "src" ? !!this.src : name in this.attributes; },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    play() { this.plays += 1; this.paused = false; return Promise.resolve(); },
    focus() { this.focused = true; },
  }]));
  const tabs = [...html.matchAll(/<button[^>]*id="([^"]+)"[^>]*data-field="([^"]+)"[^>]*data-view="([^"]+)"/g)]
    .map((m) => Object.assign(nodes.get(m[1]), { dataset: { field: m[2], view: m[3] } }));
  const captions = [{}, {}];
  const document = {
    querySelector(selector) {
      if (selector.startsWith("#")) { assert(nodes.has(selector.slice(1)), selector); return nodes.get(selector.slice(1)); }
      const match = selector.match(/^\[data-field="([^"]+)"\]\[data-view="([^"]+)"\]$/);
      assert(match, selector);
      return tabs.find((t) => t.dataset.field === match[1] && t.dataset.view === match[2]);
    },
    querySelectorAll(selector) {
      if (selector === "[role=tab]") return tabs;
      if (selector === "[data-clip-range]") return captions;
      const match = selector.match(/^\[data-field="([^"]+)"\]$/);
      assert(match, selector);
      return tabs.filter((t) => t.dataset.field === match[1]);
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(site, "best.js"), "utf8"), {
    document, URL, window: { location: new URL("http://localhost/"), matchMedia: () => ({ matches: reduced }) },
    fetch: async () => ({ ok: true, json: async () => run }),
    console: { error: (error) => { throw error; } },
  });
  await new Promise(setImmediate);
  const video = nodes.get("flow-video");
  assert(video.src.includes("oliver-flow-midplane.mp4"));
  assert.equal(video.plays, reduced ? 0 : 1);
  video.currentTime = 7.2;
  nodes.get("flow-tab-peak").listeners.click();
  video.onloadedmetadata();
  assert(video.src.includes("oliver-flow-peak.mp4"));
  assert.equal(video.currentTime, 7.2);
  assert.equal(nodes.get("flow-tab-peak").attributes["aria-selected"], "true");
  assert(nodes.get("flow-view-note").textContent.includes("moving slice"));
  assert(nodes.get("flow-gif").href.includes("oliver-flow-peak.gif"));
  nodes.get("flow-tab-peak").listeners.keydown({ key: "ArrowLeft", preventDefault() {} });
  assert(video.src.includes("oliver-flow-midplane.mp4"));
  assert(nodes.get("flow-tab-2d").focused);
  nodes.get("force-tab-peak").listeners.click();
  assert(nodes.get("force-video").src.includes("oliver-force-peak.mp4"));
  assert(video.src.includes("oliver-flow-midplane.mp4"));
  assert(captions.every((c) => c.textContent.includes("280 saved frames")));
}
exercise(false).then(() => exercise(true)).then(() => console.log("Native view selection, keyboard controls, timing and reduced motion passed."));
