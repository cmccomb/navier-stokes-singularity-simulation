const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const frame = {
  src: "",
  dataset: { src: "https://example.hf.space?__theme=dark" },
  style: {},
  contentWindow: {},
};
let intersect,
  message,
  observed,
  disconnects = 0;
vm.runInNewContext(
  fs.readFileSync(path.join(__dirname, "../site/native-explorer.js"), "utf8"),
  {
    document: { getElementById: () => frame },
    window: {
      addEventListener: (type, callback) => {
        assert.equal(type, "message");
        message = callback;
      },
    },
    URL,
    IntersectionObserver: class {
      constructor(callback) {
        intersect = callback;
      }
      observe(node) {
        observed = node;
      }
      disconnect() {
        disconnects++;
      }
    },
  },
);
assert.equal(observed, frame);
intersect([{ isIntersecting: false }]);
assert.equal(frame.src, "");
intersect([{ isIntersecting: true }]);
assert.equal(frame.src, frame.dataset.src);
assert.equal(disconnects, 1);
const event = {
  source: frame.contentWindow,
  origin: "https://example.hf.space",
  data: { type: "native-explorer-height", height: 1616.2 },
};
message(event);
assert.equal(frame.style.height, "1617px");
for (const invalid of [
  { ...event, origin: "https://unrelated.example" },
  { ...event, source: {} },
  { ...event, data: { type: "other", height: 900 } },
  ...[0, -1, NaN, Infinity, 4001, "900"].map((height) => ({
    ...event,
    data: { ...event.data, height },
  })),
]) {
  message(invalid);
  assert.equal(frame.style.height, "1617px");
}
console.log("Native embed lazy loading and trusted resizing passed.");
