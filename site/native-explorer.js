/* Start the slice app as its section approaches the viewport. */
(() => {
  const frame = document.getElementById("native-frame");
  if (!frame) return;
  window.addEventListener("message", (event) => {
    if (event.source !== frame.contentWindow || !frame.src) return;
    if (event.origin !== new URL(frame.src).origin) return;
    const height = event.data?.height;
    if (
      event.data?.type === "native-explorer-height" &&
      Number.isFinite(height) &&
      height >= 300 &&
      height <= 4000
    ) {
      frame.style.height = `${Math.ceil(height)}px`;
    }
  });
  const observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        frame.src = frame.dataset.src;
        observer.disconnect();
      }
    },
    { rootMargin: "500px" },
  );
  observer.observe(frame);
})();
