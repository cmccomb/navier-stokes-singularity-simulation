/* Load the public Space only when requested; static movies remain independent. */
(() => {
  const panel = document.getElementById("native-explorer");
  const frame = document.getElementById("native-frame");
  const coverage = document.getElementById("native-coverage");
  if (!panel || !frame || !coverage) return;
  panel.addEventListener("toggle", () => {
    if (panel.open && !frame.getAttribute("src")) frame.src = frame.dataset.src;
  });
  if (window.location.hash === "#native-explorer") panel.open = true;
  fetch("data/native-explorer.json", { cache: "no-cache" })
    .then(response => {
      if (!response.ok) throw new Error("No verified release record");
      return response.json();
    })
    .then(data => {
      if (data.validated && data.complete_history && data.saved_frames === 280 && data.start_time === 0 && data.end_time === 0.995) {
        coverage.textContent = "All 280 native states from exact rest through t = 0.995, with float64 velocity and forcing on the original fixed nested grids. Display sampling does not refine the simulation.";
      }
    })
    .catch(() => { /* Keep the conservative prototype label and working links. */ });
})();
