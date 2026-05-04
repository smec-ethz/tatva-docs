// Show/hide .parallel-only blocks based on whether any "Parallel" tab is active.
// Works with both independent tab groups and linked tabs (content.tabs.link).

function syncParallelOnly() {
    let parallelActive = false;
    document.querySelectorAll(".tabbed-set").forEach(set => {
        const checked = set.querySelector("input[type='radio']:checked");
        if (checked) {
            const label = set.querySelector(`label[for="${checked.id}"]`);
            if (label && label.textContent.trim() === "Parallel") {
                parallelActive = true;
            }
        }
    });
    document.querySelectorAll(".parallel-only").forEach(el => {
        el.style.display = parallelActive ? "" : "none";
    });
}

// React to tab switches
document.addEventListener("change", e => {
    if (e.target.matches(".tabbed-set input[type='radio']")) {
        syncParallelOnly();
    }
});

// Initial state — defer so MkDocs Material can restore saved tab from localStorage first
document.addEventListener("DOMContentLoaded", () => {
    setTimeout(syncParallelOnly, 0);
});
