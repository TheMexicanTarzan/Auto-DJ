/**
 * app.js — Sigma.js frontend for the Auto-DJ Mix Pathfinder.
 *
 * Handles:
 *   - Progress polling during graph load
 *   - Graph loading into graphology + Sigma.js WebGL renderer
 *   - Song search / autocomplete for pathfinding
 *   - Path highlighting (nodes + edges)
 *   - Node click inspection (top-5 neighbours)
 */

/* global graphology, Sigma */

// =========================================================================
// State
// =========================================================================

let sigmaInstance = null;
let graphInstance = null;
let songList = [];          // [{id, label}]

// Selections
let startId = null;
let endId = null;

// Rendering state — sets checked by reducers
const highlightedNodes = new Set();
const highlightedEdges = new Set();
let hoveredNode = null;

// =========================================================================
// 1. Progress polling
// =========================================================================

const loadingOverlay = document.getElementById("loading-overlay");
const mainContent = document.getElementById("main-content");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const headerStats = document.getElementById("header-stats");

let pollTimer = null;

function startPolling() {
  pollTimer = setInterval(pollStatus, 1000);
}

async function pollStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();

    progressBar.style.width = data.progress + "%";

    if (data.error) {
      progressText.textContent = "Error: " + data.error;
      clearInterval(pollTimer);
      return;
    }

    if (data.ready) {
      clearInterval(pollTimer);
      progressText.textContent = "Loading graph...";
      try {
        await loadGraph();
      } catch (loadErr) {
        console.error("Failed to load graph:", loadErr);
        progressText.textContent = "Error loading graph: " + loadErr.message;
      }
      return;
    }

    // Still loading
    if (data.current_file === "cache") {
      progressText.textContent = "Loading graph from cache...";
    } else if (data.total > 0) {
      progressText.textContent =
        "Analysing track " + data.current + " of " + data.total +
        " (" + data.progress + "%) \u2014 " + data.current_file;
    } else {
      progressText.textContent = "Scanning for audio files...";
    }
  } catch (err) {
    // Network blip — keep polling
  }
}

// =========================================================================
// 2. Graph loading
// =========================================================================

async function loadGraph() {
  progressText.textContent = "Loading graph data...";

  const [graphResp, songsResp] = await Promise.all([
    fetch("/api/graph"),
    fetch("/api/songs"),
  ]);
  const graphData = await graphResp.json();
  songList = await songsResp.json();

  // Build graphology instance
  const graph = new graphology.Graph({ multi: false, type: "undirected" });

  graphData.nodes.forEach(function (n) {
    graph.addNode(n.id, {
      x: n.x,
      y: n.y,
      label: n.label,
      size: 4,
      color: "#6c7a89",
      bpm: n.bpm,
      key: n.key,
    });
  });

  graphData.edges.forEach(function (e) {
    // Avoid duplicate edges
    if (!graph.hasEdge(e.source, e.target)) {
      graph.addEdge(e.source, e.target, {
        weight: e.weight,
        color: "#cbd5e0",
      });
    }
  });

  graphInstance = graph;

  // Switch views BEFORE creating Sigma so the container has dimensions
  loadingOverlay.style.display = "none";
  mainContent.classList.add("visible");

  // Instantiate Sigma renderer
  const container = document.getElementById("sigma-container");
  sigmaInstance = new Sigma(graph, container, {
    renderEdgeLabels: false,
    enableEdgeEvents: false,
    defaultNodeColor: "#6c7a89",
    defaultEdgeColor: "#cbd5e0",
    labelColor: { color: "#e2e8f0" },
    labelFont: "Segoe UI, Roboto, sans-serif",
    labelSize: 12,
    labelRenderedSizeThreshold: 14,
    // --- Reducers for selective rendering ---
    nodeReducer: nodeReducer,
    edgeReducer: edgeReducer,
  });

  // Wire up interactions
  sigmaInstance.on("enterNode", onEnterNode);
  sigmaInstance.on("leaveNode", onLeaveNode);
  sigmaInstance.on("clickNode", onClickNode);

  // Update header stats
  headerStats.textContent =
    graphData.num_nodes + " tracks loaded | " +
    graphData.num_edges + " possible transitions";

  // Populate search inputs
  populateSearch();
}

// =========================================================================
// 3. Reducers (control what is visible)
// =========================================================================

function nodeReducer(node, data) {
  var res = Object.assign({}, data);

  if (highlightedNodes.size > 0) {
    if (highlightedNodes.has(node)) {
      res.color = "#e53e3e";
      res.size = 8;
      res.zIndex = 2;
      res.label = data.label;
    } else {
      res.color = "#4a5568";
      res.size = 3;
      res.label = null;
      res.zIndex = 0;
    }
  }

  if (hoveredNode !== null) {
    if (node === hoveredNode) {
      res.color = "#fc8181";
      res.size = 10;
      res.label = data.label;
      res.zIndex = 3;
    } else if (graphInstance.hasEdge(node, hoveredNode) || graphInstance.hasEdge(hoveredNode, node)) {
      // Neighbour of hovered node — keep visible
      res.color = res.color === "#e53e3e" ? "#e53e3e" : "#a0aec0";
      res.size = res.size > 4 ? res.size : 5;
      res.label = data.label;
    }
  }

  return res;
}

function edgeReducer(edge, data) {
  var res = Object.assign({}, data);

  // Default: hide all edges for performance
  res.hidden = true;

  // Show highlighted (path) edges
  if (highlightedEdges.has(edge)) {
    res.hidden = false;
    res.color = "#e53e3e";
    res.size = 3;
    res.zIndex = 2;
  }

  // Show edges of hovered node
  if (hoveredNode !== null) {
    var extremities = graphInstance.extremities(edge);
    if (extremities[0] === hoveredNode || extremities[1] === hoveredNode) {
      res.hidden = false;
      if (!highlightedEdges.has(edge)) {
        res.color = "#718096";
        res.size = 1;
      }
    }
  }

  return res;
}

// =========================================================================
// 4. Hover / Click interactions
// =========================================================================

function onEnterNode(event) {
  hoveredNode = event.node;
  sigmaInstance.refresh();
}

function onLeaveNode() {
  hoveredNode = null;
  sigmaInstance.refresh();
}

async function onClickNode(event) {
  loadNodeDetails(event.node);
}

async function loadNodeDetails(nodeId) {
  var detailsEl = document.getElementById("node-details");
  detailsEl.textContent = "Loading...";

  var k = parseInt(document.getElementById("top-k-input").value, 10) || 10;

  try {
    var resp = await fetch("/api/neighbors/" + encodeURIComponent(nodeId) + "?k=" + k);
    var data = await resp.json();

    if (data.error) {
      detailsEl.textContent = data.error;
      return;
    }

    var lines = [
      data.node.label,
      "BPM: " + data.node.bpm + "  |  Key: " + data.node.key,
      "Path: " + data.node.id,
      "",
      "Top " + data.neighbors.length + " Mixable Tracks:",
      "------------------------------------",
    ];

    if (data.neighbors.length === 0) {
      lines.push("  (no compatible neighbours)");
    } else {
      data.neighbors.forEach(function (nbr, i) {
        lines.push(
          "  " + (i + 1) + ". " + nbr.label +
          "\n     BPM: " + nbr.bpm + " | Key: " + nbr.key +
          "\n     Transition cost: " + nbr.cost.toFixed(4)
        );
      });
    }

    detailsEl.textContent = lines.join("\n");
  } catch (err) {
    detailsEl.textContent = "Failed to load details.";
  }
}

// =========================================================================
// 5. Song search / autocomplete
// =========================================================================

function populateSearch() {
  setupAutocomplete("start-search", "start-results", function (id) {
    startId = id;
  });
  setupAutocomplete("end-search", "end-results", function (id) {
    endId = id;
  });
  setupAutocomplete("details-search", "details-results", function (id) {
    loadNodeDetails(id);
  });
}

function setupAutocomplete(inputId, resultsId, onSelect) {
  var input = document.getElementById(inputId);
  var resultsDiv = document.getElementById(resultsId);
  var activeIndex = -1;

  input.addEventListener("input", function () {
    var query = input.value.toLowerCase().trim();
    resultsDiv.innerHTML = "";
    activeIndex = -1;

    if (query.length < 1) {
      resultsDiv.classList.remove("open");
      return;
    }

    var matches = songList.filter(function (s) {
      return s.label.toLowerCase().indexOf(query) !== -1;
    }).slice(0, 50);  // Limit results

    if (matches.length === 0) {
      resultsDiv.classList.remove("open");
      return;
    }

    matches.forEach(function (m) {
      var div = document.createElement("div");
      div.className = "result-item";
      div.textContent = m.label;
      div.addEventListener("mousedown", function (e) {
        e.preventDefault();
        input.value = m.label;
        onSelect(m.id);
        resultsDiv.classList.remove("open");
      });
      resultsDiv.appendChild(div);
    });

    resultsDiv.classList.add("open");
  });

  input.addEventListener("keydown", function (e) {
    var items = resultsDiv.querySelectorAll(".result-item");
    if (!items.length) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      updateActive(items, activeIndex);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      updateActive(items, activeIndex);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIndex >= 0 && items[activeIndex]) {
        items[activeIndex].dispatchEvent(new Event("mousedown"));
      }
    } else if (e.key === "Escape") {
      resultsDiv.classList.remove("open");
    }
  });

  input.addEventListener("blur", function () {
    // Delay to allow click on result item
    setTimeout(function () {
      resultsDiv.classList.remove("open");
    }, 200);
  });

  input.addEventListener("focus", function () {
    if (input.value.length >= 1 && resultsDiv.children.length > 0) {
      resultsDiv.classList.add("open");
    }
  });
}

function updateActive(items, idx) {
  items.forEach(function (item, i) {
    if (i === idx) {
      item.classList.add("active");
      item.scrollIntoView({ block: "nearest" });
    } else {
      item.classList.remove("active");
    }
  });
}

// =========================================================================
// 6. Pathfinding
// =========================================================================

document.getElementById("find-path-btn").addEventListener("click", async function () {
  var outputEl = document.getElementById("path-output");

  if (!startId || !endId) {
    outputEl.textContent = "Please select both a start and destination song.";
    return;
  }

  var btn = document.getElementById("find-path-btn");
  btn.disabled = true;
  outputEl.textContent = "Finding path...";

  try {
    var resp = await fetch("/api/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: startId, end: endId }),
    });
    var result = await resp.json();

    if (result.error) {
      outputEl.textContent = result.error;
      clearHighlights();
      return;
    }

    // Update highlights
    highlightedNodes.clear();
    highlightedEdges.clear();

    result.path_nodes.forEach(function (nid) {
      highlightedNodes.add(nid);
    });

    // Find edge keys in graphology for each path edge pair
    result.path_edges.forEach(function (pair) {
      var src = pair[0];
      var dst = pair[1];
      // graphology edge key lookup
      var edgeKey = graphInstance.edge(src, dst);
      if (edgeKey != null) {
        highlightedEdges.add(edgeKey);
      }
      // Try reverse for undirected
      edgeKey = graphInstance.edge(dst, src);
      if (edgeKey != null) {
        highlightedEdges.add(edgeKey);
      }
    });

    sigmaInstance.refresh();
    outputEl.textContent = result.summary;
  } catch (err) {
    outputEl.textContent = "Request failed: " + err.message;
  } finally {
    btn.disabled = false;
  }
});

function clearHighlights() {
  highlightedNodes.clear();
  highlightedEdges.clear();
  if (sigmaInstance) sigmaInstance.refresh();
}

// =========================================================================
// 7. Recompute layout
// =========================================================================

document.getElementById("recompute-layout-btn").addEventListener("click", async function () {
  var btn = document.getElementById("recompute-layout-btn");
  var statusEl = document.getElementById("layout-status");
  btn.disabled = true;
  statusEl.textContent = "Recomputing layout...";

  try {
    var body = {};
    var niter = document.getElementById("layout-niter").value;
    var startTemp = document.getElementById("layout-start-temp").value;
    if (niter) body.niter = Number(niter);
    if (startTemp) body.start_temp = Number(startTemp);

    var resp = await fetch("/api/recompute-layout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    var result = await resp.json();

    if (result.error) {
      statusEl.textContent = "Error: " + result.error;
      return;
    }

    // Fetch updated coordinates
    var graphResp = await fetch("/api/graph");
    var graphData = await graphResp.json();

    // Update node positions in graphology
    graphData.nodes.forEach(function (n) {
      if (graphInstance.hasNode(n.id)) {
        graphInstance.setNodeAttribute(n.id, "x", n.x);
        graphInstance.setNodeAttribute(n.id, "y", n.y);
      }
    });

    sigmaInstance.refresh();
    statusEl.textContent = "Layout updated.";
  } catch (err) {
    statusEl.textContent = "Failed: " + err.message;
  } finally {
    btn.disabled = false;
  }
});

// =========================================================================
// 8. Bootstrap
// =========================================================================

startPolling();
