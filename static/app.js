/**
 * app.js — Sigma.js frontend for the Auto-DJ Mix Pathfinder.
 *
 * Handles:
 *   - Progress polling during graph load
 *   - Graph loading into graphology + Sigma.js WebGL renderer
 *   - Song search / autocomplete for pathfinding
 *   - Path highlighting (nodes + edges) with edge-type coloring
 *   - Node click inspection (top-K neighbours)
 *   - Tempo-relationship checkboxes for pathfinding / neighbor filtering
 *   - Weight tuning with server-side recalculation
 */

/* global graphology, Sigma */

// =========================================================================
// Edge type color palette
// =========================================================================

var EDGE_COLORS = {
  direct:  { path: "#e53e3e", hover: "#718096" },
  double:  { path: "#48bb78", hover: "#276749" },
  triplet: { path: "#4299e1", hover: "#2b6cb0" },
};

function edgePathColor(edgeType) {
  return (EDGE_COLORS[edgeType] || EDGE_COLORS.direct).path;
}
function edgeHoverColor(edgeType) {
  return (EDGE_COLORS[edgeType] || EDGE_COLORS.direct).hover;
}

// =========================================================================
// State
// =========================================================================

var sigmaInstance = null;
var graphInstance = null;
var songList = [];          // [{id, label}]

// Selections
var startId = null;
var endId = null;

// Rendering state — sets checked by reducers
var highlightedNodes = new Set();
var highlightedEdges = new Set();
var hoveredNode = null;
var hoveredNeighbors = new Set();
var hoveredEdgeKeys = new Set();
var detailNode = null;        // node selected via details/click

// Pre-computed adjacency cache: nodeId -> { neighbors: Set, edges: Set }
var adjacencyCache = new Map();

// Shared singleton returned for dimmed (non-highlighted, non-hovered) nodes
var DIMMED_NODE = Object.freeze({
  color: "#4a5568",
  size: 1.5,
  label: null,
  zIndex: 0,
});

// requestAnimationFrame gate for hover refreshes
var pendingRefresh = null;

// =========================================================================
// Helpers
// =========================================================================

/** Build adjacency cache from the current graphInstance. */
function buildAdjacencyCache() {
  adjacencyCache.clear();
  graphInstance.forEachNode(function (node) {
    adjacencyCache.set(node, {
      neighbors: new Set(graphInstance.neighbors(node)),
      edges: new Set(graphInstance.edges(node)),
    });
  });
}

/** Enable/disable zIndex sorting only when visual state requires it. */
function syncZIndex() {
  var needed = hoveredNode !== null || highlightedNodes.size > 0 || detailNode !== null;
  if (sigmaInstance.getSetting("zIndex") !== needed) {
    sigmaInstance.setSetting("zIndex", needed);
  }
}

/** Schedule a hover refresh, coalescing multiple calls per animation frame. */
function scheduleRefresh() {
  if (pendingRefresh) return;
  pendingRefresh = requestAnimationFrame(function () {
    pendingRefresh = null;
    syncZIndex();
    sigmaInstance.refresh({ skipIndexation: true });
  });
}

/** Read the tempo-type checkboxes and return an array of allowed types. */
function getAllowedTypes() {
  var types = [];
  if (document.getElementById("filter-direct").checked)  types.push("direct");
  if (document.getElementById("filter-double").checked)  types.push("double");
  if (document.getElementById("filter-triplet").checked) types.push("triplet");
  return types;
}

// =========================================================================
// 1. Progress polling
// =========================================================================

var loadingOverlay = document.getElementById("loading-overlay");
var mainContent = document.getElementById("main-content");
var progressBar = document.getElementById("progress-bar");
var progressText = document.getElementById("progress-text");
var headerStats = document.getElementById("header-stats");

var pollTimer = null;

function startPolling() {
  pollTimer = setInterval(pollStatus, 1000);
}

async function pollStatus() {
  try {
    var resp = await fetch("/api/status");
    var data = await resp.json();

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

function updateHeaderStats(graphData) {
  var counts = graphData.edge_type_counts || {};
  var parts = [graphData.num_nodes + " tracks"];
  var edgeParts = [];
  if (counts.direct)  edgeParts.push(counts.direct + " direct");
  if (counts.double)  edgeParts.push(counts.double + " double");
  if (counts.triplet) edgeParts.push(counts.triplet + " triplet");
  if (edgeParts.length > 0) {
    parts.push(edgeParts.join(" + ") + " transitions (" + graphData.num_edges + " total)");
  } else {
    parts.push(graphData.num_edges + " transitions");
  }
  headerStats.textContent = parts.join(" | ");
}

function buildGraphology(graphData) {
  var graph = new graphology.Graph({ multi: false, type: "undirected" });

  graphData.nodes.forEach(function (n) {
    graph.addNode(n.id, {
      x: n.x,
      y: n.y,
      label: n.label,
      size: 2,
      color: "#6c7a89",
      bpm: n.bpm,
      key: n.key,
    });
  });

  graphData.edges.forEach(function (e) {
    if (!graph.hasEdge(e.source, e.target)) {
      var etype = e.edge_type || "direct";
      graph.addEdge(e.source, e.target, {
        weight: e.weight,
        edge_type: etype,
        color: edgePathColor(etype),
        hidden: true,
      });
    }
  });

  return graph;
}

async function loadGraph() {
  progressText.textContent = "Loading graph data...";

  var graphResp = await fetch("/api/graph");
  var songsResp = await fetch("/api/songs");
  var graphData = await graphResp.json();
  songList = await songsResp.json();
  songList.forEach(function (s) { s._lower = s.label.toLowerCase(); });

  graphInstance = buildGraphology(graphData);
  buildAdjacencyCache();

  // Switch views BEFORE creating Sigma so the container has dimensions
  loadingOverlay.style.display = "none";
  mainContent.classList.add("visible");

  // Instantiate Sigma renderer
  var container = document.getElementById("sigma-container");
  sigmaInstance = new Sigma(graphInstance, container, {
    renderEdgeLabels: false,
    enableEdgeEvents: false,
    defaultNodeColor: "#6c7a89",
    defaultEdgeColor: "#cbd5e0",
    labelColor: { color: "#e2e8f0" },
    labelFont: "Segoe UI, Roboto, sans-serif",
    labelSize: 12,
    labelRenderedSizeThreshold: 100,
    zIndex: false,
    // --- Reducers for selective rendering ---
    nodeReducer: nodeReducer,
    edgeReducer: edgeReducer,
  });

  // Wire up interactions
  sigmaInstance.on("enterNode", onEnterNode);
  sigmaInstance.on("leaveNode", onLeaveNode);
  sigmaInstance.on("clickNode", onClickNode);

  // Update header stats
  updateHeaderStats(graphData);

  // Populate search inputs
  populateSearch();
}

/** Reload graph data from server (after recalculate) without full page reload. */
async function reloadGraphData() {
  var graphResp = await fetch("/api/graph");
  var graphData = await graphResp.json();

  // Clear old state
  highlightedNodes.clear();
  highlightedEdges.clear();
  hoveredNode = null;
  hoveredNeighbors.clear();
  hoveredEdgeKeys.clear();
  detailNode = null;

  // Rebuild graphology keeping existing positions
  graphInstance.clear();
  graphData.nodes.forEach(function (n) {
    graphInstance.addNode(n.id, {
      x: n.x, y: n.y,
      label: n.label,
      size: 2,
      color: "#6c7a89",
      bpm: n.bpm,
      key: n.key,
    });
  });
  graphData.edges.forEach(function (e) {
    if (!graphInstance.hasEdge(e.source, e.target)) {
      var etype = e.edge_type || "direct";
      graphInstance.addEdge(e.source, e.target, {
        weight: e.weight,
        edge_type: etype,
        color: edgePathColor(etype),
        hidden: true,
      });
    }
  });

  buildAdjacencyCache();
  sigmaInstance.refresh();
  updateHeaderStats(graphData);
}

// =========================================================================
// 3. Reducers (control what is visible)
// =========================================================================

function nodeReducer(node, data) {
  // Fast path: nothing active — return data as-is (zero allocation)
  if (highlightedNodes.size === 0 && hoveredNode === null && detailNode === null) {
    return data;
  }

  var isHovered = node === hoveredNode;
  var isNeighbor = hoveredNeighbors.has(node);
  var isHighlighted = highlightedNodes.has(node);
  var isDetail = detailNode !== null && node === detailNode;

  // Dim path: node is not involved in any active state — return frozen singleton
  if (highlightedNodes.size > 0 && !isHighlighted && !isHovered && !isNeighbor && !isDetail) {
    return DIMMED_NODE;
  }

  // Only remaining: nodes that actually need modification
  if (!isHighlighted && !isHovered && !isNeighbor && !isDetail) return data;

  var res = Object.assign({}, data);

  if (highlightedNodes.size > 0 && isHighlighted) {
    res.color = "#e53e3e";
    res.size = 4;
    res.zIndex = 2;
    res.label = data.label;
    res.forceLabel = true;
  }

  if (isDetail) {
    res.color = "#f6ad55";
    res.size = 4;
    res.zIndex = 2;
    res.label = data.label;
    res.forceLabel = true;
  }

  if (isHovered) {
    res.color = "#fc8181";
    res.size = 5;
    res.label = data.label;
    res.labelColor = "#000000";
    res.zIndex = 3;
    res.forceLabel = true;
  } else if (isNeighbor) {
    res.color = res.color === "#e53e3e" ? "#e53e3e" : "#a0aec0";
    res.size = res.size > 2 ? res.size : 2.5;
  }

  return res;
}

function edgeReducer(edge, data) {
  // Fast path: check the two cheap conditions first
  var isHighlighted = highlightedEdges.has(edge);
  var isHovered = hoveredNode !== null && hoveredEdgeKeys.has(edge);

  // If neither highlighted nor hovered, hide without allocating a copy
  if (!isHighlighted && !isHovered) {
    if (data.hidden) return data;
    var hidden = Object.assign({}, data);
    hidden.hidden = true;
    return hidden;
  }

  var res = Object.assign({}, data);
  var etype = data.edge_type || "direct";

  if (isHighlighted) {
    res.hidden = false;
    res.color = edgePathColor(etype);
    res.size = 1.5;
    res.zIndex = 2;
  } else {
    // isHovered must be true
    res.hidden = false;
    res.color = edgeHoverColor(etype);
    res.size = 0.5;
  }

  return res;
}

// =========================================================================
// 4. Hover / Click interactions
// =========================================================================

function onEnterNode(event) {
  hoveredNode = event.node;
  var cached = adjacencyCache.get(event.node);
  if (cached) {
    hoveredNeighbors = cached.neighbors;
    hoveredEdgeKeys = cached.edges;
  } else {
    hoveredNeighbors = new Set(graphInstance.neighbors(event.node));
    hoveredEdgeKeys = new Set(graphInstance.edges(event.node));
  }
  scheduleRefresh();
}

function onLeaveNode() {
  hoveredNode = null;
  hoveredNeighbors = new Set();
  hoveredEdgeKeys = new Set();
  scheduleRefresh();
}

async function onClickNode(event) {
  loadNodeDetails(event.node);
}

async function loadNodeDetails(nodeId) {
  // Track detail-selected node for z-index
  detailNode = nodeId;
  if (sigmaInstance) scheduleRefresh();

  var detailsEl = document.getElementById("node-details");
  detailsEl.textContent = "Loading...";

  var k = parseInt(document.getElementById("top-k-input").value, 10) || 10;
  var types = getAllowedTypes();
  var typesParam = types.length < 3 ? "&types=" + types.join(",") : "";

  try {
    var resp = await fetch(
      "/api/neighbors/" + encodeURIComponent(nodeId) + "?k=" + k + typesParam
    );
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
        var typeTag = nbr.edge_type && nbr.edge_type !== "direct"
          ? " [" + nbr.edge_type + "]" : "";
        lines.push(
          "  " + (i + 1) + ". " + nbr.label + typeTag +
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
  var debounceTimer = null;

  input.addEventListener("input", function () {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () {
      var query = input.value.toLowerCase().trim();
      activeIndex = -1;

      if (query.length < 1) {
        resultsDiv.innerHTML = "";
        resultsDiv.classList.remove("open");
        return;
      }

      var matches = songList.filter(function (s) {
        return s._lower.indexOf(query) !== -1;
      }).slice(0, 50);

      if (matches.length === 0) {
        resultsDiv.innerHTML = "";
        resultsDiv.classList.remove("open");
        return;
      }

      var frag = document.createDocumentFragment();
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
        frag.appendChild(div);
      });

      resultsDiv.innerHTML = "";
      resultsDiv.appendChild(frag);
      resultsDiv.classList.add("open");
    }, 100);
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
    var allowedTypes = getAllowedTypes();
    var resp = await fetch("/api/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: startId,
        end: endId,
        allowed_types: allowedTypes.length < 3 ? allowedTypes : null,
      }),
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
    // (undirected graph — edge() returns the key regardless of argument order)
    result.path_edges.forEach(function (pair) {
      var edgeKey = graphInstance.edge(pair[0], pair[1]);
      if (edgeKey != null) {
        highlightedEdges.add(edgeKey);
      }
    });

    syncZIndex();
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
  detailNode = null;
  if (sigmaInstance) {
    syncZIndex();
    sigmaInstance.refresh();
  }
}

// =========================================================================
// 7. Recalculate edges
// =========================================================================

document.getElementById("recalculate-btn").addEventListener("click", async function () {
  var btn = document.getElementById("recalculate-btn");
  var statusEl = document.getElementById("recalculate-status");
  btn.disabled = true;
  statusEl.textContent = "Recalculating...";

  try {
    var resp = await fetch("/api/recalculate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        harmonic: parseFloat(document.getElementById("w-harmonic").value) || 0.35,
        tempo:    parseFloat(document.getElementById("w-tempo").value)    || 0.25,
        semantic: parseFloat(document.getElementById("w-semantic").value) || 0.40,
        double_penalty:  parseFloat(document.getElementById("w-double").value)  || 0.0,
        triplet_penalty: parseFloat(document.getElementById("w-triplet").value) || 0.0,
      }),
    });
    var result = await resp.json();

    if (result.error) {
      statusEl.textContent = "Error: " + result.error;
      return;
    }

    statusEl.textContent = result.message + " (" + result.num_edges + " edges)";

    // Reload graph to reflect new edges
    await reloadGraphData();
  } catch (err) {
    statusEl.textContent = "Request failed: " + err.message;
  } finally {
    btn.disabled = false;
  }
});

// =========================================================================
// 8. Bootstrap
// =========================================================================

startPolling();
