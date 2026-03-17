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

// Directory filtering state
var songsBaseDir = "";           // absolute path prefix from server
var directoryTree = null;        // nested tree from /api/graph
var activeDirectories = null;    // null = all visible; Set of active dir paths when filtering

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

/** Compute the relative directory of a node id given the base path. */
function nodeDirectory(nodeId, base) {
  if (!base) return ".";
  // Strip base prefix + trailing separator
  var rel = nodeId;
  if (nodeId.indexOf(base) === 0) {
    rel = nodeId.slice(base.length);
    if (rel.charAt(0) === "/") rel = rel.slice(1);
  }
  var lastSlash = rel.lastIndexOf("/");
  return lastSlash === -1 ? "." : rel.slice(0, lastSlash);
}

function buildGraphology(graphData) {
  var graph = new graphology.Graph({ multi: false, type: "undirected" });
  var base = graphData.songs_directory || "";

  graphData.nodes.forEach(function (n) {
    graph.addNode(n.id, {
      x: n.x,
      y: n.y,
      label: n.label,
      size: 2,
      color: "#6c7a89",
      bpm: n.bpm,
      key: n.key,
      directory: nodeDirectory(n.id, base),
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

  songsBaseDir = graphData.songs_directory || "";
  directoryTree = graphData.directory_tree || null;

  graphInstance = buildGraphology(graphData);
  buildAdjacencyCache();

  // Switch views BEFORE creating Sigma so the container has dimensions
  loadingOverlay.style.display = "none";
  mainContent.classList.add("visible");

  // Custom label renderer: draws a dark pill behind label text for contrast
  function drawLabelWithBackground(context, data, settings) {
    if (!data.label) return;

    var size = settings.labelSize;
    var font = settings.labelFont;
    var text = data.label;

    context.font = (data.forceLabel ? "bold " : "") + size + "px " + font;

    var textWidth = context.measureText(text).width;
    var padX = 4;
    var padY = 2;
    var x = data.x + data.size + 3;
    var y = data.y + size / 3;

    // Draw background pill
    var bgX = x - padX;
    var bgY = y - size + padY;
    var bgW = textWidth + padX * 2;
    var bgH = size + padY * 2;
    var radius = 3;

    context.fillStyle = "rgba(26, 32, 44, 0.85)";
    context.beginPath();
    context.moveTo(bgX + radius, bgY);
    context.lineTo(bgX + bgW - radius, bgY);
    context.quadraticCurveTo(bgX + bgW, bgY, bgX + bgW, bgY + radius);
    context.lineTo(bgX + bgW, bgY + bgH - radius);
    context.quadraticCurveTo(bgX + bgW, bgY + bgH, bgX + bgW - radius, bgY + bgH);
    context.lineTo(bgX + radius, bgY + bgH);
    context.quadraticCurveTo(bgX, bgY + bgH, bgX, bgY + bgH - radius);
    context.lineTo(bgX, bgY + radius);
    context.quadraticCurveTo(bgX, bgY, bgX + radius, bgY);
    context.closePath();
    context.fill();

    // Draw text
    context.fillStyle = data.labelColor || settings.labelColor.color || "#e2e8f0";
    context.fillText(text, x, y);
  }

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
    defaultDrawNodeLabel: drawLabelWithBackground,
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

  // Render directory filter tree
  if (directoryTree) {
    renderDirectoryTree(directoryTree);
  }
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

  songsBaseDir = graphData.songs_directory || songsBaseDir;
  directoryTree = graphData.directory_tree || directoryTree;

  // Rebuild graphology keeping existing positions
  var base = songsBaseDir;
  graphInstance.clear();
  graphData.nodes.forEach(function (n) {
    graphInstance.addNode(n.id, {
      x: n.x, y: n.y,
      label: n.label,
      size: 2,
      color: "#6c7a89",
      bpm: n.bpm,
      key: n.key,
      directory: nodeDirectory(n.id, base),
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

// Frozen singleton for hidden (directory-filtered) nodes
var HIDDEN_NODE = Object.freeze({
  hidden: true,
  color: "#4a5568",
  size: 0,
  label: null,
  zIndex: 0,
});

function nodeReducer(node, data) {
  // Directory filter: hide nodes not in active directories
  if (activeDirectories !== null) {
    var nodeDir = data.directory || ".";
    if (!isDirectoryActive(nodeDir)) {
      return HIDDEN_NODE;
    }
  }

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
    res.labelColor = "#ffffff";
    res.zIndex = 3;
    res.forceLabel = true;
  } else if (isNeighbor) {
    res.color = res.color === "#e53e3e" ? "#e53e3e" : "#a0aec0";
    res.size = res.size > 2 ? res.size : 2.5;
  }

  return res;
}

function edgeReducer(edge, data) {
  // Directory filter: hide edges touching excluded nodes
  if (activeDirectories !== null) {
    var extremities = graphInstance.extremities(edge);
    var srcDir = graphInstance.getNodeAttribute(extremities[0], "directory") || ".";
    var tgtDir = graphInstance.getNodeAttribute(extremities[1], "directory") || ".";
    if (!isDirectoryActive(srcDir) || !isDirectoryActive(tgtDir)) {
      if (data.hidden) return data;
      var h = Object.assign({}, data);
      h.hidden = true;
      return h;
    }
  }

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
// 5. Directory filter tree
// =========================================================================

/** Collect all leaf directory paths from the tree. */
function collectAllDirs(node, result) {
  if (node.path) result.push(node.path);
  if (node.children) {
    node.children.forEach(function (c) { collectAllDirs(c, result); });
  }
  return result;
}

/** Render the directory filter tree into #directory-tree. */
function renderDirectoryTree(tree) {
  var container = document.getElementById("directory-tree");
  container.innerHTML = "";

  // If tree has no children (flat library), hide the filter entirely
  if (!tree.children || tree.children.length === 0) {
    document.getElementById("directory-filter").style.display = "none";
    return;
  }

  document.getElementById("directory-filter").style.display = "";

  // Collect all directory paths for the "all active" baseline
  var allDirs = collectAllDirs(tree, []);

  function buildNode(node, depth) {
    var li = document.createElement("li");
    li.className = "dir-node";

    var row = document.createElement("div");
    row.className = "dir-row";

    // Indent based on depth
    row.style.paddingLeft = (depth * 16) + "px";

    // Collapse toggle (only if has children)
    var toggle = document.createElement("span");
    toggle.className = "dir-toggle";
    if (node.children && node.children.length > 0) {
      toggle.textContent = "\u25BE"; // down arrow
      toggle.addEventListener("click", function () {
        var childUl = li.querySelector("ul");
        if (childUl) {
          var collapsed = childUl.style.display === "none";
          childUl.style.display = collapsed ? "" : "none";
          toggle.textContent = collapsed ? "\u25BE" : "\u25B8"; // down / right arrow
        }
      });
    } else {
      toggle.textContent = " ";
      toggle.style.visibility = "hidden";
    }
    row.appendChild(toggle);

    // Checkbox
    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.className = "dir-checkbox";
    cb.dataset.path = node.path || ".";
    row.appendChild(cb);

    // Label
    var label = document.createElement("span");
    label.className = "dir-label";
    label.textContent = node.name;
    row.appendChild(label);

    // Count badge
    var badge = document.createElement("span");
    badge.className = "dir-count";
    badge.textContent = node.count;
    row.appendChild(badge);

    li.appendChild(row);

    // Recursively render children
    if (node.children && node.children.length > 0) {
      var childUl = document.createElement("ul");
      childUl.className = "dir-children";
      node.children.forEach(function (child) {
        childUl.appendChild(buildNode(child, depth + 1));
      });
      li.appendChild(childUl);
    }

    // Checkbox change: cascade to children, update parent, apply filter
    cb.addEventListener("change", function () {
      var checked = cb.checked;
      // Cascade down
      var childCbs = li.querySelectorAll(".dir-checkbox");
      childCbs.forEach(function (ccb) { ccb.checked = checked; });
      // Update parent checkboxes (bubble up)
      updateParentCheckboxes(container);
      // Apply filter
      applyDirectoryFilter(container, allDirs);
    });

    return li;
  }

  var ul = document.createElement("ul");
  ul.className = "dir-tree-root";
  tree.children.forEach(function (child) {
    ul.appendChild(buildNode(child, 0));
  });
  container.appendChild(ul);
}

/** Walk up from each checkbox and set parent checkbox state. */
function updateParentCheckboxes(container) {
  // Process from deepest children upward
  var lists = container.querySelectorAll("ul.dir-children");
  for (var i = lists.length - 1; i >= 0; i--) {
    var parentLi = lists[i].parentElement;
    var parentCb = parentLi.querySelector(":scope > .dir-row > .dir-checkbox");
    if (!parentCb) continue;
    var childCbs = lists[i].querySelectorAll(":scope > li > .dir-row > .dir-checkbox");
    var allChecked = true;
    var anyChecked = false;
    childCbs.forEach(function (cb) {
      if (cb.checked) anyChecked = true;
      else allChecked = false;
    });
    parentCb.checked = allChecked;
    parentCb.indeterminate = !allChecked && anyChecked;
  }
}

/** Read checkbox states and update activeDirectories + refresh graph. */
function applyDirectoryFilter(container, allDirs) {
  var checkedPaths = new Set();
  container.querySelectorAll(".dir-checkbox").forEach(function (cb) {
    if (cb.checked) checkedPaths.add(cb.dataset.path);
  });

  // If all directories are checked, set null (no filtering)
  if (checkedPaths.size >= allDirs.length) {
    activeDirectories = null;
  } else {
    activeDirectories = checkedPaths;
  }

  // Rebuild filtered song list for autocomplete
  rebuildFilteredSongList();

  // Refresh sigma to apply reducer filtering
  if (sigmaInstance) {
    scheduleRefresh();
  }

  // Update header stats with filtered counts
  updateFilteredStats();
}

// Filtered song list for autocomplete (rebuilt on directory change)
var filteredSongList = null; // null = use songList (no filter)

function rebuildFilteredSongList() {
  if (activeDirectories === null) {
    filteredSongList = null;
    return;
  }
  filteredSongList = songList.filter(function (s) {
    var dir = nodeDirectory(s.id, songsBaseDir);
    // Check if this dir or any parent is in activeDirectories
    return isDirectoryActive(dir);
  });
}

/** Check if a directory path is active (matches or is a child of an active dir). */
function isDirectoryActive(dir) {
  if (activeDirectories === null) return true;
  if (activeDirectories.has(dir)) return true;
  // Check parent paths
  var idx = dir.lastIndexOf("/");
  while (idx !== -1) {
    var parent = dir.slice(0, idx);
    if (activeDirectories.has(parent)) return true;
    idx = parent.lastIndexOf("/");
  }
  return activeDirectories.has(".");
}

/** Update header stats to reflect filtered node/edge counts. */
function updateFilteredStats() {
  if (!graphInstance || !headerStats) return;

  if (activeDirectories === null) {
    // No filter — show original stats
    var totalNodes = graphInstance.order;
    var totalEdges = graphInstance.size;
    headerStats.textContent = totalNodes + " tracks | " + totalEdges + " transitions";
    return;
  }

  // Count visible nodes
  var visibleNodes = 0;
  graphInstance.forEachNode(function (node, attrs) {
    var dir = attrs.directory || ".";
    if (isDirectoryActive(dir)) visibleNodes++;
  });

  // Count visible edges (both endpoints visible)
  var visibleEdges = 0;
  graphInstance.forEachEdge(function (edge, attrs, src, tgt, srcAttrs, tgtAttrs) {
    var srcDir = srcAttrs.directory || ".";
    var tgtDir = tgtAttrs.directory || ".";
    if (isDirectoryActive(srcDir) && isDirectoryActive(tgtDir)) visibleEdges++;
  });

  var total = graphInstance.order;
  headerStats.textContent =
    visibleNodes + " of " + total + " tracks | " + visibleEdges + " transitions (filtered)";
}

// =========================================================================
// 6. Song search / autocomplete
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

      var searchList = filteredSongList || songList;
      var matches = searchList.filter(function (s) {
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
    // Build excluded directories list (send the smaller set)
    var excludedDirs = null;
    if (activeDirectories !== null && directoryTree) {
      var allDirs = collectAllDirs(directoryTree, []);
      excludedDirs = allDirs.filter(function (d) {
        return !isDirectoryActive(d);
      });
      if (excludedDirs.length === 0) excludedDirs = null;
    }
    var resp = await fetch("/api/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: startId,
        end: endId,
        allowed_types: allowedTypes.length < 3 ? allowedTypes : null,
        excluded_dirs: excludedDirs,
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
