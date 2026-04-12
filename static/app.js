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
 *   - Directory tree filtering
 *   - Tabbed sidebar navigation
 */

/* global graphology, Sigma */

// =========================================================================
// Edge type color palette
// =========================================================================

var EDGE_COLORS = {
  direct:  { path: "#48bb78", hover: "#276749" },
  double:  { path: "#e53e3e", hover: "#718096" },
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
var waypoints = [];        // [{key, id}] – ordered intermediate stops
var waypointCounter = 0;

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

/** Escape a string for safe insertion into innerHTML. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Build the HTML string for a vertical path-flow from an array of node IDs.
 * Shared by the Pathfinding and Create Setlist tabs.
 */
function renderPathFlowHtml(pathNodes) {
  if (!pathNodes || pathNodes.length === 0) {
    return '<p class="output-msg">No tracks found.</p>';
  }
  var html = '<div class="path-flow">';
  pathNodes.forEach(function (nid, i) {
    var label = graphInstance.hasNode(nid)
      ? graphInstance.getNodeAttribute(nid, "label") : nid;
    var bpm = graphInstance.hasNode(nid)
      ? graphInstance.getNodeAttribute(nid, "bpm") : "";
    var key = graphInstance.hasNode(nid)
      ? graphInstance.getNodeAttribute(nid, "key") : "";

    html += '<div class="path-flow-card">';
    html += '<div class="path-flow-card-title">' + escapeHtml(label) + '</div>';
    if (bpm || key) {
      html += '<div class="path-flow-card-meta">';
      if (bpm) html += '<span>BPM: ' + escapeHtml(String(bpm)) + '</span>';
      if (key) html += '<span>Key: ' + escapeHtml(String(key)) + '</span>';
      html += '</div>';
    }
    html += '</div>';

    if (i < pathNodes.length - 1) {
      var nextNid = pathNodes[i + 1];
      var edgeType = "direct";
      if (graphInstance.hasNode(nid) && graphInstance.hasNode(nextNid)) {
        var ek = graphInstance.edge(nid, nextNid);
        if (ek != null) edgeType = graphInstance.getEdgeAttribute(ek, "edge_type") || "direct";
      }
      html += '<div class="path-flow-arrow">\u2193 <span class="edge-badge edge-badge--'
        + escapeHtml(edgeType) + '">' + escapeHtml(edgeType) + '</span></div>';
    }
  });
  html += '</div>';
  return html;
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
      color: "#4A4A4A",
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

    context.fillStyle = "rgba(30, 30, 30, 0.85)";
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
    defaultNodeColor: "#4A4A4A",
    defaultEdgeColor: "#333333",
    labelColor: { color: "#4A4A4A" },
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


// =========================================================================
// 3. Reducers (control what is visible)
// =========================================================================


function nodeReducer(node, data) {
  if (activeDirectories !== null) {
    var nodeDir = data.directory || ".";
    if (!isDirectoryActive(nodeDir)) {
      return { x: data.x, y: data.y, hidden: true, color: "#2D2D2D", size: 0, label: null, zIndex: 0 };
    }
  }

  if (highlightedNodes.size === 0 && hoveredNode === null && detailNode === null) {
    return data;
  }

  var isHovered = node === hoveredNode;
  var isNeighbor = hoveredNeighbors.has(node);
  var isHighlighted = highlightedNodes.has(node);
  var isDetail = detailNode !== null && node === detailNode;

  if (highlightedNodes.size > 0 && !isHighlighted && !isHovered && !isNeighbor && !isDetail) {
    return { x: data.x, y: data.y, color: "#2D2D2D", size: 1.5, label: null, zIndex: 0 };
  }

  if (!isHighlighted && !isHovered && !isNeighbor && !isDetail) return data;

  var res = Object.assign({}, data);

  if (highlightedNodes.size > 0 && isHighlighted) {
    res.color = "#39FF14";
    res.size = 4;
    res.zIndex = 2;
    res.label = data.label;
    res.forceLabel = true;
  }

  if (isDetail) {
    res.color = "#39FF14";
    res.size = 4;
    res.zIndex = 2;
    res.label = data.label;
    res.forceLabel = true;
  }

  if (isHovered) {
    res.color = "#39FF14";
    res.size = 5;
    res.label = data.label;
    res.labelColor = "#4A4A4A";
    res.zIndex = 3;
    res.forceLabel = true;
  } else if (isNeighbor) {
    res.color = res.color === "#39FF14" ? "#39FF14" : "#9CA3AF";
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
  detailsEl.innerHTML = '<p class="output-msg">Loading...</p>';

  var k = parseInt(document.getElementById("top-k-input").value, 10) || 10;
  var types = getAllowedTypes();
  var typesParam = types.length < 3 ? "&types=" + types.join(",") : "";

  try {
    var resp = await fetch(
      "/api/neighbors/" + encodeURIComponent(nodeId) + "?k=" + k + typesParam
    );
    var data = await resp.json();

    if (data.error) {
      detailsEl.innerHTML = '<p class="output-msg">' + escapeHtml(data.error) + '</p>';
      return;
    }

    var html = '<div class="node-header">';
    html += '<div class="node-title">' + escapeHtml(data.node.label) + '</div>';
    html += '<div class="node-meta">';
    html += '<span>BPM: ' + escapeHtml(String(data.node.bpm)) + '</span>';
    html += '<span>Key: ' + escapeHtml(data.node.key) + '</span>';
    html += '</div>';
    html += '<div class="node-path">' + escapeHtml(data.node.id) + '</div>';
    html += '</div>';

    html += '<div class="neighbors-section">';
    html += '<div class="neighbors-heading">Top ' + data.neighbors.length + ' Mixable Tracks</div>';

    if (data.neighbors.length === 0) {
      html += '<p class="no-neighbors">No compatible neighbours found.</p>';
    } else {
      html += '<table class="neighbors-table">';
      html += '<thead><tr><th>#</th><th>Track</th><th>BPM</th><th>Key</th></tr></thead>';
      html += '<tbody>';
      data.neighbors.forEach(function (nbr, i) {
        var badge = (nbr.edge_type && nbr.edge_type !== "direct")
          ? ' <span class="edge-badge edge-badge--' + escapeHtml(nbr.edge_type) + '">'
            + escapeHtml(nbr.edge_type) + '</span>'
          : '';
        html += '<tr>';
        html += '<td class="col-rank">' + (i + 1) + '</td>';
        html += '<td>' + escapeHtml(nbr.label) + badge + '</td>';
        html += '<td>' + escapeHtml(String(nbr.bpm)) + '</td>';
        html += '<td>' + escapeHtml(nbr.key) + '</td>';
        html += '</tr>';
      });
      html += '</tbody></table>';
    }

    html += '</div>';
    detailsEl.innerHTML = html;
  } catch (err) {
    detailsEl.innerHTML = '<p class="output-msg">Failed to load details.</p>';
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

  // If tree has no children (flat library), hide the whole settings section
  var dirSection = document.getElementById("settings-dir-section");
  if (!tree.children || tree.children.length === 0) {
    if (dirSection) dirSection.style.display = "none";
    return;
  }
  if (dirSection) dirSection.style.display = "";

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
  // Root-level songs ("." directory) are always visible — there is no
  // UI checkbox to exclude them and hiding them would be surprising.
  if (dir === ".") return true;
  if (activeDirectories.has(dir)) return true;
  // Check parent paths
  var idx = dir.lastIndexOf("/");
  while (idx !== -1) {
    var parent = dir.slice(0, idx);
    if (activeDirectories.has(parent)) return true;
    idx = parent.lastIndexOf("/");
  }
  return false;
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

/**
 * Wire autocomplete behaviour onto pre-existing input + results elements.
 * Called both for static inputs (via setupAutocomplete) and for dynamically
 * created waypoint rows.
 */
function wireAutocomplete(inputEl, resultsEl, onSelect) {
  var activeIndex = -1;
  var debounceTimer = null;

  inputEl.addEventListener("input", function () {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () {
      var query = inputEl.value.toLowerCase().trim();
      activeIndex = -1;

      if (query.length < 1) {
        resultsEl.innerHTML = "";
        resultsEl.classList.remove("open");
        return;
      }

      var searchList = filteredSongList || songList;
      var matches = searchList.filter(function (s) {
        return s._lower.indexOf(query) !== -1;
      }).slice(0, 50);

      if (matches.length === 0) {
        resultsEl.innerHTML = "";
        resultsEl.classList.remove("open");
        return;
      }

      var frag = document.createDocumentFragment();
      matches.forEach(function (m) {
        var div = document.createElement("div");
        div.className = "result-item";
        div.textContent = m.label;
        div.addEventListener("mousedown", function (e) {
          e.preventDefault();
          inputEl.value = m.label;
          onSelect(m.id);
          resultsEl.classList.remove("open");
        });
        frag.appendChild(div);
      });

      resultsEl.innerHTML = "";
      resultsEl.appendChild(frag);
      resultsEl.classList.add("open");
    }, 100);
  });

  inputEl.addEventListener("keydown", function (e) {
    var items = resultsEl.querySelectorAll(".result-item");
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
      resultsEl.classList.remove("open");
    }
  });

  inputEl.addEventListener("blur", function () {
    // Delay to allow click on result item
    setTimeout(function () {
      resultsEl.classList.remove("open");
    }, 200);
  });

  inputEl.addEventListener("focus", function () {
    if (inputEl.value.length >= 1 && resultsEl.children.length > 0) {
      resultsEl.classList.add("open");
    }
  });
}

/** Convenience wrapper for statically-rendered inputs addressed by ID. */
function setupAutocomplete(inputId, resultsId, onSelect) {
  wireAutocomplete(
    document.getElementById(inputId),
    document.getElementById(resultsId),
    onSelect
  );
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
// 6. Waypoints
// =========================================================================

/** Dynamically insert a new "Via" search row into the waypoints container. */
function addWaypointRow() {
  var key = waypointCounter++;
  var wp = { key: key, id: null };
  waypoints.push(wp);

  var container = document.getElementById("waypoints-container");

  var row = document.createElement("div");
  row.className = "waypoint-row";
  row.dataset.key = String(key);

  var lbl = document.createElement("label");
  lbl.textContent = "Via";
  row.appendChild(lbl);

  var sw = document.createElement("div");
  sw.className = "search-wrapper";

  var input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Search for a track...";
  input.autocomplete = "off";

  var results = document.createElement("div");
  results.className = "search-results";

  sw.appendChild(input);
  sw.appendChild(results);
  row.appendChild(sw);

  var removeBtn = document.createElement("button");
  removeBtn.className = "waypoint-remove-btn";
  removeBtn.title = "Remove";
  removeBtn.textContent = "\u00D7"; // ×
  removeBtn.addEventListener("click", function () {
    var idx = waypoints.findIndex(function (w) { return w.key === key; });
    if (idx !== -1) waypoints.splice(idx, 1);
    row.remove();
  });
  row.appendChild(removeBtn);

  container.appendChild(row);
  wireAutocomplete(input, results, function (id) { wp.id = id; });
  input.focus();
}

document.getElementById("add-waypoint-btn").addEventListener("click", addWaypointRow);

// =========================================================================
// 7. Pathfinding
// =========================================================================

document.getElementById("find-path-btn").addEventListener("click", async function () {
  var outputEl = document.getElementById("path-output");

  if (!startId || !endId) {
    outputEl.innerHTML = '<p class="output-msg">Please select both a start and destination song.</p>';
    return;
  }

  // Collect ordered waypoint IDs, skipping any that were left blank
  var waypointIds = waypoints.map(function (wp) { return wp.id; }).filter(Boolean);

  var btn = document.getElementById("find-path-btn");
  btn.disabled = true;
  outputEl.innerHTML = '<p class="output-msg">Finding path...</p>';

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
        waypoints: waypointIds.length > 0 ? waypointIds : null,
        allowed_types: allowedTypes.length < 3 ? allowedTypes : null,
        excluded_dirs: excludedDirs,
      }),
    });
    var result = await resp.json();

    if (result.error) {
      outputEl.innerHTML = '<p class="output-msg">' + escapeHtml(result.error) + '</p>';
      clearHighlights();
      return;
    }

    // Update graph highlights
    highlightedNodes.clear();
    highlightedEdges.clear();

    result.path_nodes.forEach(function (nid) {
      if (graphInstance.hasNode(nid)) highlightedNodes.add(nid);
    });

    // Find edge keys in graphology for each path edge pair.
    // (undirected graph — edge() returns the key regardless of argument order)
    // Guard with hasNode() in case the backend graph was updated after the
    // frontend loaded its snapshot (stale node IDs would crash .edge()).
    result.path_edges.forEach(function (pair) {
      if (graphInstance.hasNode(pair[0]) && graphInstance.hasNode(pair[1])) {
        var edgeKey = graphInstance.edge(pair[0], pair[1]);
        if (edgeKey != null) {
          highlightedEdges.add(edgeKey);
        }
      }
    });

    syncZIndex();
    sigmaInstance.refresh();

    // Render path as a vertical flow of track cards with arrows between them
    outputEl.innerHTML = renderPathFlowHtml(result.path_nodes);
  } catch (err) {
    outputEl.innerHTML = '<p class="output-msg">Request failed: ' + escapeHtml(err.message) + '</p>';
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
// 8. Create Setlist
// =========================================================================

// Mutable setlist state (updated when user swaps songs)
var setlistNodes = [];
var activeSetlistIndex = -1;
var currentSetlistSummary = "";
var currentSetlistName = "";

/** Apply graph highlights from a path result (shared with pathfinding). */
function applyPathHighlights(pathNodes, pathEdges) {
  highlightedNodes.clear();
  highlightedEdges.clear();

  pathNodes.forEach(function (nid) {
    if (graphInstance.hasNode(nid)) highlightedNodes.add(nid);
  });
  pathEdges.forEach(function (pair) {
    if (graphInstance.hasNode(pair[0]) && graphInstance.hasNode(pair[1])) {
      var ek = graphInstance.edge(pair[0], pair[1]);
      if (ek != null) highlightedEdges.add(ek);
    }
  });

  syncZIndex();
  sigmaInstance.refresh();
}

document.getElementById("generate-setlist-btn").addEventListener("click", async function () {
  var outputEl = document.getElementById("setlist-output");
  var btn = document.getElementById("generate-setlist-btn");

  var minBpm = parseFloat(document.getElementById("sl-min-bpm").value);
  var maxBpm = parseFloat(document.getElementById("sl-max-bpm").value);
  var targetMin = parseFloat(document.getElementById("sl-target-duration").value) || 60;
  var startingKey = document.getElementById("sl-starting-key").value || null;
  var setKey = document.getElementById("sl-set-key").value || null;

  // Basic client-side validation
  if (!isNaN(minBpm) && !isNaN(maxBpm) && minBpm > maxBpm) {
    outputEl.innerHTML = '<p class="output-msg">Min BPM must be \u2264 Max BPM.</p>';
    return;
  }

  btn.disabled = true;
  outputEl.innerHTML = '<p class="output-msg">Generating setlist\u2026</p>';

  try {
    var resp = await fetch("/api/setlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        min_bpm: isNaN(minBpm) ? 0 : minBpm,
        max_bpm: isNaN(maxBpm) ? 999 : maxBpm,
        target_duration_min: targetMin,
        starting_key: startingKey,
        set_key: setKey,
        allowed_types: getAllowedTypes(),
      }),
    });
    var result = await resp.json();

    if (result.error) {
      outputEl.innerHTML = '<p class="output-msg">' + escapeHtml(result.error) + '</p>';
      return;
    }

    // Store mutable setlist state
    setlistNodes = result.path_nodes.slice();
    currentSetlistSummary = result.summary;
    currentSetlistName = "";
    activeSetlistIndex = -1;

    // Highlight the generated setlist on the graph
    applyPathHighlights(result.path_nodes, result.path_edges);

    // Render: summary + save button, then interactive track flow
    renderSetlistOutput();
  } catch (err) {
    outputEl.innerHTML = '<p class="output-msg">Request failed: ' + escapeHtml(err.message) + '</p>';
  } finally {
    btn.disabled = false;
  }
});

/** (Re-)render the full setlist output using the current setlistNodes array. */
function renderSetlistOutput() {
  var outputEl = document.getElementById("setlist-output");
  outputEl.innerHTML = "";

  // Setlist name input
  var nameRow = document.createElement("div");
  nameRow.className = "setlist-name-row";
  var nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "setlist-name-input";
  nameInput.placeholder = "Setlist name\u2026";
  nameInput.value = currentSetlistName;
  nameInput.addEventListener("input", function () {
    currentSetlistName = nameInput.value;
  });
  nameRow.appendChild(nameInput);
  outputEl.appendChild(nameRow);

  // Header: summary + save button
  var header = document.createElement("div");
  header.className = "setlist-result-header";

  var summaryEl = document.createElement("span");
  summaryEl.className = "setlist-summary";
  summaryEl.textContent = currentSetlistSummary;
  header.appendChild(summaryEl);

  var saveBtn = document.createElement("button");
  saveBtn.className = "save-setlist-btn";
  saveBtn.textContent = "Save Setlist";
  saveBtn.addEventListener("click", async function () {
    var name = nameInput.value.trim() || "My Setlist";
    saveBtn.textContent = "Saving\u2026";
    saveBtn.disabled = true;

    // Remove any previous error
    var prevErr = outputEl.querySelector(".setlist-save-error");
    if (prevErr) prevErr.remove();

    try {
      var resp = await fetch("/api/save_setlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          setlist_name: name,
          track_paths: setlistNodes,
        }),
      });
      var result = await resp.json();

      if (result.error) {
        saveBtn.textContent = "Save Setlist";
        saveBtn.disabled = false;
        var errEl = document.createElement("p");
        errEl.className = "setlist-save-error";
        errEl.textContent = result.error;
        header.after(errEl);
      } else {
        saveBtn.textContent = "Saved \u2713";
        saveBtn.classList.add("save-setlist-btn--saved");
        // stays disabled
      }
    } catch (err) {
      saveBtn.textContent = "Save Setlist";
      saveBtn.disabled = false;
      var errEl = document.createElement("p");
      errEl.className = "setlist-save-error";
      errEl.textContent = "Request failed: " + err.message;
      header.after(errEl);
    }
  });
  header.appendChild(saveBtn);

  outputEl.appendChild(header);
  outputEl.appendChild(buildSetlistFlow(setlistNodes));
}

/** Build the DOM for the setlist path-flow with interactive (clickable) cards. */
function buildSetlistFlow(pathNodes) {
  var flow = document.createElement("div");
  flow.className = "path-flow";

  pathNodes.forEach(function (nid, i) {
    var label = graphInstance.hasNode(nid) ? graphInstance.getNodeAttribute(nid, "label") : nid;
    var bpm   = graphInstance.hasNode(nid) ? graphInstance.getNodeAttribute(nid, "bpm")   : "";
    var key   = graphInstance.hasNode(nid) ? graphInstance.getNodeAttribute(nid, "key")   : "";

    var card = document.createElement("div");
    card.className = "path-flow-card path-flow-card--interactive";
    if (i === activeSetlistIndex) card.classList.add("path-flow-card--active");
    card.dataset.index = String(i);

    var titleEl = document.createElement("div");
    titleEl.className = "path-flow-card-title";
    titleEl.textContent = label;
    card.appendChild(titleEl);

    if (bpm || key) {
      var metaEl = document.createElement("div");
      metaEl.className = "path-flow-card-meta";
      if (bpm) { var s1 = document.createElement("span"); s1.textContent = "BPM: " + bpm; metaEl.appendChild(s1); }
      if (key) { var s2 = document.createElement("span"); s2.textContent = "Key: " + key; metaEl.appendChild(s2); }
      card.appendChild(metaEl);
    }

    var capturedIndex = i;
    var capturedNodeId = nid;
    card.addEventListener("click", function () {
      onSetlistCardClick(capturedIndex, capturedNodeId, flow);
    });

    flow.appendChild(card);

    // Inline neighbor picker (shown when this card is active)
    if (i === activeSetlistIndex) {
      var picker = document.createElement("div");
      picker.className = "setlist-neighbor-picker";
      picker.innerHTML = '<p class="output-msg" style="margin:6px 8px">Loading alternatives\u2026</p>';
      flow.appendChild(picker);
      // Fetch and populate asynchronously
      populateSetlistPicker(picker, capturedNodeId, capturedIndex);
    }

    if (i < pathNodes.length - 1) {
      var nextNid = pathNodes[i + 1];
      var edgeType = "direct";
      if (graphInstance.hasNode(nid) && graphInstance.hasNode(nextNid)) {
        var ek = graphInstance.edge(nid, nextNid);
        if (ek != null) edgeType = graphInstance.getEdgeAttribute(ek, "edge_type") || "direct";
      }
      var arrow = document.createElement("div");
      arrow.className = "path-flow-arrow";
      var badge = document.createElement("span");
      badge.className = "edge-badge edge-badge--" + edgeType;
      badge.textContent = edgeType;
      arrow.appendChild(document.createTextNode("\u2193"));
      arrow.appendChild(badge);
      flow.appendChild(arrow);
    }
  });

  return flow;
}

/** Fetch neighbors and populate the picker panel. */
async function populateSetlistPicker(pickerEl, nodeId, index) {
  try {
    var resp = await fetch("/api/neighbors/" + encodeURIComponent(nodeId) + "?k=8");
    var data = await resp.json();

    if (data.error || !data.neighbors || data.neighbors.length === 0) {
      pickerEl.innerHTML = '<p class="output-msg" style="margin:6px 8px">No alternatives found.</p>';
      return;
    }

    pickerEl.innerHTML = '<div class="setlist-picker-label">Replace with:</div>';

    data.neighbors.forEach(function (nbr) {
      var row = document.createElement("div");
      row.className = "setlist-picker-row";

      var nameEl = document.createElement("span");
      nameEl.className = "setlist-picker-name";
      nameEl.textContent = nbr.label;

      var metaEl = document.createElement("span");
      metaEl.className = "setlist-picker-meta";
      metaEl.textContent = String(nbr.bpm) + " BPM \u00B7 " + nbr.key;

      row.appendChild(nameEl);
      row.appendChild(metaEl);

      row.addEventListener("click", function () {
        replaceSetlistSong(index, nbr.id);
      });

      pickerEl.appendChild(row);
    });
  } catch (_err) {
    pickerEl.innerHTML = '<p class="output-msg" style="margin:6px 8px">Failed to load alternatives.</p>';
  }
}

/** Toggle the neighbor picker for the card at the given index. */
function onSetlistCardClick(index, nodeId, flow) {
  if (activeSetlistIndex === index) {
    // Second click: collapse
    activeSetlistIndex = -1;
  } else {
    activeSetlistIndex = index;
  }
  // Re-render the flow in place (preserves header)
  var outputEl = document.getElementById("setlist-output");
  var oldFlow = outputEl.querySelector(".path-flow");
  var newFlow = buildSetlistFlow(setlistNodes);
  outputEl.replaceChild(newFlow, oldFlow);
}

/** Swap one song in the setlist and re-render. */
function replaceSetlistSong(index, newNodeId) {
  setlistNodes[index] = newNodeId;
  activeSetlistIndex = -1;
  renderSetlistOutput();
  // Highlight nodes only — edges may not align after manual swap
  applyPathHighlights(setlistNodes, []);
}

// =========================================================================
// 9. Tab navigation
// =========================================================================

document.querySelectorAll(".tab-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var tabId = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(function (b) {
      b.classList.remove("active");
    });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.classList.remove("active");
    });
    btn.classList.add("active");
    document.getElementById("tab-" + tabId).classList.add("active");
  });
});

// =========================================================================
// 10. Weight settings
// =========================================================================

(function () {
  var sliders = [
    { id: "weight-harmonic",       valId: "weight-harmonic-val" },
    { id: "weight-tempo",          valId: "weight-tempo-val" },
    { id: "weight-semantic",       valId: "weight-semantic-val" },
    { id: "weight-double-penalty", valId: "weight-double-penalty-val" },
    { id: "weight-triplet-penalty",valId: "weight-triplet-penalty-val" },
  ];

  sliders.forEach(function (s) {
    document.getElementById(s.id).addEventListener("input", function () {
      document.getElementById(s.valId).textContent =
        parseFloat(this.value).toFixed(2);
    });
  });

  document.getElementById("recalculate-btn").addEventListener("click", async function () {
    var btn = this;
    var msgEl = document.getElementById("recalculate-msg");

    btn.disabled = true;
    msgEl.textContent = "Recalculating\u2026";

    try {
      var resp = await fetch("/api/recalculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          harmonic:        parseFloat(document.getElementById("weight-harmonic").value),
          tempo:           parseFloat(document.getElementById("weight-tempo").value),
          semantic:        parseFloat(document.getElementById("weight-semantic").value),
          double_penalty:  parseFloat(document.getElementById("weight-double-penalty").value),
          triplet_penalty: parseFloat(document.getElementById("weight-triplet-penalty").value),
        }),
      });
      var data = await resp.json();

      if (data.error) {
        msgEl.textContent = "Error: " + data.error;
      } else {
        msgEl.textContent = data.message + " (" + data.num_edges + " edges)";
        loadGraph();
      }
    } catch (err) {
      msgEl.textContent = "Request failed.";
    } finally {
      btn.disabled = false;
    }
  });
}());

// =========================================================================
// 11. Bootstrap
// =========================================================================

startPolling();
