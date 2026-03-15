# Optimize Dashboard Graph Plot & Hover Performance

## Step 1 — Cache neighbor/edge sets on hover instead of rebuilding them (app.js:330-342)

**Problem**: `onEnterNode` creates two new Sets (`neighbors()` + `edges()`) and triggers a full `refresh()` on every single mouse-enter event. For high-degree nodes this is expensive.

**Fix**:
- Pre-build a `Map<nodeId, { neighbors: Set, edges: Set }>` once after graph construction (in `loadGraph()` and `reloadGraphData()`).
- In `onEnterNode`, do a simple `Map.get()` lookup instead of rebuilding Sets.
- This turns every hover from O(degree) allocation + iteration into O(1).

---

## Step 2 — Optimize node/edge reducers to avoid unnecessary object copies (app.js:257-324)

**Problem**: Every `refresh()` calls `nodeReducer` on **every** node and `edgeReducer` on **every** edge. Each one does `Object.assign({}, data)` — allocating a new object — even when nothing needs to change.

**Fix**:
- Add early-return fast paths: if no hover, no highlights, and no detailNode are active, return `data` directly (zero allocation).
- Only create the copy (`Object.assign`) when the node/edge actually needs modification.
- In `edgeReducer`, check the two cheap conditions (`highlightedEdges.has` / `hoveredEdgeKeys.has`) first and skip the copy when both are false.

---

## Step 3 — Debounce the search input (app.js:421-434)

**Problem**: The `input` event fires on every keystroke with no throttling. Each fires an O(n) `songList.filter()` and a full DOM clear + rebuild.

**Fix**:
- Wrap the input handler in a simple debounce (e.g. 100ms via `setTimeout` / `clearTimeout`).
- Use a `DocumentFragment` for batch DOM insertion instead of individual `appendChild` calls.

---

## Step 4 — Eliminate double edge lookup in path highlighting (app.js:546-560)

**Problem**: For each path edge pair, the code does `graphInstance.edge(src, dst)` and then unconditionally also tries `graphInstance.edge(dst, src)`, even for an undirected graph where only one lookup is needed.

**Fix**:
- Use `graphInstance.hasUndirectedEdge(src, dst)` or try one direction and skip the reverse if the first succeeds. Since graphology is undirected here, `edge(src, dst)` already returns the key regardless of direction — the second lookup is redundant.

---

## Step 5 — Move `has_edge_type` check outside the edge serialization loop (app.py:259)

**Problem**: `"edge_type" in graph.graph.es.attributes()` is evaluated conceptually once but sits in a position that's easy to misread. More importantly, the edge loop itself builds Python dicts one-by-one for potentially tens of thousands of edges.

**Fix**:
- Hoist the `has_edge_type` check (already outside the loop — confirmed on re-read, this is fine).
- Vectorize edge serialization: pull `graph.graph.es["weight"]`, `graph.graph.vs["name"]`, and edge type attributes as bulk lists, then zip them into the response. This avoids per-edge attribute lookups on igraph objects.

---

## Step 6 — Add HTTP caching headers to `/api/graph` and `/api/songs` (app.py:238, 286)

**Problem**: Every call to `/api/graph` re-serializes the entire graph. On page refresh or after recalculate, the browser re-fetches the full payload even if nothing changed.

**Fix**:
- Track a graph version counter (incremented on recalculate / incremental sync).
- Return `ETag` header based on the version.
- Support `If-None-Match` → 304 Not Modified, avoiding re-serialization and re-transfer.

---

## Order of implementation

1. Step 1 (hover cache) — highest impact on perceived interactivity
2. Step 2 (reducer fast paths) — directly compounds with Step 1
3. Step 3 (search debounce + fragment) — easy, independent
4. Step 4 (edge lookup dedup) — easy, independent
5. Step 5 (bulk edge serialization) — backend perf
6. Step 6 (HTTP caching) — backend perf
