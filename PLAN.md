# Fix Plan: Directory Filtering + Pathfinding Bugs

## Bug 1: Pathfinding broken — "Song not found"

**Symptom:** `get_song()` receives a valid file_path but can't find it in `_songs`.

**Root causes:**

### 1a. Thread-safety during incremental sync
`app.py` lines 155-165: The incremental sync calls `graph.remove_songs()` and
`graph.add_songs_incremental()` **without holding `_graph_lock`**, while the API
endpoints may be reading the same graph concurrently. A request hitting
`/api/songs` during mutation could return stale or inconsistent song data.

**Fix:** Wrap the incremental sync's remove + add operations (and the subsequent
cache save) inside `_graph_lock` so API reads are blocked during mutation.

### 1b. Unresolved `SONGS_DIRECTORY` path
`src/config.py` line 19 builds the path as `_PROJECT_ROOT / ".." / 'Playlists'`
which contains `..`. This unresolved path is passed verbatim to
`get_shortest_path(songs_directory=SONGS_DIRECTORY)` at `app.py:434`.
Inside `get_shortest_path`, vertex file_paths (which **are** resolved) are
compared via `Path.relative_to(base)` where `base = Path(songs_directory).resolve()`.
While `.resolve()` is called inside the method, it's called per-vertex on every
pathfinding request. Fix by resolving once at config time.

**Fix:** In `src/config.py`, resolve the default path:
```python
str((_PROJECT_ROOT / ".." / 'Playlists').resolve())
```

### 1c. Fragile `get_song()` lookup
`get_song()` in `src/graph.py:638-658` only tries exact `file_path` match and
exact `filename` match. If the stored file_path diverges from what the API
sent (e.g. due to a stale cache from a moved directory), it raises `KeyError`.

**Fix:** Add a fallback that normalises both sides (e.g. `Path(identifier).resolve()`)
before giving up. This makes the lookup resilient to trivial path representation
differences.

---

## Bug 2: Directory filtering does not work

**Symptom:** Unchecking directories in the sidebar doesn't visually filter nodes
or affect pathfinding.

**Root causes:**

### 2a. Root-level songs always hidden when filtering is active
The "All Songs" root tree node has no `path` property (`app.py:253` sets
`"path"` only on child nodes, not on root). `collectAllDirs()` in `app.js:580`
skips nodes without `path`, so `"."` never appears in `allDirs`. Since the root
checkbox is never rendered (line 681 only iterates `tree.children`), the `"."` path
is never in `checkedPaths` either. Therefore `isDirectoryActive(".")` always
returns `false` when filtering is active, hiding all root-level songs.

**Fix (app.js):**
- In `renderDirectoryTree()`, render the root "All Songs" node as a proper
  checkbox with `path: "."` at depth 0, then render its children at depth 1+.
  This gives users a way to toggle root-level songs and ensures `"."` is in
  `allDirs` and `checkedPaths`.

### 2b. `excluded_dirs` may not reach the backend or may mismatch
The frontend builds `excludedDirs` from unchecked tree node `path` values
(e.g. `"BCHT"`, `"Other"`). These are sent to the backend's
`get_shortest_path()` which computes `rel_parent` for each vertex using
`Path(v["name"]).resolve().relative_to(base)`. If `base` (from
`SONGS_DIRECTORY`) doesn't resolve identically to the base used when building
the tree, the relative paths won't match and exclusion silently fails.

**Fix:** Ensure `SONGS_DIRECTORY` is resolved consistently (see Fix 1b). Also,
in `_build_directory_tree()`, use the exact same resolved base that the
frontend receives via `songs_directory` in the API response.

### 2c. Songs outside `SONGS_DIRECTORY` not filterable
`_build_directory_tree()` catches `ValueError` from `relative_to()` and puts
such songs at root (`"."`). But `get_shortest_path()` also catches `ValueError`
and sets `rel_parent = "."`. These songs can't be excluded because `"."` is
never in `excluded_dirs` (the root checkbox doesn't exist — see 2a).

**Fix:** Already addressed by 2a (rendering root checkbox).

---

## Files to modify

| File | Changes |
|------|---------|
| `src/config.py` | Resolve `SONGS_DIRECTORY` default path |
| `src/graph.py` | Harden `get_song()` with path normalisation fallback |
| `app.py` | Wrap incremental sync in `_graph_lock` |
| `static/app.js` | Render root "All Songs" checkbox; include `"."` in `allDirs` |
| `tests/test_models.py` | Add `fingerprint=""` to `AudioAnalysis` constructions (3 places) |
