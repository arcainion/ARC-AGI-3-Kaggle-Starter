# FORGE v31 — v30 + fused Cython state/reward update
# =====================================================================
# FORGE v20 — v19 base + targeted performance upgrades
#
# v26 additional performance changes:
# - Compact uint8 replay frames plus parallel scalar arrays; batch features are
#   built on-device, eliminating the very large per-state CPU feature cache.
# - Cache ActionEffectAttention's encoded memory until a new effect or optimizer
#   update invalidates it.
# - Prefer fused Adam/channels-last on CUDA and skip the click decoder for
#   direction-only training batches.
# - Configure the safe BFS clone backend before action scanning and reuse
#   immutable no-payload ActionInput instances in the search hot path.
#
# Changes on top of v19:
#
# FIX 1: _visited_hashes was never initialized in __init__ — reward
#         signal was broken: always gave +1.5 for ANY hash change,
#         never penalizing loops. Now properly tracks and deduplicates.
#
# FIX 2: CLTI frame extraction used get_pixels() which is inconsistent
#         with _raw() (which reads frame[-1] from perform_action).
#         Now uses perform_action result frames throughout, so injected
#         expert demos have correct state representations.
#
# FIX 3: BFS hidden retry used 3 RESET calls instead of 2, landing
#         in a different initial state than the first pass scan,
#         causing the retry to search from a mismatched baseline.
#
# FIX 4: Epsilon always reset to 0.15 on level change even when BFS
#         already solved the level. Now only resets if BFS failed,
#         preserving learned exploration for CNN fallback.
# =====================================================================
import copy
import gc
import pickle
from contextlib import contextmanager, nullcontext
import glob
import hashlib
import zlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque
from array import array

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

try:
    from sprite_detector import detect_sprites as _detect_sprites_helper
except Exception:
    try:
        from agent.sprite_detector import detect_sprites as _detect_sprites_helper
    except Exception:
        _detect_sprites_helper = None

logger = logging.getLogger(__name__)


def _frame_view(frame, dtype=None):
    """Return a NumPy view of a frame whenever possible.

    ARC action results commonly already contain a contiguous NumPy frame.  Using
    np.array(...) at every BFS expansion needlessly allocates/copies a 64x64
    array before hashing.  astype(copy=False) only copies when a caller truly
    requires a different dtype.
    """
    arr = np.asarray(frame)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _frame_crc(frame):
    """Fast non-cryptographic frame hash for local effect/replay checks."""
    try:
        arr = np.ascontiguousarray(frame)
        return zlib.crc32(memoryview(arr).cast('B')) & 0xffffffff
    except Exception:
        return zlib.crc32(np.asarray(frame).tobytes()) & 0xffffffff


def _frame_signature(frame):
    """Compact 64-bit-like transposition signature for BFS visited states.

    Two independent C-level checksums avoid allocating a hexadecimal string for
    every expanded state.  This is intentionally separate from _frame_crc(),
    which remains suitable for local ACTION6 effect de-duplication.
    """
    arr = np.ascontiguousarray(frame)
    data = memoryview(arr).cast('B')
    return (
        zlib.crc32(data) & 0xffffffff,
        zlib.adler32(data) & 0xffffffff,
    )

@contextmanager
def _paused_bfs_gc():
    """Temporarily pause cyclic-GC during allocation-heavy BFS.

    BFS creates and releases a very large number of short-lived snapshots.
    Reference counting still reclaims ordinary objects immediately; pausing only
    avoids repeated cyclic-GC scans.  GC is restored even if a game raises.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
            # A young-generation sweep at the boundary prevents stale cycles from
            # accumulating across levels without turning every expansion into GC work.
            gc.collect(0)


class _ImplicitSearchGraph:
    """Compact graph metadata plus ephemeral state slots for implicit BFS.

    The game transition graph is discovered only by simulating an action, so a
    conventional explicit graph library cannot remove the expensive simulator
    call.  This structure removes *Python graph bookkeeping* from the hot
    path: the frontier stores integer node IDs; node metadata lives in compact
    typed arrays; and a full game snapshot remains referenced only until its
    frontier node is expanded or pruned.
    """

    __slots__ = (
        'parents', 'action_ids', 'payloads', 'depths', 'last_actions', 'states',
    )

    def __init__(self, root_state):
        self.parents = array('i', [-1])
        self.action_ids = array('B', [0])
        self.payloads = [None]
        self.depths = array('B', [0])
        self.last_actions = array('B', [0])
        self.states = [root_state]

    @staticmethod
    def _compact_payload(act_id, data):
        if not data:
            return None
        # ACTION6 is the only high-volume payload in ARC games.  Store its
        # coordinates as two integers instead of a Python dictionary.
        if int(act_id) == 6 and 'x' in data and 'y' in data:
            return (int(data['x']), int(data['y']))
        return dict(data)

    @staticmethod
    def _restore_payload(act_id, payload):
        if payload is None:
            return None
        if int(act_id) == 6 and isinstance(payload, tuple):
            return {'x': payload[0], 'y': payload[1], 'game_id': 'bfs'}
        return dict(payload)

    def add_child(self, parent_idx, act_id, data, state):
        self.parents.append(int(parent_idx))
        self.action_ids.append(int(act_id))
        self.payloads.append(self._compact_payload(act_id, data))
        self.depths.append(min(255, int(self.depths[parent_idx]) + 1))
        self.last_actions.append(int(act_id))
        self.states.append(state)
        return len(self.parents) - 1

    def take_state(self, node_idx):
        """Consume the only full game snapshot held by an expanded node."""
        state = self.states[node_idx]
        self.states[node_idx] = None
        return state

    def discard_state(self, node_idx):
        """Release a pruned frontier snapshot while retaining compact metadata."""
        self.states[node_idx] = None

    def get_depth(self, node_idx):
        return int(self.depths[node_idx])

    def get_last_action(self, node_idx):
        return int(self.last_actions[node_idx])

    def reconstruct(self, node_idx):
        path = []
        while node_idx > 0:
            act_id = int(self.action_ids[node_idx])
            path.append((act_id, self._restore_payload(act_id, self.payloads[node_idx])))
            node_idx = int(self.parents[node_idx])
        path.reverse()
        return path

# ==================== BFS SOLVER ====================
class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        # Hidden-state retry is useful on a few games, but it can double BFS cost.
        # Gate it behind strong evidence and a small remaining-time budget.
        self.hidden_retry_min_explored = 200
        self.hidden_retry_unique_ratio = 0.08
        self.hidden_retry_time_cap = 8.0
        # BFS can accumulate thousands of full Python game snapshots in the queue.
        # Cap only when it gets excessive; this protects memory/GC without changing
        # normal small searches.  FIFO retention preserves the shallowest BFS nodes.
        self.max_bfs_queue = 10000
        self.max_bfs_queue_retry = 5000
        self._last_queue_trim_log = 0.0
        # GameAction.from_id() is called for every expanded branch.  Cache both
        # the enum lookup and immutable no-payload ActionInput objects.  Click
        # inputs remain freshly constructed because their coordinates vary.
        self._action_ids = {}
        self._plain_action_inputs = {}
        # Most ARC game classes only expose Python-object snapshots, so deepcopy
        # is the safe default.  Some classes support a faster native clone or
        # pickle round-trip, however.  Select one only after an isolation check;
        # otherwise stay with deepcopy.
        self._clone_fn = copy.deepcopy
        self._clone_backend = 'deepcopy'
        self._clone_backend_ready = False
        # Traversal metadata: compact inverse lookup and root-effect priorities
        # improve sibling ordering while preserving BFS depth optimality.
        self._opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3}
        self._action_priority = {}
        self.max_bfs_depth = 30
        self._queue_cls = deque
        self._graph_cls = _ImplicitSearchGraph
        # Optional snapshot/restore BFS.  This is the generic safe approximation of
        # undo/delta search for arbitrary Python ARC game classes: store one pickle
        # snapshot per frontier node, restore it for each outgoing action, and only
        # serialize a child once if it survives visited/depth pruning.  It avoids
        # serializing the same parent once per branch when pickle snapshots are safe.
        self._snapshot_bfs_ready = False
        self._snapshot_bfs_enabled = False
        self._snapshot_protocol = pickle.HIGHEST_PROTOCOL
        # Pick best safe branch strategy at runtime.
        self._snapshot_branch_mode = 'restore_each'

    def load(self):
        """Load the game class from source."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False


    def _clone_game(self, game):
        """Clone a game through the validated per-class backend."""
        return self._clone_fn(game)

    def _visible_clone_frame(self, game):
        """Best-effort frame used only while validating a clone backend."""
        try:
            return _frame_view(game.get_pixels(0, 0, 64, 64))
        except Exception:
            return None

    def _validate_clone_backend(self, clone_fn, game, probe_action):
        """Check that a candidate clone starts equivalent and cannot mutate its parent.

        The check runs against a disposable deepcopy, never the active BFS root.
        A backend is accepted only if the clone is separate, begins with the same
        visible frame when available, and a simulated action leaves the parent
        byte-identical under pickle serialization.
        """
        try:
            seed = copy.deepcopy(game)
            parent_before = pickle.dumps(seed, protocol=pickle.HIGHEST_PROTOCOL)
            parent_frame = self._visible_clone_frame(seed)

            child = clone_fn(seed)
            if child is seed or not isinstance(child, type(seed)):
                return False
            child_frame = self._visible_clone_frame(child)
            if parent_frame is not None and child_frame is not None:
                if parent_frame.shape != child_frame.shape or not np.array_equal(parent_frame, child_frame):
                    return False

            if probe_action is not None:
                act_id, data = probe_action
                child.perform_action(self._make_action(act_id, data), raw=True)

            # If a native clone shares mutable state, this comparison catches it
            # before that backend is ever used for live BFS nodes.
            parent_after = pickle.dumps(seed, protocol=pickle.HIGHEST_PROTOCOL)
            return parent_before == parent_after
        except Exception:
            return False

    def _benchmark_clone_backend(self, clone_fn, game, repeats=2):
        try:
            t0 = time.perf_counter()
            for _ in range(repeats):
                clone_fn(game)
            return (time.perf_counter() - t0) / repeats
        except Exception:
            return float('inf')

    def _pickle_snapshot(self, game):
        return pickle.dumps(game, protocol=self._snapshot_protocol)

    def _restore_snapshot(self, blob):
        return pickle.loads(blob)

    def _validate_snapshot_bfs(self, game, probe_actions):
        """Validate pickle snapshot/restore as an undo-style BFS backend.

        This deliberately avoids assuming any ARC game internals.  A snapshot
        backend is accepted only if a restored child starts equivalent, mutating
        that child does not change the root snapshot, and the restored object has
        the same concrete type.
        """
        try:
            root_blob = self._pickle_snapshot(game)
            child = self._restore_snapshot(root_blob)
            if child is game or not isinstance(child, type(game)):
                return False
            parent_frame = self._visible_clone_frame(game)
            child_frame = self._visible_clone_frame(child)
            if parent_frame is not None and child_frame is not None:
                if parent_frame.shape != child_frame.shape or not np.array_equal(parent_frame, child_frame):
                    return False
            if probe_actions:
                act_id, data = probe_actions[0]
                child.perform_action(self._make_action(act_id, data), raw=True)
            return root_blob == self._pickle_snapshot(game)
        except Exception:
            return False

    def _benchmark_snapshot_bfs(self, game, actions, repeats=2):
        """Benchmark clone BFS against two snapshot branch strategies.

        ``restore_each`` restores the serialized parent once per outgoing edge.
        ``restore_once_clone`` restores the parent once per expanded node and
        then uses the validated clone backend for each sibling.  The latter is a
        closer generic approximation of undo/delta BFS when pickle.loads() is
        expensive but native/Cython clone is cheap.
        """
        if not actions:
            return False
        probes = actions[:min(3, len(actions))]
        repeats = max(1, int(repeats))
        try:
            parent_blob = self._pickle_snapshot(game)

            t0 = time.perf_counter()
            for _ in range(repeats):
                for act_id, data in probes:
                    g2 = self._clone_game(game)
                    g2.perform_action(self._make_action(act_id, data), raw=True)
            clone_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(repeats):
                for act_id, data in probes:
                    g2 = self._restore_snapshot(parent_blob)
                    g2.perform_action(self._make_action(act_id, data), raw=True)
                    self._pickle_snapshot(g2)
            restore_each_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(repeats):
                parent_game = self._restore_snapshot(parent_blob)
                for act_id, data in probes:
                    g2 = self._clone_game(parent_game)
                    g2.perform_action(self._make_action(act_id, data), raw=True)
                    self._pickle_snapshot(g2)
            restore_once_clone_t = time.perf_counter() - t0

            best_mode, best_t = min(
                (('restore_each', restore_each_t), ('restore_once_clone', restore_once_clone_t)),
                key=lambda kv: kv[1],
            )
            if best_t < clone_t * 0.92:
                self._snapshot_branch_mode = best_mode
                logger.info(
                    'BFS snapshot backend enabled: %s (%.2f ms vs clone %.2f ms per probe batch)',
                    best_mode, best_t * 1000.0 / repeats, clone_t * 1000.0 / repeats,
                )
                return True
            logger.info(
                'BFS snapshot backend rejected (restore_each %.2f ms, restore_once_clone %.2f ms, clone %.2f ms per probe batch)',
                restore_each_t * 1000.0 / repeats,
                restore_once_clone_t * 1000.0 / repeats,
                clone_t * 1000.0 / repeats,
            )
        except Exception as e:
            logger.info('BFS snapshot backend unavailable: %s', e)
        return False

    def _configure_snapshot_bfs(self, game, actions):
        if self._snapshot_bfs_ready:
            return self._snapshot_bfs_enabled
        self._snapshot_bfs_ready = True
        self._snapshot_bfs_enabled = False
        if not actions:
            return False
        if not self._validate_snapshot_bfs(game, actions):
            return False
        self._snapshot_bfs_enabled = self._benchmark_snapshot_bfs(game, actions)
        return self._snapshot_bfs_enabled

    def _solve_bfs_snapshot(self, game, f0, level_idx, actions, effective_timeout, max_states,
                            hidden_fields=None, cap=None, label=None):
        """BFS using serialized snapshots as frontier states.

        This is an undo/delta-style generic backend: the frontier holds compact
        state snapshots instead of live game objects.  For each edge we restore
        the parent snapshot, simulate one action, and store the child snapshot
        only after it passes pruning.  It preserves the same BFS semantics as the
        clone backend and is only called after validation/benchmark selection.
        """
        make_action = self._make_action
        last_frame = self._last_frame
        state_hash = self._state_hash
        is_complete = self._is_complete
        cap = self.max_bfs_queue if cap is None else cap
        label = f'L{level_idx}' if label is None else label

        visited = set()
        queue = self._queue_cls()
        h0 = state_hash(game, f0, hidden_fields)
        visited.add(h0)
        graph = self._graph_cls(self._pickle_snapshot(game))
        queue.append(0)
        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < effective_timeout:
            node_idx = queue.popleft()
            parent_blob = graph.take_state(node_idx)
            if parent_blob is None:
                continue
            depth = graph.get_depth(node_idx)
            last_act = graph.get_last_action(node_idx) or None

            parent_game = None
            if self._snapshot_branch_mode == 'restore_once_clone':
                try:
                    parent_game = self._restore_snapshot(parent_blob)
                except Exception:
                    continue

            for act_id, data in actions:
                if last_act is not None and self._opposite_actions.get(last_act) == act_id:
                    continue
                try:
                    if parent_game is not None:
                        g2 = self._clone_game(parent_game)
                    else:
                        g2 = self._restore_snapshot(parent_blob)
                    r = g2.perform_action(make_action(act_id, data), raw=True)
                except Exception:
                    continue
                explored += 1

                if is_complete(g2, r, level_idx):
                    child_idx = graph.add_child(node_idx, act_id, data, None)
                    new_hist = self._reconstruct_solution(graph, child_idx)
                    elapsed = time.time() - t0
                    logger.info(f'BFS {label}: SOLVED via snapshot backend in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)')
                    self.solutions[level_idx] = new_hist
                    return new_hist, explored, len(visited), elapsed

                if depth >= self.max_bfs_depth:
                    continue
                f = last_frame(r)
                if f is None:
                    continue
                h = state_hash(g2, f, hidden_fields)
                if h in visited:
                    continue
                visited.add(h)
                try:
                    child_blob = self._pickle_snapshot(g2)
                except Exception:
                    # If a later state becomes unpicklable, abort this backend and
                    # let the caller fall back to clone-BFS on future levels.
                    self._snapshot_bfs_enabled = False
                    logger.info(f'BFS {label}: snapshot backend disabled; child state became unpicklable')
                    return None, explored, len(visited), time.time() - t0
                child_idx = graph.add_child(node_idx, act_id, data, child_blob)
                queue.append(child_idx)
                if len(queue) > cap * 2:
                    self._trim_frontier_if_needed(queue, graph, cap, f'{label} snapshot')

        elapsed = time.time() - t0
        logger.info(f'BFS {label}: snapshot pass failed ({explored} explored, {len(visited)} unique, {elapsed:.1f}s)')
        return None, explored, len(visited), elapsed

    def _configure_clone_backend(self, game, actions):
        """Choose a faster safe clone implementation once per solver instance.

        `deepcopy` remains the guaranteed fallback.  The optional candidates are
        validated for independent state and benchmarked before being selected.
        This makes the optimisation opportunistic: unsupported ARC games retain
        exactly the prior behaviour.
        """
        if self._clone_backend_ready:
            return
        self._clone_backend_ready = True

        probe_action = actions[0] if actions else None
        baseline = self._benchmark_clone_backend(copy.deepcopy, game)
        best_name = 'deepcopy'
        best_fn = copy.deepcopy
        best_time = baseline

        candidates = []
        # Pickle can be markedly faster than recursive deepcopy for plain Python
        # game state.  It is only considered when round-tripping succeeds.
        candidates.append((
            'pickle',
            lambda g: pickle.loads(pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)),
        ))
        # Use an explicitly supplied native clone only; do not try copy.copy,
        # because shallow copies can silently share mutable board state.
        native_clone = getattr(game, 'clone', None)
        if callable(native_clone):
            candidates.append(('native_clone', lambda g: getattr(g, 'clone')()))

        for name, fn in candidates:
            if not self._validate_clone_backend(fn, game, probe_action):
                continue
            candidate_time = self._benchmark_clone_backend(fn, game)
            # Avoid a backend change for noise-level differences; the alternate
            # path needs to be clearly faster before it is selected.
            if candidate_time < best_time * 0.92:
                best_name, best_fn, best_time = name, fn, candidate_time

        self._clone_backend = best_name
        self._clone_fn = best_fn
        if baseline != float('inf'):
            logger.info(
                'BFS clone backend: %s (%.2f ms vs deepcopy %.2f ms)',
                best_name, best_time * 1000.0, baseline * 1000.0,
            )

    def _make_action(self, act_id, data=None):
        act_id = int(act_id)
        try:
            game_action = self._action_ids[act_id]
        except KeyError:
            game_action = GameAction.from_id(act_id)
            self._action_ids[act_id] = game_action
        if data:
            # Payloads can be mutated by game implementations, so never cache them.
            return ActionInput(id=game_action, data=data)
        try:
            return self._plain_action_inputs[act_id]
        except KeyError:
            action_input = ActionInput(id=game_action)
            self._plain_action_inputs[act_id] = action_input
            return action_input

    def _last_frame(self, result):
        if result is not None and getattr(result, 'frame', None):
            return _frame_view(result.frame[-1])
        return None

    def _is_complete(self, g, r, level_idx):
        return (getattr(r, 'levels_completed', 0) > level_idx) or (getattr(g, '_current_level_index', level_idx) > level_idx)

    def _reconstruct_solution(self, graph, node_idx):
        """Rebuild a solution only when a terminal node is found."""
        return graph.reconstruct(node_idx)

    def _trim_frontier_if_needed(self, frontier, graph, cap, label):
        """Trim an implicit-graph frontier and release dropped snapshots.

        Metadata remains as compact arrays because surviving descendants may
        still reference their parents.  The expensive game objects for dropped
        queue entries are immediately released.
        """
        if len(frontier) <= cap * 2:
            return
        before = len(frontier)
        keep_count = min(cap, before)
        kept = deque()
        for _ in range(keep_count):
            kept.append(frontier.popleft())
        while frontier:
            graph.discard_state(frontier.popleft())
        frontier.extend(kept)
        now = time.time()
        if now - self._last_queue_trim_log > 5.0:
            logger.info(f"BFS {label}: trimmed frontier {before} -> {len(frontier)} to limit memory/GC")
            self._last_queue_trim_log = now

    def _action_key(self, act_id, data):
        """Hashable action identity for static sibling ordering."""
        if not data:
            return (int(act_id),)
        return (int(act_id), int(data.get('x', -1)), int(data.get('y', -1)))

    def _ordered_actions(self, actions):
        """Order siblings by observed root-state effect without changing depth.

        BFS still explores all nodes at depth d before d+1.  Reordering only
        makes promising siblings run earlier, which can find an equal-length
        solution sooner and avoids arbitrary enum-order bias.
        """
        return sorted(
            actions,
            key=lambda item: (
                -self._action_priority.get(self._action_key(item[0], item[1]), 0),
                int(item[0]),
                -1 if not item[1] else int(item[1].get('y', -1)),
                -1 if not item[1] else int(item[1].get('x', -1)),
            ),
        )

    def _cnn_sorted_actions(self, actions, net, frame_tensor, device):
        """Sort actions by CNN logit score (highest first) for guided BFS.
        Falls back to original order on any error."""
        if net is None or frame_tensor is None or not actions:
            return actions
        try:
            with torch.inference_mode():
                inp = frame_tensor.unsqueeze(0).to(device)
                has_click = any(a[0] == 6 for a in actions)
                if has_click:
                    logits = net(inp).squeeze(0)
                else:
                    logits = net.forward_actions(inp).squeeze(0)
            def score(a):
                aid, d = a
                if aid <= 5:
                    return logits[aid - 1].item()
                if d:
                    return logits[5 + d.get('y', 0) * 64 + d.get('x', 0)].item()
                return float('-inf')
            return sorted(actions, key=score, reverse=True)
        except Exception:
            return actions

    def _state_hash(self, g, frame, hidden_fields=None):
        """Hash visible frame plus optional hidden scalar state compactly."""
        signature = _frame_signature(frame)
        if not hidden_fields:
            return signature
        extras = []
        for field_name in hidden_fields:
            try:
                v = getattr(g, field_name, None)
                if v is not None:
                    # Hidden fields are probed as scalar numbers, so this tuple is
                    # immutable, compact, and directly usable by the visited set.
                    extras.append((field_name, v))
            except Exception:
                pass
        return (signature, tuple(extras)) if extras else signature

    def _probe_hidden_fields(self, game, actions):
        """Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        changing_fields = set()
        # Probe only a small, diverse prefix.  This avoids spending lots of time
        # deep-copying for hidden-state detection after a failed BFS pass.
        probe_actions = actions[:min(6, len(actions))]
        for act_id, data in probe_actions:
            g = self._clone_game(game)
            try:
                r = g.perform_action(self._make_action(act_id, data), raw=True)
                if self._last_frame(r) is None:
                    continue
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)

        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _click_candidates(self, frame, bg, max_candidates=160):
        """Prioritised ACTION6 candidates."""
        ordered = []
        seen = set()

        def add(x, y):
            x = int(max(0, min(63, round(x))))
            y = int(max(0, min(63, round(y))))
            key = (x, y)
            if key not in seen:
                seen.add(key)
                ordered.append(key)

        # Object-derived clicks, smallest/most specific objects first.
        objects = []
        cnt = np.bincount(frame.ravel(), minlength=16)
        for c in range(16):
            if c == bg or cnt[c] == 0 or cnt[c] > 3000:
                continue
            ys, xs = np.where(frame == c)
            if len(xs) == 0:
                continue
            objects.append((len(xs), c, xs, ys))
        objects.sort(key=lambda t: t[0])

        for _, _c, xs, ys in objects:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx, cy = float(xs.mean()), float(ys.mean())
            mx, my = float(np.median(xs)), float(np.median(ys))
            for x, y in (
                (mx, my), (cx, cy),
                ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1),
                (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2),
                (x0, y0), (x1, y0), (x0, y1), (x1, y1),
            ):
                add(x, y)
                if len(ordered) >= max_candidates:
                    return ordered

        # Sparse foreground fallback catches irregular sprites/large shapes.
        ys, xs = np.where(frame != bg)
        if len(xs):
            stride = max(1, len(xs) // 64)
            for x, y in zip(xs[::stride], ys[::stride]):
                add(x, y)
                if len(ordered) >= max_candidates:
                    return ordered

        # Very coarse grid fallback for click-sensitive empty areas.
        for y in range(4, 64, 8):
            for x in range(4, 64, 8):
                add(x, y)
                if len(ordered) >= max_candidates:
                    return ordered

        return ordered

    def _scan_actions(self, game, f0, bg):
        """Scan effective actions and record a cheap static effect priority."""
        avail = game._available_actions
        clone_game = self._clone_game
        actions = []
        self._action_priority = {}
        # Directional/interact actions
        for a in [a for a in avail if a <= 5]:
            g = clone_game(game)
            try:
                r = g.perform_action(self._make_action(a), raw=True)
                f = self._last_frame(r)
                if f is not None:
                    delta = int(np.count_nonzero(f0 != f))
                    if delta:
                        actions.append((a, None))
                        self._action_priority[self._action_key(a, None)] = delta
            except:
                pass
        # Click actions: prioritised candidate list instead of brute 32x32 scan.
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            candidates = self._click_candidates(f0, bg, max_candidates=80)
            tested = 0
            for x, y in candidates:
                if time.time() - t0 > self.scan_timeout:
                    break
                # Prefer non-background candidates, but keep coarse fallback clicks.
                if f0[y, x] == bg and tested < 96:
                    continue
                tested += 1
                g = clone_game(game)
                try:
                    r = g.perform_action(
                        self._make_action(6, {'x': x, 'y': y, 'game_id': 'bfs'}),
                        raw=True
                    )
                    f = self._last_frame(r)
                    if f is None:
                        continue
                    delta = int(np.count_nonzero(f0 != f))
                    if delta:
                        effect_hash = _frame_crc(f)
                        if effect_hash not in seen_effects:
                            seen_effects.add(effect_hash)
                            click_data = {'x': x, 'y': y, 'game_id': 'bfs'}
                            actions.append((6, click_data))
                            self._action_priority[self._action_key(6, click_data)] = delta
                except:
                    pass
        return actions

    def solve_level(self, level_idx, max_states=150000, prev_solution=None, timeout=None,
                     net=None, frame_tensor=None):
        """Run the allocation-heavy search with cyclic-GC paused."""
        with _paused_bfs_gc():
            return self._solve_level_impl(level_idx, max_states=max_states,
                                          prev_solution=prev_solution, timeout=timeout,
                                          net=net, frame_tensor=frame_tensor)

    def _solve_level_impl(self, level_idx, max_states=500000, prev_solution=None, timeout=None,
                          net=None, frame_tensor=None):
        """Find an action sequence with BFS.
        If timeout is provided, it overrides self.bfs_timeout for this call."""
        if not self.game_cls:
            return None

        # Use provided timeout or fall back to default
        effective_timeout = timeout if timeout is not None else self.bfs_timeout

        # Bind hot callables once.  These names sit in the innermost BFS loop,
        # where repeated attribute/module lookups add measurable Python overhead.
        clone_game = self._clone_game
        make_action = self._make_action
        last_frame = self._last_frame
        state_hash = self._state_hash
        is_complete = self._is_complete

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        f0 = last_frame(r0)
        if f0 is None:
            return None
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Try solution transfer from previous level first
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        # Phase 1: Choose a safe clone backend before scanning.  The scanner can
        # otherwise spend most of its click probes in recursive deepcopy.
        probe_actions = [(a, None) for a in game._available_actions if int(a) <= 5]
        self._configure_clone_backend(game, probe_actions)
        clone_game = self._clone_game
        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock for locked initial states (sc25-type)
        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = clone_game(game)
                try:
                    rw = g_warmup.perform_action(make_action(warmup_id), raw=True)
                    f_after = last_frame(rw)
                    if f_after is None:
                        continue
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}! {len(warmup_actions)} actions")
                        game = g_warmup; f0 = f_after; actions = warmup_actions
                        break
                except:
                    pass

        # Keep breadth-first depth semantics, but expand stronger observed
        # effects earlier within each depth layer.
        actions = self._ordered_actions(actions)
        # CNN-guided: further sort so the most promising actions are tried first
        if net is not None:
            actions = self._cnn_sorted_actions(actions, net, frame_tensor, next(net.parameters()).device)
        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        # Phase 2A: optional snapshot/restore BFS.  This is the generic safe
        # version of undo/delta search for arbitrary Python game classes.  It is
        # used only when validation and a real branch benchmark show it is faster
        # than the current clone backend; otherwise the original clone-BFS below
        # remains the source of truth.
        if self._configure_snapshot_bfs(game, actions):
            sol, explored_s, unique_s, elapsed_s = self._solve_bfs_snapshot(
                game, f0, level_idx, actions, effective_timeout, max_states,
                hidden_fields=None, cap=self.max_bfs_queue, label=f"L{level_idx}")
            if sol:
                return sol
            # If snapshot states became invalid early, fall back to clone-BFS with
            # the remaining budget.  If it merely timed out, do not repeat the same
            # full search with a slower backend.
            if self._snapshot_bfs_enabled:
                if explored_s < 20 and elapsed_s > 10.0:
                    logger.info(f"BFS L{level_idx}: early exit after snapshot pass — handing off to CNN")
                return None
            effective_timeout = max(1.0, effective_timeout - elapsed_s)

        # ==========================================
        # Phase 2: BFS — Memory Optimised Replay
        # ==========================================
        hidden_fields = None
        visited = set()
        queue = self._queue_cls()
        h0 = state_hash(game, f0, None)
        visited.add(h0)

        # Custom implicit graph: the frontier contains only integer node IDs.
        # Full game objects live in graph.states only while waiting to expand;
        # once popped, the state slot is cleared and only compact parent/action
        # metadata remains for eventual solution reconstruction.
        graph = self._graph_cls(clone_game(game))
        queue.append(0)

        t0 = time.time()
        explored = 0

        # CNN-guided beam search: expand all actions at shallow depths (full
        # BFS), then switch to top-K actions when state budget runs low.  This
        # lets BFS reach much deeper levels (depth ~13+) within the same time
        # budget, critical for Sokoban-style puzzles with long solutions.
        beam_transition = 800  # after this many explored states, restrict to top-K
        beam_K = 2

        while queue and explored < max_states and (time.time() - t0) < effective_timeout:
            node_idx = queue.popleft()
            g = graph.take_state(node_idx)
            if g is None:
                continue
            depth = graph.get_depth(node_idx)
            last_act = graph.get_last_action(node_idx) or None
            # Full expansion near the root; beam search deeper
            use_beam = explored > beam_transition
            candidates = actions[:beam_K] if use_beam else actions

            for act_id, data in candidates:
                # Partial-order reduction: immediate inverse directions cannot
                # improve a shortest path when the prior state is already visited.
                if last_act is not None and self._opposite_actions.get(last_act) == act_id:
                    continue
                child = clone_game(g)
                try:
                    r = child.perform_action(make_action(act_id, data), raw=True)
                except Exception:
                    continue
                explored += 1

                # A winning edge does not need a frame conversion or a visited-set
                # lookup.  This is especially useful at the depth limit.
                if is_complete(child, r, level_idx):
                    child_idx = graph.add_child(node_idx, act_id, data, None)
                    new_hist = self._reconstruct_solution(graph, child_idx)
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist

                f = last_frame(r)
                is_new_state = (f is not None and depth < self.max_bfs_depth)
                if is_new_state:
                    h = state_hash(child, f, hidden_fields)
                    is_new_state = h not in visited
                    if is_new_state:
                        visited.add(h)

                if is_new_state:
                    g2 = clone_game(child)
                    child_idx = graph.add_child(node_idx, act_id, data, g2)
                    queue.append(child_idx)
                    if len(queue) > self.max_bfs_queue * 2:
                        self._trim_frontier_if_needed(queue, graph, self.max_bfs_queue, f"L{level_idx}")

                # child is discarded by GC after this iteration

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")
        self._last_effective_actions = list(actions)

        # Smart early exit — game may be too expensive to BFS
        if explored < 20 and elapsed_first > 10.0:
            logger.info(f"BFS L{level_idx}: early exit (only {explored} explored in {elapsed_first:.1f}s) — handing off to CNN")
            return None

        # If too few unique states were produced despite enough exploration, hidden
        # state may be collapsing the visible-frame hash.  Guard this retry tightly:
        # without the guard, a failed hidden retry can double level time.
        unique_ratio = len(visited) / max(1, explored)
        time_left = max(0.0, effective_timeout - elapsed_first)
        should_hidden_retry = (
            explored >= self.hidden_retry_min_explored
            and len(visited) < 80
            and unique_ratio <= self.hidden_retry_unique_ratio
            and elapsed_first < effective_timeout * 0.75
            and time_left >= 6.0
        )
        if should_hidden_retry:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields} (unique_ratio={unique_ratio:.3f})")

                # FIX 3: Use exactly 2 RESET calls (not 3) to match the first pass baseline
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r0_2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = last_frame(r0_2)
                if f0_2 is None:
                    return None
                h0_2 = state_hash(game2, f0_2, hidden_fields)

                visited2 = set()
                visited2.add(h0_2)
                queue2 = self._queue_cls()
                graph2 = self._graph_cls(clone_game(game2))
                queue2.append(0)

                t0_2 = time.time()
                explored2 = 0
                # Keep retry bounded.  It is a fallback, not a second full BFS.
                remaining = min(self.hidden_retry_time_cap, max(0.0, effective_timeout - elapsed_first))

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    node_idx = queue2.popleft()
                    g = graph2.take_state(node_idx)
                    if g is None:
                        continue
                    depth = graph2.get_depth(node_idx)
                    last_act = graph2.get_last_action(node_idx) or None
                    use_beam2 = explored2 > beam_transition
                    candidates2 = actions[:beam_K] if use_beam2 else actions

                    for act_id, data in candidates2:
                        if last_act is not None and self._opposite_actions.get(last_act) == act_id:
                            continue
                        child = clone_game(g)
                        try:
                            r = child.perform_action(make_action(act_id, data), raw=True)
                        except Exception:
                            continue
                        explored2 += 1

                        if is_complete(child, r, level_idx):
                            child_idx = graph2.add_child(node_idx, act_id, data, None)
                            new_hist = self._reconstruct_solution(graph2, child_idx)
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in {len(new_hist)} actions ({explored2} explored)")
                            self.solutions[level_idx] = new_hist
                            return new_hist

                        f = last_frame(r)
                        is_new_state = (f is not None and depth < self.max_bfs_depth)
                        if is_new_state:
                            h = state_hash(child, f, hidden_fields)
                            is_new_state = h not in visited2
                            if is_new_state:
                                visited2.add(h)

                        if is_new_state:
                            g2 = clone_game(child)
                            child_idx = graph2.add_child(node_idx, act_id, data, g2)
                            queue2.append(child_idx)
                            if len(queue2) > self.max_bfs_queue_retry * 2:
                                self._trim_frontier_if_needed(queue2, graph2, self.max_bfs_queue_retry, f"L{level_idx} hidden-retry")

                logger.info(f"BFS L{level_idx}: hidden retry also failed ({explored2} explored, {len(visited2)} unique)")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly
            g = self._clone_game(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            # Try object-relative transfer
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            f0 = self._last_frame(r_prev)
            if f0 is None:
                return None
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))

            if not matched:
                return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = self._clone_game(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None

def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]

    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break

    if not src:
        import re
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"**/game_sources/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break

    return src, cls_name


# ==================== CNN FALLBACK ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5

    def encode_memory(s, mem_diffs, mem_actions, mem_rewards):
        """Encode static action-effect memory once until a new effect arrives."""
        B,M=mem_actions.shape
        if M==0:
            empty=mem_diffs.new_zeros((B,0,s.mem_dim))
            return empty, mem_diffs.new_zeros((B,0,s.mem_dim+1+5))
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).to(dtype=keys.dtype)
        vals=torch.cat([keys,mem_rewards.to(dtype=keys.dtype).unsqueeze(-1),act_oh],dim=-1)
        return keys, vals

    def forward(s, cnn_feat, mem_diffs=None, mem_actions=None, mem_rewards=None, encoded_memory=None):
        if encoded_memory is None:
            if mem_diffs is None or mem_actions is None or mem_rewards is None:
                return cnn_feat.new_zeros((cnn_feat.size(0),5))
            keys,vals=s.encode_memory(mem_diffs,mem_actions,mem_rewards)
        else:
            keys,vals=encoded_memory
        if keys.size(1)==0:
            return cnn_feat.new_zeros((cnn_feat.size(0),5))
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1);s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1);s.c4=nn.Conv2d(128,256,3,padding=1)
        s.skip1=nn.Conv2d(32,64,1);s.skip2=nn.Conv2d(64,256,1)
        s.attn=CBAM(256);s.ar=nn.Conv2d(256,64,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256);s.ah=nn.Linear(256,5);s.dr=nn.Dropout(0.15)
        s.vf=nn.Linear(64*16*16,256);s.vh=nn.Linear(256,1)
        s.ln=nn.LayerNorm(256)
        s.cc1=nn.Conv2d(256,128,3,padding=1);s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1)
        s.cc_d1=nn.Conv2d(32,16,3,padding=2,dilation=2)
        s.cc_d2=nn.Conv2d(32,16,3,padding=4,dilation=4)
        s.cc_fuse=nn.Conv2d(64,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def _features(s, x):
        x=F.relu(s.c1(x));h=F.relu(s.c2(x));x=F.relu(s.c3(h))
        f=F.relu(s.c4(x))+s.skip2(h);return s.attn(f)

    def _action_logits(s, f, mem_diffs=None, mem_actions=None, mem_rewards=None, mem_encoded=None):
        af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        h=F.relu(s.af(af));h=s.dr(h);h=s.ln(h)
        adv=s.ah(h);val=s.vh(s.dr(F.relu(s.vf(af))))
        if mem_encoded is not None or (mem_diffs is not None and mem_actions is not None):
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            adv=adv+s.aea(gf,mem_diffs,mem_actions,mem_rewards,encoded_memory=mem_encoded)
        return val+(adv-adv.mean(1,keepdim=True))

    def forward_actions(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None, mem_encoded=None):
        # Fast path when ACTION6/click is unavailable: skip the 4096-cell click head.
        f=s._features(x)
        return s._action_logits(f,mem_diffs,mem_actions,mem_rewards,mem_encoded)

    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None, mem_encoded=None):
        f=s._features(x)
        al=s._action_logits(f,mem_diffs,mem_actions,mem_rewards,mem_encoded)
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        d1=F.relu(s.cc_d1(cf));d2=F.relu(s.cc_d2(cf))
        cl=s.cc_fuse(torch.cat([cf,d1,d2],1)).reshape(f.size(0),-1)
        return torch.cat([al,cl],1)

def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        s.total_time_budget = 6 * 3600 - 180     # 6 ঘণ্টা, শেষ 300 সেকেন্ড সেফটি মার্জিন
        s.estimated_total_levels = 50            # গড়ে কয়টি লেভেল আসতে পারে (পরিবেশ থেকে পাওয়া গেলে ভালো)
        s.current_level_start_time = None        # এই লেভেলে আসার সময়
        s.current_level_budget = 0
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        # CNN work is the only substantial CUDA workload in this agent.  Use
        # Tensor Cores for convolution/linear ops when CUDA is present, without
        # relying on torch.compile/Triton (which is often unavailable on Windows).
        s._amp_enabled = (s.device.type == 'cuda')
        s._grad_scaler = None
        if s._amp_enabled:
            try:
                torch.backends.cudnn.benchmark = True  # inputs are always 64x64
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision('high')
                except Exception:
                    pass
                try:
                    s._grad_scaler = torch.amp.GradScaler('cuda', enabled=True)
                except (AttributeError, TypeError):
                    # Compatibility with older PyTorch builds.
                    s._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception as e:
                logger.info('CUDA AMP/TF32 setup unavailable: %s', e)
                s._amp_enabled = False
                s._grad_scaler = None
        s.G=64; s.IN=26
        # Reusable CPU-side constants to avoid reallocating them every action/train step.
        s._reward_mask=np.ones((64,64),dtype=bool); s._reward_mask[:2]=False; s._reward_mask[62:]=False
        _rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        _cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        s._pos_aug=torch.from_numpy(np.stack([_rp,_cp]))
        s._pos_aug_device=None
        s._wm_dev=None
        s._wm_log_dev=None
        s._wm_cache_key=None
        # Replay stores compact uint8 frames plus parallel scalar arrays.  The
        # prior dict/int64 representation could exceed 1.6 GB at capacity.
        s.buf=[]; s.buf_actions=array('H'); s.buf_rewards=array('f'); s.buf_next_frames=[]; s.buf_priorities=[]; s.buf_keys=[]; s.buf_max=50000; s.buf_pos=0; s.buf_h=set()
        # Cache AEM tensors and their expensive diff-encoder output separately.
        s._aem_cache_sig=None; s._aem_cache=(None,None,None); s._aem_max_active=128
        s._aem_encoded_cache_sig=None; s._aem_encoded_cache=None; s._model_revision=0
        s.net=None; s.opt=None; s.scheduler=None
        s._weights_loaded=False
        s.bsz=128 if s.device.type=='cuda' else 64; s.tfreq=5
        s._last_train_action=-10**9; s._train_min_gap=1; s._max_train_burst=5
        s._clear_recent_action_state()
        s._semantic_target_coord=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.02; s._eps_decay=0.9997; s._eps_steps=0
        s._prev_objs=None; s._obj_moved=0
        # FIX 1: Initialize _visited_hashes so _reward() deduplication works correctly
        s._visited_hashes = set()
        # Count-based intrinsic exploration bonus tracking
        s._state_visit_counts = {}
        # _tensor() static frame cache: avoids re-encoding 21 channels when frame unchanged
        s._tensor_last_frame_hash = None
        s._tensor_cached_static = None
        # _replay_batch_tensor frame feature cache: avoids recomputing one-hot/edge/rarity
        s._frame_feature_cache = {}
        s._frame_feature_cache_max = 500
        # Semantic analysis caches: choose_action may query the same frame
        # several times through target ranking, click priors, and rescoring.
        s._semantic_components_cache_key=None
        s._semantic_components_cache_value=None
        s._semantic_target_candidates_cache_key=None
        s._semantic_target_candidates_cache_value=None
        # TD-learning hyperparameters
        s.gamma = 0.9; s.tau = 0.005; s._target_net = None
        s._mdqn_alpha = 0.9; s._mdqn_tau = 0.03
        # PER hyperparameters
        s._per_alpha = 0.6; s._per_beta = 0.4; s._per_beta_step = 0.001
        s._target_update_counter = 0; s._target_hard_update_interval = 500
        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        s._semantic_detector = _detect_sprites_helper

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return _frame_view(fd.frame[-1], np.uint8)
    def _fast_frame_hash(s, frame): return _frame_crc(frame)

    def _fresh_action(s, act_id, data=None):
        action = GameAction.from_id(int(act_id))
        if data:
            action.set_data(data)
        return action

    def _click_action_data(s, coord):
        """Build ACTION6 payload data from a `(y, x)` grid coordinate."""
        y, x = coord
        return {"x": int(x), "y": int(y)}

    def _click_action_index(s, coord):
        """Map a `(y, x)` grid coordinate to the internal click action index."""
        y, x = coord
        return 5 + int(y) * s.G + int(x)

    def _click_coord_from_action_index(s, action_idx):
        """Map an internal click action index back to a `(y, x)` grid coordinate."""
        click_idx=int(action_idx) - 5
        return (int(click_idx // s.G), int(click_idx % s.G))

    def _click_data_from_policy_index(s, idx):
        """Map a flat policy-logit click index to ACTION6 payload data."""
        return s._click_action_data(s._click_coord_from_action_index(idx))

    def _click_coord_distance(s, coord_a, coord_b):
        """Return Manhattan distance between two `(y, x)` click coordinates."""
        return abs(int(coord_a[0]) - int(coord_b[0])) + abs(int(coord_a[1]) - int(coord_b[1]))

    def _click_action(s, coord):
        """Create an ACTION6 click action from a `(y, x)` grid coordinate."""
        return s._fresh_action(6, s._click_action_data(coord))

    def _action_id(s, action):
        """Return a numeric action id from either enum-like or plain action values."""
        return action.value if hasattr(action, 'value') else int(action)

    def _available_action_ids(s, avail):
        """Normalize a legal-action collection into plain integer ids."""
        return [s._action_id(action) for action in (avail or [])]

    def _preferred_click_coord(s):
        """Return the tracked semantic click target as an integer `(y, x)` tuple."""
        if s._semantic_target_coord is None:
            return None
        return (int(s._semantic_target_coord[0]), int(s._semantic_target_coord[1]))

    def _nearest_coord_within(s, coords, preferred_coord, max_distance):
        """Return the nearest `(y, x)` coord within a Manhattan distance threshold."""
        nearest_coord=None
        nearest_distance=None
        for coord in coords:
            dist=(abs(int(coord[0]) - int(preferred_coord[0])) +
                  abs(int(coord[1]) - int(preferred_coord[1])))
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance=dist
                nearest_coord=(int(coord[0]), int(coord[1]))
        if nearest_coord is not None and nearest_distance is not None and nearest_distance <= max_distance:
            return nearest_coord
        return None

    def _prepend_nearest_preferred_coord(s, frame, candidates, coords, preferred_coord, seen, limit,
                                         blocked_click_coord=None):
        """Seed `coords` with the nearest preferred click candidate when it is nearby."""
        if preferred_coord is None:
            return False
        nearest_coord=s._nearest_coord_within(
            (coord for coord in candidates if not s._blocked_click_matches_coord(
                frame,
                coord,
                blocked_click_coord=blocked_click_coord,
            )),
            preferred_coord,
            2,
        )
        if nearest_coord is None:
            return False
        seen.add(nearest_coord)
        coords.insert(0, nearest_coord)
        return len(coords) >= limit

    def _append_unblocked_coords(s, frame, candidates, coords, seen, limit, blocked_click_coord=None):
        """Append unseen, unblocked coords until `limit` is reached."""
        for coord in candidates:
            if (coord in seen or
                    s._blocked_click_matches_coord(
                        frame,
                        coord,
                        blocked_click_coord=blocked_click_coord,
                    )):
                continue
            seen.add(coord)
            coords.append(coord)
            if len(coords) >= limit:
                return True
        return False

    def _append_candidate_index(s, candidate_indices, candidate_seen, idx, scored=None, avail_mask=None):
        """Append a candidate action index once when it is legal and unseen."""
        idx=int(idx)
        if idx in candidate_seen:
            return False
        if scored is not None and idx >= len(scored):
            return False
        if avail_mask is not None and not torch.isfinite(avail_mask[idx]):
            return False
        candidate_seen.add(idx)
        candidate_indices.append(idx)
        return True

    def _decode_policy_action_index(s, idx):
        """Decode a flat policy-logit index into `(action_idx, click_coord)`."""
        idx=int(idx)
        if idx < 5:
            return idx, None
        return 5, s._click_coord_from_action_index(idx)

    def _bfs_priority_bonus(s, act_id, data=None):
        """Return the BFS-derived tie-break bonus for an action candidate."""
        if s._bfs is None:
            return 0.0
        bfs_key=s._bfs._action_key(act_id, data)
        return s._bfs._action_priority.get(bfs_key, 0) * 0.25

    def _preferred_click_bonus(s, click_coord, preferred_click_coord):
        """Return the continuity bonus for clicks near the preferred semantic target."""
        if preferred_click_coord is None:
            return 0.0
        click_pref_dist=s._click_coord_distance(click_coord, preferred_click_coord)
        if click_pref_dist == 0:
            return 0.08
        if click_pref_dist <= 2:
            return 0.04
        return 0.0

    def _preferred_direction_choice(s, preferred_dir, blocked, legal_action_ids):
        """Return the preferred direction index when it is still legal and unblocked."""
        if preferred_dir is None:
            return None
        preferred_action_id=preferred_dir + 1
        if preferred_dir == blocked or preferred_action_id not in legal_action_ids:
            return None
        return preferred_dir, None

    def _preferred_click_target_choice(s, targets, preferred_coord, step):
        """Choose a click target by preferred continuity, then by step offset."""
        if preferred_coord is not None:
            if preferred_coord in targets:
                return preferred_coord
            nearest_target=s._nearest_coord_within(targets, preferred_coord, 2)
            if nearest_target is not None:
                return nearest_target
        pidx=step-4
        if 0 <= pidx < len(targets):
            return targets[pidx]
        return None

    def _semantic_click_bonus_map(s, frame, limit, click_scale, click_targets=None):
        """Return ranked semantic click bonuses keyed by `(y, x)` coordinate."""
        bonuses={}
        if click_targets is None:
            click_targets=s._semantic_click_targets_compat(frame, limit=limit)
        for rank,(ty,tx) in enumerate(click_targets):
            bonuses[(int(ty), int(tx))]=max(0.0, 0.8 - 0.1 * rank) * click_scale
        return bonuses

    def _count_action(s):
        s.action_counter += 1

    def _clear_recent_action_state(s):
        """Drop the previous-frame/action cache used for reward shaping."""
        s.pt = None
        s.pai = None
        s.pr = None
        s.ph = None

    def _remember_recent_action(s, tensor, raw, frame_hash, action_idx):
        """Store the current observation and chosen action for the next step."""
        s.pt = tensor
        s.pai = action_idx
        s.pr = raw.copy()
        s.ph = frame_hash
        s.la += 1

    def _finalize_action(s, action, reasoning, *, tensor=None, raw=None, frame_hash=None,
                         action_idx=None, remember_recent=False, clear_recent=False):
        """Attach reasoning and finish an action return with consistent bookkeeping."""
        action.reasoning = reasoning
        if clear_recent:
            s._clear_recent_action_state()
        elif remember_recent:
            s._remember_recent_action(tensor, raw, frame_hash, action_idx)
        s._count_action()
        return action

    def _finalize_control_action(s, act_id, reasoning, *, tensor=None, raw=None, frame_hash=None,
                                 remember_recent=False, clear_recent=False):
        """Finalize RESET/UNDO/NO-ACTION style branches with shared semantic cleanup."""
        s._semantic_target_coord=None
        return s._finalize_action(
            s._fresh_action(act_id),
            reasoning,
            tensor=tensor,
            raw=raw,
            frame_hash=frame_hash,
            action_idx=None,
            remember_recent=remember_recent,
            clear_recent=clear_recent,
        )

    def _amp_context(s):
        """Return a CUDA autocast context without requiring Triton/compile."""
        if s._amp_enabled:
            try:
                return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
            except AttributeError:
                return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
        return nullcontext()

    def _make_optimizer(s):
        """AdamW with weight decay for better generalization."""
        if s.device.type == 'cuda':
            try:
                return optim.AdamW(s.net.parameters(), lr=0.0003, weight_decay=1e-5, fused=True)
            except (TypeError, RuntimeError):
                try:
                    return optim.AdamW(s.net.parameters(), lr=0.0003, weight_decay=1e-5, foreach=True)
                except TypeError:
                    pass
        return optim.AdamW(s.net.parameters(), lr=0.0003, weight_decay=1e-5)

    def _make_scheduler(s):
        return optim.lr_scheduler.CosineAnnealingLR(s.opt, T_max=10000, eta_min=3e-5)

    def _boost_recent_replay_rewards(s, reward, current_idx):
        if reward < 1.0 or not s.buf_rewards:
            return
        boost = min(reward * 0.15, 0.5)
        limit = min(5, len(s.buf_rewards) - 1)
        for j in range(1, limit + 1):
            idx = (current_idx - j) % len(s.buf_rewards)
            s.buf_rewards[idx] = float(s.buf_rewards[idx]) + boost * (0.85 ** (j - 1))
            if idx < len(s.buf_priorities):
                s.buf_priorities[idx] = abs(float(s.buf_rewards[idx])) + 0.01

    def _clear_replay(s, keep_frac=0.2):
        """Clear replay buffer, optionally retaining top-K highest-reward transitions
        for cross-level transfer of learned action-effect patterns.
        Small buffers (< keep_frac threshold) are preserved intact so expert
        demonstrations (BFS solutions, CLTI, etc.) persist across level changes."""
        if keep_frac > 0 and len(s.buf) <= s.bsz:
            return
        if keep_frac > 0 and len(s.buf) > s.bsz:
            n_keep = max(s.bsz, int(len(s.buf) * keep_frac))
            rewards = np.array(s.buf_rewards, dtype=np.float32)
            if len(rewards) > n_keep:
                top_idx = np.argsort(rewards)[-n_keep:]
                s.buf = [s.buf[i] for i in top_idx]
                s.buf_actions = array('H', [s.buf_actions[i] for i in top_idx])
                s.buf_rewards = array('f', [s.buf_rewards[i] for i in top_idx])
                if s.buf_next_frames:
                    s.buf_next_frames = [s.buf_next_frames[i] for i in top_idx]
                if s.buf_priorities:
                    s.buf_priorities = [s.buf_priorities[i] for i in top_idx]
                if s.buf_keys:
                    s.buf_keys = [s.buf_keys[i] for i in top_idx]
            else:
                return  # buffer small enough — keep all entries intact
            # Always reset dedup hash and position when entries are pruned
            s.buf_h = {key for key in s.buf_keys if key is not None}; s.buf_pos = 0
            return
        # Full clear (keep_frac <= 0 or buffer empty)
        s.buf.clear(); s.buf_actions=array('H'); s.buf_rewards=array('f')
        s.buf_next_frames.clear(); s.buf_priorities.clear(); s.buf_keys.clear(); s.buf_h.clear(); s.buf_pos=0

    def _add_replay(s, frame, action_idx, reward, next_frame=None, dedup_key=None):
        """Append a compact transition without per-entry dict or int64 overhead."""
        snapshot=np.ascontiguousarray(frame, dtype=np.uint8).copy()
        next_snapshot=np.ascontiguousarray(next_frame, dtype=np.uint8).copy() if next_frame is not None else None
        action_idx=max(0,min(65535,int(action_idx)))
        reward=float(reward)
        priority=abs(reward)+0.01
        if len(s.buf) < s.buf_max:
            s.buf.append(snapshot)
            s.buf_actions.append(action_idx)
            s.buf_rewards.append(reward)
            s.buf_next_frames.append(next_snapshot)
            s.buf_priorities.append(priority)
            s.buf_keys.append(dedup_key)
            if dedup_key is not None:
                s.buf_h.add(dedup_key)
            s._boost_recent_replay_rewards(reward, len(s.buf_rewards) - 1)
        else:
            i=s.buf_pos
            old_key = s.buf_keys[i] if i < len(s.buf_keys) else None
            if old_key is not None and sum(1 for key in s.buf_keys if key == old_key) <= 1:
                s.buf_h.discard(old_key)
            s.buf[i]=snapshot
            s.buf_actions[i]=action_idx
            s.buf_rewards[i]=reward
            s.buf_next_frames[i]=next_snapshot
            s.buf_priorities[i]=priority
            s.buf_keys[i]=dedup_key
            if dedup_key is not None:
                s.buf_h.add(dedup_key)
            s._boost_recent_replay_rewards(reward, i)
            s.buf_pos=(i+1)%s.buf_max

    def _init_bfs(s):
        """Initialize BFS solver on first call."""
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=2, bfs_timeout=60)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")


    def _adaptive_bfs_timeout(s, level_idx):
        # Keep BFS useful without letting early hard levels consume the full run.
        # Levels 0-2 get extra budget because they are most likely solvable and
        # their solutions transfer to later levels (CLTI).
        elapsed = time.time() - s.start_time
        remaining = max(60.0, s.total_time_budget - elapsed)
        remaining_levels = max(1, s.estimated_total_levels - int(level_idx))
        fair_share = remaining / remaining_levels
        lvl = int(level_idx)
        if lvl <= 1:
            cap = 60.0
        elif lvl <= 3:
            cap = 40.0
        else:
            cap = 25.0
        return max(10.0, min(cap, fair_share))

    def _try_bfs_solve(s, level_idx, lf=None):
        """Try to solve current level with BFS, using previous solution for transfer."""
        if s._bfs is None:
            return None
        # Try cached solution for this level first (avoids re-solving on resets)
        cached = s._bfs.solutions.get(level_idx)
        if cached:
            try:
                g_cache = s._bfs.game_cls()
                g_cache.set_level(level_idx)
                g_cache.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g_cache.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for i, (act_id, data) in enumerate(cached):
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g_cache.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g_cache._current_level_index > level_idx:
                        sol = cached[:i+1]
                        s._bfs_solution = sol
                        s._bfs_step = 0
                        logger.info(f"BFS L{level_idx}: using cached solution ({i+1} actions)")
                        return sol
            except Exception:
                pass
        # Compute CNN frame tensor for guided BFS action ordering
        net = s.net if s.net is not None else None
        frame_tensor = None
        if net is not None and lf is not None:
            try:
                frame_tensor = s._tensor(lf)
            except Exception:
                pass
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol, timeout=s._adaptive_bfs_timeout(level_idx),
                                  net=net, frame_tensor=frame_tensor)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        return None

    def _encode_static_frame_cpu(s, frame, update_bg=False):
        """Create the 21 history-independent channels with vectorised NumPy work."""
        frame=np.ascontiguousarray(frame, dtype=np.uint8)
        cnt=np.bincount(frame.ravel(),minlength=16).astype(np.float32, copy=False)
        bg=int(cnt.argmax()); mx=max(float(cnt[bg]),1.0)
        if update_bg:
            s._bg=bg

        out=torch.zeros(21,64,64,dtype=torch.float32)
        idx=torch.from_numpy(frame).to(torch.long)
        out[:16].scatter_(0,idx.unsqueeze(0),1.0)
        out[16]=torch.from_numpy((frame==bg).astype(np.float32,copy=False))
        rarity=(1.0-cnt/mx).astype(np.float32,copy=False)
        out[17]=torch.from_numpy(rarity[frame])

        edge=np.zeros((64,64),dtype=bool)
        edge[1:,:] |= frame[1:,:] != frame[:-1,:]
        edge[:-1,:] |= frame[:-1,:] != frame[1:,:]
        edge[:,1:] |= frame[:,1:] != frame[:,:-1]
        edge[:,:-1] |= frame[:,:-1] != frame[:,1:]
        out[18]=torch.from_numpy(edge.astype(np.float32,copy=False))
        out[19:21]=s._pos_aug
        return out

    def _encode_frame_tensor(s, frame):
        """Encode one raw frame into the 26-channel network input.

        Replay training stores single-step snapshots only, so keep the dynamic
        history channels zeroed here as well.  This avoids train/inference drift
        and keeps encoding side-effect free for BFS probes and retries.
        """
        fh=s._fast_frame_hash(frame)
        if fh == s._tensor_last_frame_hash and s._tensor_cached_static is not None:
            static = s._tensor_cached_static
        else:
            static = s._encode_static_frame_cpu(frame, update_bg=True)
            s._tensor_last_frame_hash = fh
            s._tensor_cached_static = static

        out=torch.zeros(26,64,64,dtype=torch.float32)
        out[:21]=static
        return out.to(s.device,non_blocking=True)

    def _tensor(s, fd):
        frame=s._raw(fd)
        return s._encode_frame_tensor(frame)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                mask.view(64,64)[:, :c+1] = 0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                mask.view(64,64)[:r+1, :] = 0.05
                return mask
        return mask

    def _template_log_bias(s):
        """Return the cached flattened log click-prior tensor on the active device."""
        if s._wm is None:
            return None
        cache_key=(id(s._wm), s.device.type, getattr(s.device, 'index', None))
        if s._wm_log_dev is not None and s._wm_cache_key == cache_key:
            return s._wm_log_dev
        if isinstance(s._wm, torch.Tensor):
            wm_tensor=s._wm
        else:
            wm_tensor=torch.as_tensor(s._wm, dtype=torch.float32)
        if wm_tensor.device != s.device:
            wm_tensor=wm_tensor.to(s.device)
        else:
            wm_tensor=wm_tensor.to(dtype=torch.float32)
        s._wm_dev=wm_tensor
        s._wm_log_dev=torch.log(wm_tensor.reshape(-1).clamp(min=0.01))
        s._wm_cache_key=cache_key
        return s._wm_log_dev

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h, changed=None, curr_objs=None, move_bonus=0.0, moved=0):
        # FIX 1: Use s._visited_hashes (now properly initialized) for deduplication.
        r=0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2  # small reward for revisiting — not zero, avoids cliff in sparse games
        else:
            r -= 0.1
        if changed is None:
            changed=bool(np.any((prev_raw!=curr_raw)&s._reward_mask))
        if changed:
            r+=0.5
            prev_goal_dist=s._semantic_goal_distance(prev_raw)
            curr_goal_dist=s._semantic_goal_distance(curr_raw)
            if prev_goal_dist is not None and curr_goal_dist is not None:
                delta=prev_goal_dist-curr_goal_dist
                if delta > 0.5:
                    r += min(0.4, 0.08 * delta)
                elif delta < -0.5:
                    r -= min(0.2, 0.04 * (-delta))
        # Fast path: choose_action can pass curr_objs + movement bonus from the
        # fused Cython state update.  Fallback preserves the previous behavior.
        if curr_objs is None:
            curr_objs=fast_objects(curr_raw,s._bg)
            if s._prev_objs and curr_objs:
                moved=0
                for co in curr_objs:
                    for po in s._prev_objs:
                        if co[0]==po[0]:
                            dist=abs(co[1]-po[1])+abs(co[2]-po[2])
                            if 2<dist<20:moved+=1;break
                move_bonus=0.3*min(moved,3) if moved>0 else 0.0
        if moved>0:
            r+=float(move_bonus);s._obj_moved=int(moved)
        s._prev_objs=curr_objs
        # Count-based intrinsic exploration bonus: rewards novel states
        count = s._state_visit_counts.get(curr_h, 0)
        s._state_visit_counts[curr_h] = count + 1
        r += 0.3 / (count ** 0.5 + 1)
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        has_click_logits = logits.numel() >= 4101
        avail_ids=s._available_action_ids(avail)
        a6=has_click_logits and ((not avail_ids) or 6 in avail_ids)
        total_len=4101 if a6 else 5
        allp=torch.zeros(total_len, dtype=logits.dtype, device=logits.device)
        eligible=torch.zeros(total_len, dtype=torch.bool, device=logits.device)
        dir_logits=logits[:5]
        if avail_ids:
            for aid in avail_ids:
                if 1 <= aid <= 5:
                    idx=aid - 1
                    logit=dir_logits[idx]
                    allp[idx]=torch.sigmoid(logit / temp)
                    eligible[idx]=torch.isfinite(logit)
        else:
            allp[:5]=torch.sigmoid(dir_logits / temp)
            eligible[:5]=torch.isfinite(dir_logits)
        if a6:
            click_logits=logits[5:5+4096]
            template_log_bias=s._template_log_bias()
            if template_log_bias is not None:
                click_logits=click_logits + template_log_bias
            allp[5:]=torch.sigmoid(click_logits / temp) / (s.G * s.G)
            eligible[5:]=torch.isfinite(click_logits)
        sm=allp.sum()
        if sm<1e-8:
            if torch.any(eligible):
                allp.zero_()
                allp[eligible]=1.0
                allp=allp / allp.sum()
            else:
                allp.fill_(1.0 / len(allp))
        else:
            allp=allp / sm
        idx=int(torch.multinomial(allp, 1).item())
        return s._decode_policy_action_index(idx)

    def _legal_action_mask(s, logits, avail):
        """Mask logits down to currently legal modeled actions."""
        mask=torch.full((len(logits),),-float('inf'),device=logits.device)
        if avail is None or len(avail)==0:
            mask.zero_()
            return mask
        click_avail=False
        for aid in s._available_action_ids(avail):
            if 1<=aid<=5:
                mask[aid-1]=0.0
            elif aid==6 and len(logits)>5:
                click_avail=True
        if click_avail and len(logits)>5:
            mask[5:]=0.0
        return mask

    def _semantic_click_targets(s, frame, limit=8, blocked_click_coord=None):
        """Rank likely interactive click targets from connected components."""
        frame=np.ascontiguousarray(frame, dtype=np.uint8)
        preferred=s._semantic_target_coord
        preferred_coord=s._preferred_click_coord()
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame)
        scored=s._semantic_target_candidates(frame, blocked_click_coord=blocked_click_coord)
        if scored:
            coords=[]
            seen=set()
            scored_coords=[
                (int(round(item['target_y'])), int(round(item['target_x'])))
                for item in scored
            ]
            if s._prepend_nearest_preferred_coord(
                    frame,
                    scored_coords,
                    coords,
                    preferred_coord,
                    seen,
                    limit,
                    blocked_click_coord=blocked_click_coord):
                return coords
            if s._append_unblocked_coords(
                    frame,
                    scored_coords,
                    coords,
                    seen,
                    limit,
                    blocked_click_coord=blocked_click_coord):
                return coords
            if coords:
                return coords
        detector=getattr(s, '_semantic_detector', None)
        if detector is not None:
            try:
                semantic=detector(frame.tolist())
                comps=(semantic or {}).get('components_per_value') or {}
                color_priority={14:0,6:1,9:2,11:3,5:4,7:5,13:6,15:7}
                scored=[]
                for key, items in comps.items():
                    try:
                        color=int(key)
                    except Exception:
                        continue
                    if color not in color_priority:
                        continue
                    for comp in items or []:
                        center=comp.get('center')
                        if not center or len(center) != 2:
                            continue
                        cy=int(round(float(center[0])))
                        cx=int(round(float(center[1])))
                        area=int(comp.get('cell_count', 0))
                        if area <= 0 or area > 512:
                            continue
                        scored.append({
                            'score': float(color_priority[color]),
                            'priority': color_priority[color],
                            'distance': float(abs(cy-32)+abs(cx-32)),
                            'continuity_distance': float(abs(cy-preferred[0])+abs(cx-preferred[1])) if preferred is not None else float(abs(cy-32)+abs(cx-32)),
                            'target_y': float(cy),
                            'target_x': float(cx),
                            'area': area,
                        })
                scored.sort(key=lambda item: (item['score'], item['continuity_distance'], -item['area'], item['distance']))
                if scored:
                    coords=[]
                    seen=set()
                    scored_coords=[
                        (int(round(item['target_y'])), int(round(item['target_x'])))
                        for item in scored
                    ]
                    if s._prepend_nearest_preferred_coord(
                            frame,
                            scored_coords,
                            coords,
                            preferred_coord,
                            seen,
                            limit,
                            blocked_click_coord=blocked_click_coord):
                        return coords
                    if s._append_unblocked_coords(
                            frame,
                            scored_coords,
                            coords,
                            seen,
                            limit,
                            blocked_click_coord=blocked_click_coord):
                        return coords
                    if coords:
                        return coords
            except Exception:
                pass
        h,w=frame.shape
        visited=np.zeros((h,w),dtype=bool)
        # Prefer clearly interactive colors over large structural regions.
        color_priority={14:0,6:1,9:2,11:3,5:4,7:5,13:6,15:7}
        targets=[]
        for y in range(h):
            for x in range(w):
                if visited[y,x]:
                    continue
                visited[y,x]=True
                color=int(frame[y,x])
                if color==s._bg or color in (1,2,3,4,8,10,12):
                    continue
                stack=[(y,x)]
                area=0
                min_y=max_y=y
                min_x=max_x=x
                sum_y=0
                sum_x=0
                while stack:
                    cy,cx=stack.pop()
                    area+=1
                    sum_y+=cy
                    sum_x+=cx
                    if cy<min_y:min_y=cy
                    if cy>max_y:max_y=cy
                    if cx<min_x:min_x=cx
                    if cx>max_x:max_x=cx
                    if cy>0 and not visited[cy-1,cx] and frame[cy-1,cx]==color:
                        visited[cy-1,cx]=True; stack.append((cy-1,cx))
                    if cy+1<h and not visited[cy+1,cx] and frame[cy+1,cx]==color:
                        visited[cy+1,cx]=True; stack.append((cy+1,cx))
                    if cx>0 and not visited[cy,cx-1] and frame[cy,cx-1]==color:
                        visited[cy,cx-1]=True; stack.append((cy,cx-1))
                    if cx+1<w and not visited[cy,cx+1] and frame[cy,cx+1]==color:
                        visited[cy,cx+1]=True; stack.append((cy,cx+1))
                if area <= 0 or area > 512:
                    continue
                cy=int(round(sum_y/area))
                cx=int(round(sum_x/area))
                cy=max(min_y,min(max_y,cy))
                cx=max(min_x,min(max_x,cx))
                targets.append((
                    color_priority.get(color, 50),
                    abs(cy-preferred[0])+abs(cx-preferred[1]) if preferred is not None else abs(cy-32)+abs(cx-32),
                    area,
                    abs(cy-32)+abs(cx-32),
                    (cy,cx),
                ))
        targets.sort(key=lambda item: (item[0], item[1], -item[2], item[3], item[4]))
        coords=[]
        seen=set()
        target_coords=[coord for _,_,_,_,coord in targets]
        if s._prepend_nearest_preferred_coord(
                frame,
                target_coords,
                coords,
                preferred_coord,
                seen,
                limit,
                blocked_click_coord=blocked_click_coord):
            return coords
        s._append_unblocked_coords(
            frame,
            target_coords,
            coords,
            seen,
            limit,
            blocked_click_coord=blocked_click_coord,
        )
        return coords

    def _semantic_click_targets_compat(s, frame, limit=8, blocked_click_coord=None):
        """Call `_semantic_click_targets` with kwarg fallback for test doubles."""
        try:
            return s._semantic_click_targets(
                frame,
                limit=limit,
                blocked_click_coord=blocked_click_coord,
            )
        except TypeError as exc:
            if 'blocked_click_coord' not in str(exc):
                raise
            return s._semantic_click_targets(frame, limit=limit)

    def _semantic_components(s, frame):
        """Return semantic components when the sprite detector is available."""
        frame_hash=s._fast_frame_hash(frame)
        detector=getattr(s, '_semantic_detector', None)
        cache_key=(frame_hash, id(detector))
        if s._semantic_components_cache_key == cache_key:
            return s._semantic_components_cache_value
        if detector is None:
            comps=s._raw_semantic_components(frame)
            s._semantic_components_cache_key=cache_key
            s._semantic_components_cache_value=comps
            return comps
        try:
            semantic=detector(np.ascontiguousarray(frame, dtype=np.uint8).tolist())
        except Exception:
            comps=s._raw_semantic_components(frame)
            s._semantic_components_cache_key=cache_key
            s._semantic_components_cache_value=comps
            return comps
        comps=(semantic or {}).get('components_per_value') or None
        if not comps:
            comps=s._raw_semantic_components(frame)
        s._semantic_components_cache_key=cache_key
        s._semantic_components_cache_value=comps
        return comps

    def _raw_semantic_components(s, frame):
        """Cheap fallback semantic components directly from the raw color grid."""
        arr=np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.ndim != 2:
            return None
        h,w=arr.shape
        if h <= 0 or w <= 0:
            return None
        visited=np.zeros((h,w), dtype=bool)
        components={}
        for y in range(h):
            for x in range(w):
                if visited[y,x]:
                    continue
                color=int(arr[y,x])
                visited[y,x]=True
                stack=[(y,x)]
                area=0
                sum_y=0
                sum_x=0
                while stack:
                    cy,cx=stack.pop()
                    area+=1
                    sum_y+=cy
                    sum_x+=cx
                    if cy>0 and not visited[cy-1,cx] and int(arr[cy-1,cx])==color:
                        visited[cy-1,cx]=True; stack.append((cy-1,cx))
                    if cy+1<h and not visited[cy+1,cx] and int(arr[cy+1,cx])==color:
                        visited[cy+1,cx]=True; stack.append((cy+1,cx))
                    if cx>0 and not visited[cy,cx-1] and int(arr[cy,cx-1])==color:
                        visited[cy,cx-1]=True; stack.append((cy,cx-1))
                    if cx+1<w and not visited[cy,cx+1] and int(arr[cy,cx+1])==color:
                        visited[cy,cx+1]=True; stack.append((cy,cx+1))
                components.setdefault(str(color), []).append({
                    'center': (float(sum_y/area), float(sum_x/area)),
                    'cell_count': int(area),
                })
        return components

    def _semantic_target_candidates(s, frame, blocked_click_coord=None):
        """Rank semantic targets using class priority plus player-relative distance."""
        frame_hash=s._fast_frame_hash(frame)
        preferred=s._semantic_target_coord
        recent_direction=s._recent_direction_action_index(frame)
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame)
        cache_key=(
            frame_hash,
            None if preferred is None else (int(preferred[0]), int(preferred[1])),
            None if blocked_click_coord is None else (int(blocked_click_coord[0]), int(blocked_click_coord[1])),
            recent_direction,
        )
        if s._semantic_target_candidates_cache_key == cache_key:
            return s._semantic_target_candidates_cache_value
        comps=s._semantic_components(frame)
        if not comps:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        players=(comps.get('4') or []) + (comps.get('12') or [])
        if not players:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        player=max(players, key=lambda comp: int(comp.get('cell_count', 0)))
        center=player.get('center')
        if not center or len(center) != 2:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        py=float(center[0]); px=float(center[1])
        target_specs=[]
        for color, priority in ((14,0), (6,1), (11,2), (5,3), (9,4), (7,5), (13,6), (15,7)):
            for comp in comps.get(str(color)) or []:
                tcenter=comp.get('center')
                if not tcenter or len(tcenter) != 2:
                    continue
                ty=float(tcenter[0]); tx=float(tcenter[1])
                if s._blocked_click_matches_coord(
                        frame,
                        (int(round(ty)), int(round(tx))),
                        blocked_click_coord=blocked_click_coord):
                    continue
                dist=abs(ty-py)+abs(tx-px)
                if dist < 1.0:
                    continue
                area=int(comp.get('cell_count', 0))
                if area <= 0 or area > 512:
                    continue
                score=float(priority) * 2.0 + float(dist) / 6.0
                continuity_bonus=0.0
                if preferred is not None:
                    continuity_dist=abs(float(preferred[0])-ty)+abs(float(preferred[1])-tx)
                    continuity_bonus=max(0.0, 0.6 - 0.1 * continuity_dist)
                    score -= continuity_bonus
                momentum_bonus=0.0
                counter_momentum_penalty=0.0
                if recent_direction == 0 and ty < py:
                    momentum_bonus=0.12
                elif recent_direction == 0 and ty > py:
                    counter_momentum_penalty=0.18
                elif recent_direction == 1 and ty > py:
                    momentum_bonus=0.12
                elif recent_direction == 1 and ty < py:
                    counter_momentum_penalty=0.18
                elif recent_direction == 2 and tx < px:
                    momentum_bonus=0.12
                elif recent_direction == 2 and tx > px:
                    counter_momentum_penalty=0.18
                elif recent_direction == 3 and tx > px:
                    momentum_bonus=0.12
                elif recent_direction == 3 and tx < px:
                    counter_momentum_penalty=0.18
                score -= momentum_bonus
                score += counter_momentum_penalty
                target_specs.append({
                    'score': score,
                    'priority': priority,
                    'distance': dist,
                    'target_y': ty,
                    'target_x': tx,
                    'player_y': py,
                    'player_x': px,
                    'area': area,
                    'continuity_bonus': continuity_bonus,
                    'momentum_bonus': momentum_bonus,
                    'counter_momentum_penalty': counter_momentum_penalty,
                })
        target_specs.sort(key=lambda item: (round(item['score'], 6), -item['continuity_bonus'], item['counter_momentum_penalty'], -item['momentum_bonus'], -item['area']))
        s._semantic_target_candidates_cache_key=cache_key
        s._semantic_target_candidates_cache_value=target_specs
        return target_specs

    def _semantic_target_choice(s, frame, blocked_click_coord=None):
        """Return the best semantic target using class priority plus distance."""
        target_specs=s._semantic_target_candidates(frame, blocked_click_coord=blocked_click_coord)
        if not target_specs:
            return None
        return dict(target_specs[0])

    def _semantic_direction_action(s, frame, avail):
        """Choose a directional move that heads toward a likely target."""
        legal_dirs={aid for aid in s._available_action_ids(avail) if 1<=aid<=4}
        if not legal_dirs:
            return None
        blocked=s._blocked_direction_action_index(frame)
        preferred_axis=s._recent_direction_axis(frame)
        for choice in s._semantic_target_candidates(frame):
            py=choice['player_y']; px=choice['player_x']
            ty=choice['target_y']; tx=choice['target_x']
            dy=ty-py
            dx=tx-px
            pref=None
            alt=None
            if abs(dx) > abs(dy) or (abs(dx) == abs(dy) and preferred_axis != 'vertical'):
                if dx > 0:
                    pref=4
                elif dx < 0:
                    pref=3
                if dy > 0:
                    alt=2
                elif dy < 0:
                    alt=1
            else:
                if dy > 0:
                    pref=2
                elif dy < 0:
                    pref=1
                if dx > 0:
                    alt=4
                elif dx < 0:
                    alt=3
            for aid in (pref, alt):
                if aid is None:
                    continue
                if blocked is not None and (aid-1) == blocked:
                    continue
                if aid in legal_dirs:
                    return aid-1, None
        return None

    def _frame_matches_previous(s, frame):
        """Return True when the current raw frame matches the stored previous frame."""
        if s.pr is None:
            return False
        try:
            return np.shape(frame) == np.shape(s.pr) and np.array_equal(frame, s.pr)
        except Exception:
            return False

    def _frame_changed_since_previous(s, frame):
        """Return True when the current frame safely differs from the stored previous frame."""
        return s.pr is not None and not s._frame_matches_previous(frame)

    def _recent_direction_action_index(s, frame):
        """Return the last directional action index when it changed the frame."""
        if s.pai is None:
            return None
        if not (0 <= int(s.pai) < 4):
            return None
        if s._frame_changed_since_previous(frame):
            return int(s.pai)
        return None

    def _recent_direction_axis(s, frame):
        """Return the axis implied by the most recent effective directional action."""
        recent_direction=s._recent_direction_action_index(frame)
        if recent_direction is None:
            return None
        return 'vertical' if recent_direction in (0, 1) else 'horizontal'

    def _recent_click_action_index(s, frame):
        """Return the last click action index when it changed the frame."""
        if s.pai is None:
            return None
        click_base=5
        click_limit=click_base + s.G * s.G
        if not (click_base <= int(s.pai) < click_limit):
            return None
        if s._frame_changed_since_previous(frame):
            return int(s.pai)
        return None

    def _blocked_direction_action_index(s, frame):
        """Return the last directional action index if it left the state unchanged."""
        recent_direction=s._recent_direction_action_index(frame)
        if recent_direction is not None:
            return None
        if s.pai is None or s.pr is None:
            return None
        if not (0 <= int(s.pai) < 4):
            return None
        if s._frame_matches_previous(frame):
            return int(s.pai)
        return None

    def _blocked_click_coord(s, frame):
        """Return the last click coordinate if it left the state unchanged."""
        recent_click=s._recent_click_action_index(frame)
        if recent_click is not None:
            return None
        if s.pai is None or s.pr is None:
            return None
        click_base=5
        click_limit=click_base + s.G * s.G
        if not (click_base <= int(s.pai) < click_limit):
            return None
        if s._frame_matches_previous(frame):
            return s._click_coord_from_action_index(s.pai)
        return None

    def _blocked_click_action_index(s, frame):
        """Return the last click action index if it left the state unchanged."""
        coord=s._blocked_click_coord(frame)
        if coord is None:
            return None
        return s._click_action_index(coord)

    def _coord_matches_blocked_click(s, coord, blocked_click_coord):
        """Treat nearby click jitter as the same blocked click region."""
        return (blocked_click_coord is not None and
                (abs(int(coord[0])-int(blocked_click_coord[0])) +
                 abs(int(coord[1])-int(blocked_click_coord[1]))) <= 2)

    def _blocked_click_matches_coord(s, frame, coord, blocked_click_coord=None):
        """Treat nearby click jitter as the same blocked click region."""
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame)
        return s._coord_matches_blocked_click(coord, blocked_click_coord)

    def _semantic_direction_bonuses(s, frame, avail=None):
        """Soft directional preferences derived from semantic targets."""
        blocked=s._blocked_direction_action_index(frame)
        legal_dirs=None
        if avail is not None:
            legal_dirs={aid for aid in s._available_action_ids(avail) if 1<=aid<=4}
            if not legal_dirs:
                return {}
        preferred_axis=s._recent_direction_axis(frame)
        for choice in s._semantic_target_candidates(frame):
            py=choice['player_y']; px=choice['player_x']
            ty=choice['target_y']; tx=choice['target_x']
            dy=ty-py
            dx=tx-px
            pref=None
            alt=None
            if abs(dx) > abs(dy) or (abs(dx) == abs(dy) and preferred_axis != 'vertical'):
                if dx > 0:
                    pref=3
                elif dx < 0:
                    pref=2
                if dy > 0:
                    alt=1
                elif dy < 0:
                    alt=0
            else:
                if dy > 0:
                    pref=1
                elif dy < 0:
                    pref=0
                if dx > 0:
                    alt=3
                elif dx < 0:
                    alt=2
            bonuses={}
            for idx, bonus in ((pref, 0.45), (alt, 0.18)):
                if idx is None:
                    continue
                if legal_dirs is not None and (idx + 1) not in legal_dirs:
                    continue
                if idx not in bonuses:
                    bonuses[idx]=bonus
            if bonuses:
                if blocked is not None:
                    if blocked in bonuses:
                        bonuses[blocked] = min(bonuses[blocked], -0.12)
                    elif 0 <= blocked < 4:
                        bonuses[blocked] = -0.12
                return bonuses
        return {}

    def _semantic_exploration_logits(s, frame, avail, include_clicks, blocked_click_coord=None):
        """Bias exploratory sampling toward semantic movement/click targets."""
        size=4101 if include_clicks else 5
        logits=torch.zeros(size, device=s.device)
        blocked_direction=s._blocked_direction_action_index(frame)
        for action_idx, bonus in s._semantic_direction_bonuses(frame, avail).items():
            if 0 <= int(action_idx) < 5:
                logits[int(action_idx)] = float(bonus)
        if blocked_direction is not None and 0 <= int(blocked_direction) < 5:
            logits[int(blocked_direction)] = -float('inf')
        if include_clicks:
            if blocked_click_coord is None:
                blocked_click_coord=s._blocked_click_coord(frame)
            click_scale=s._semantic_click_bonus_scale(frame)
            preferred_click_coord=s._preferred_click_coord()
            for rank,(ty,tx) in enumerate(s._semantic_click_targets_compat(
                    frame,
                    limit=6,
                    blocked_click_coord=blocked_click_coord)):
                idx=s._click_action_index((ty, tx))
                if 5 <= idx < logits.numel():
                    logits[idx] = max(float(logits[idx].item()), max(0.0, 0.8 - 0.1 * rank) * click_scale)
            if (preferred_click_coord is not None and
                    not s._blocked_click_matches_coord(
                        frame,
                        preferred_click_coord,
                        blocked_click_coord=blocked_click_coord)):
                preferred_idx=s._click_action_index(preferred_click_coord)
                if 5 <= preferred_idx < logits.numel():
                    logits[preferred_idx] = max(float(logits[preferred_idx].item()), 0.08 * click_scale)
            blocked_click=s._blocked_click_action_index(frame)
            if blocked_click is not None and blocked_click < logits.numel():
                logits[blocked_click] = -float('inf')
            if blocked_click_coord is not None:
                by,bx=int(blocked_click_coord[0]), int(blocked_click_coord[1])
                for dy in range(-2,3):
                    for dx in range(-2,3):
                        if abs(dy) + abs(dx) > 2:
                            continue
                        ny=by+dy
                        nx=bx+dx
                        if not (0 <= ny < s.G and 0 <= nx < s.G):
                            continue
                        idx=s._click_action_index((ny, nx))
                        logits[idx] = -float('inf')
        return logits

    def _semantic_candidate_action_indices(s, frame, include_clicks, avail=None,
                                           direction_bonuses=None, click_targets=None,
                                           blocked_click_coord=None):
        """Semantic action indices that should always participate in rescoring."""
        candidates=[]
        seen=set()
        if direction_bonuses is None:
            direction_bonuses=s._semantic_direction_bonuses(frame, avail)
        for action_idx in direction_bonuses.keys():
            idx=int(action_idx)
            if 0 <= idx < 5 and idx not in seen:
                seen.add(idx)
                candidates.append(idx)
        if include_clicks:
            if click_targets is None:
                click_targets=s._semantic_click_targets_compat(
                    frame,
                    limit=6,
                    blocked_click_coord=blocked_click_coord,
                )
            for ty,tx in click_targets:
                idx=s._click_action_index((ty, tx))
                if idx not in seen and 5 <= idx < 4101:
                    seen.add(idx)
                    candidates.append(idx)
            preferred_click_coord=s._preferred_click_coord()
            if (preferred_click_coord is not None and
                    not s._blocked_click_matches_coord(
                        frame,
                        preferred_click_coord,
                        blocked_click_coord=blocked_click_coord)):
                preferred_idx=s._click_action_index(preferred_click_coord)
                if preferred_idx not in seen and 5 <= preferred_idx < 4101:
                    seen.add(preferred_idx)
                    candidates.append(preferred_idx)
        return candidates

    def _semantic_goal_distance(s, frame, blocked_click_coord=None):
        """Estimated player-to-target Manhattan distance from semantic detections."""
        choice=s._semantic_target_choice(frame, blocked_click_coord=blocked_click_coord)
        if not choice:
            return None
        return float(choice['distance'])

    def _semantic_click_bonus_scale(s, frame, blocked_click_coord=None):
        """Reduce click priors when the semantic target is far from the player."""
        goal_distance=s._semantic_goal_distance(frame, blocked_click_coord=blocked_click_coord)
        if goal_distance is None:
            return 1.0
        return max(0.25, min(1.0, 4.0 / max(float(goal_distance), 1.0)))

    def _refresh_semantic_target_coord(s, frame, fallback_coord=None, blocked_click_coord=None):
        """Track the current semantic target so later tie-breaks keep pursuing it."""
        choice=s._semantic_target_choice(frame, blocked_click_coord=blocked_click_coord)
        if choice is not None:
            s._semantic_target_coord=(int(round(choice['target_y'])), int(round(choice['target_x'])))
        elif (fallback_coord is not None and
              not s._blocked_click_matches_coord(
                  frame,
                  fallback_coord,
                  blocked_click_coord=blocked_click_coord)):
            s._semantic_target_coord=(int(fallback_coord[0]), int(fallback_coord[1]))
        else:
            s._semantic_target_coord=None

    def _heuristic(s, frame, avail, step, blocked_click_coord=None):
        av=set(s._available_action_ids(avail))
        semantic_dir=s._semantic_direction_action(frame, avail)
        if semantic_dir is not None:
            return semantic_dir
        blocked=s._blocked_direction_action_index(frame)
        preferred_dir=int(s.pai) if s.pai is not None and 0 <= int(s.pai) < 4 else None
        preferred_coord=s._preferred_click_coord()
        if step < 4:
            preferred_choice=s._preferred_direction_choice(preferred_dir, blocked, av)
            if preferred_choice is not None:
                return preferred_choice
        for d in [1, 2, 3, 4]:
            if blocked is not None and (d-1) == blocked:
                continue
            if d in av and step < 4:
                return d - 1, None
        if 6 in av:
            semantic_targets=s._semantic_click_targets_compat(
                frame,
                blocked_click_coord=blocked_click_coord,
            )
            semantic_target_choice=s._preferred_click_target_choice(semantic_targets, preferred_coord, step)
            if semantic_target_choice is not None:
                return 5, semantic_target_choice
            cnt=np.bincount(frame.flatten(), minlength=16)
            targets=[]
            for c in range(16):
                if c == s._bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys,xs=np.where(frame==c)
                if len(ys) >= 2:
                    coord=(int(np.median(ys)), int(np.median(xs)))
                    if s._blocked_click_matches_coord(
                            frame,
                            coord,
                            blocked_click_coord=blocked_click_coord):
                        continue
                    targets.append((coord[1],coord[0],len(ys)))
            targets.sort(key=lambda t:t[2])
            fallback_targets=[(int(ty), int(tx)) for tx,ty,_ in targets]
            fallback_target_choice=s._preferred_click_target_choice(fallback_targets, preferred_coord, step)
            if fallback_target_choice is not None:
                return 5, fallback_target_choice
        choices=[a for a in av if 1<=a<=5]
        preferred_choice=s._preferred_direction_choice(preferred_dir, blocked, choices)
        if preferred_choice is not None:
            return preferred_choice
        if 5 in av and not any(a in av for a in (1, 2, 3, 4)):
            return 4, None
        directional_choices=[a for a in choices if 1<=a<=4]
        if blocked is not None:
            unblocked_directional_choices=[a for a in directional_choices if (a-1) != blocked]
            if unblocked_directional_choices:
                directional_choices=unblocked_directional_choices
            elif 5 in av:
                return 4, None
        if directional_choices:
            return random.choice(directional_choices) - 1, None
        if choices:
            return random.choice(choices) - 1, None
        return 0, None

    def _replay_batch_tensor(s, indices):
        """Encode an entire replay batch on the target device.

        One-hot/edge/rarity features are cached per unique frame hash so that
        frames sampled multiple times across consecutive training steps avoid
        redundant GPU feature engineering.
        """
        frame_hashes=[s._fast_frame_hash(s.buf[i]) for i in indices]
        # Group uncached frames and compute features in one batched pass
        uncached=[]
        seen=set()
        for i,h in enumerate(frame_hashes):
            if h not in s._frame_feature_cache and h not in seen:
                seen.add(h)
                uncached.append(i)
        if uncached:
            frames_np=np.stack([s.buf[indices[i]] for i in uncached],axis=0)
            frames=torch.from_numpy(frames_np).to(s.device,non_blocking=True).long()
            B=frames.size(0)
            oh=F.one_hot(frames,num_classes=16).permute(0,3,1,2).to(dtype=torch.float32)
            counts=oh.sum(dim=(2,3))
            bg=counts.argmax(dim=1)
            mx=counts.gather(1,bg.unsqueeze(1)).clamp_min_(1.0)
            bg_m=(frames==bg.view(B,1,1)).unsqueeze(1).to(dtype=torch.float32)
            rarity_lut=1.0-counts/mx
            rarity=rarity_lut.gather(1,frames.reshape(B,-1)).reshape(B,1,64,64)
            pad=F.pad(frames.unsqueeze(1),(1,1,1,1),mode='replicate').squeeze(1)
            edge=((frames!=pad[:,:-2,1:-1]) | (frames!=pad[:,2:,1:-1]) |
                  (frames!=pad[:,1:-1,:-2]) | (frames!=pad[:,1:-1,2:]))
            edge=edge.unsqueeze(1).to(dtype=torch.float32)
            for j,pos_in_indices in enumerate(uncached):
                h=frame_hashes[pos_in_indices]
                s._frame_feature_cache[h]=(oh[j:j+1],bg_m[j:j+1],rarity[j:j+1],edge[j:j+1])
            # Evict oldest entries when cache exceeds limit
            if len(s._frame_feature_cache)>s._frame_feature_cache_max:
                for _ in range(len(s._frame_feature_cache)-s._frame_feature_cache_max):
                    s._frame_feature_cache.pop(next(iter(s._frame_feature_cache)))
        # Gather cached features for all indices
        oh_parts,bg_parts,rar_parts,edge_parts=[],[],[],[]
        for h in frame_hashes:
            oh_i,bg_i,ra_i,ed_i=s._frame_feature_cache[h]
            oh_parts.append(oh_i);bg_parts.append(bg_i);rar_parts.append(ra_i);edge_parts.append(ed_i)
        oh=torch.cat(oh_parts,dim=0)
        bg_m=torch.cat(bg_parts,dim=0)
        rarity=torch.cat(rar_parts,dim=0)
        edge=torch.cat(edge_parts,dim=0)
        B=len(indices)
        if s._pos_aug_device is None or s._pos_aug_device.device!=s.device:
            s._pos_aug_device=s._pos_aug.to(s.device)
        pos=s._pos_aug_device.unsqueeze(0).expand(B,-1,-1,-1)
        zeros=oh.new_zeros((B,5,64,64))
        states=torch.cat([oh,bg_m,rarity,edge,pos,zeros],dim=1)
        if s.device.type=='cuda':
            states=states.contiguous(memory_format=torch.channels_last)
        return states

    def _train(s):
        if len(s.buf)<s.bsz:return False
        # PER sampling: importance-weighted by priority
        n=len(s.buf)
        priorities=np.array(s.buf_priorities if s.buf_priorities else [1.0]*n,dtype=np.float64)
        probs=priorities**s._per_alpha
        probs/=probs.sum()
        indices=list(np.random.choice(n,size=s.bsz,p=probs))
        acts_np=np.fromiter((s.buf_actions[i] for i in indices),dtype=np.int64,count=s.bsz)
        rews_np=np.fromiter((s.buf_rewards[i] for i in indices),dtype=np.float32,count=s.bsz)
        needs_click_head=any(s.buf_actions[i]>=5 for i in indices)
        # Importance sampling weights
        is_weights=(1.0/(n*probs[indices]))**s._per_beta
        is_weights/=is_weights.max()
        s._per_beta=min(1.0,s._per_beta+s._per_beta_step)
        states=s._replay_batch_tensor(indices)
        acts=torch.from_numpy(acts_np).to(s.device,non_blocking=True)
        rews=torch.from_numpy(rews_np).to(s.device,non_blocking=True)
        isw=torch.from_numpy(is_weights.astype(np.float32)).to(s.device,non_blocking=True)
        s.net.train();s.opt.zero_grad(set_to_none=True)
        try:
            with s._amp_context():
                logits=s.net(states) if needs_click_head else s.net.forward_actions(states)
                acts_c=acts.clamp(0,logits.size(1)-1)
                q_sa=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
                # Munchausen DQN target: r + alpha*tau*log(pi(a|s)) + gamma*max_a' Q_target(s',a')
                td_target=rews.clone()
                has_next_mask=torch.tensor([idx<len(s.buf_next_frames) and s.buf_next_frames[idx] is not None for idx in indices],device=s.device)
                if has_next_mask.any() and s._target_net is not None:
                    next_indices=[indices[i] for i in range(s.bsz) if has_next_mask[i]]
                    next_states=s._replay_batch_tensor(next_indices)
                    with torch.no_grad():
                        online_logits=s.net(next_states)
                        best_actions=online_logits.argmax(dim=1)
                        target_logits=s._target_net(next_states)
                        max_next_q=target_logits.gather(1,best_actions.unsqueeze(1)).squeeze(1)
                    td_target[has_next_mask]=rews[has_next_mask]+s.gamma*max_next_q
                # Munchausen bonus: alpha*tau*log(pi(a|s)) for direction actions only
                with torch.no_grad():
                    mdqn_acts=acts_c<5
                    log_pi_a=acts_c.new_zeros(acts_c.shape,dtype=torch.float32)
                    if mdqn_acts.any():
                        dir_lp=F.log_softmax(logits[mdqn_acts,:5]/s._mdqn_tau,dim=1)
                        log_pi_a[mdqn_acts]=dir_lp.gather(1,acts_c[mdqn_acts].unsqueeze(1)).squeeze(1)
                td_target=td_target+s._mdqn_alpha*s._mdqn_tau*log_pi_a
                loss=(isw*F.mse_loss(q_sa,td_target,reduction='none')).mean()
                loss=loss-0.0001*logits[:,:5].mean()
                if needs_click_head:
                    loss=loss-0.00001*logits[:,5:].mean()
            if s._grad_scaler is not None:
                s._grad_scaler.scale(loss).backward()
                s._grad_scaler.unscale_(s.opt)
                torch.nn.utils.clip_grad_norm_(s.net.parameters(), max_norm=1.0)
                s._grad_scaler.step(s.opt)
                s._grad_scaler.update()
                if s.scheduler is not None: s.scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s.net.parameters(), max_norm=1.0)
                s.opt.step()
                if s.scheduler is not None: s.scheduler.step()
            # Update priorities with TD error
            with torch.no_grad():
                td_error=(q_sa-td_target).abs().cpu().numpy()
                for i,idx in enumerate(indices):
                    s.buf_priorities[idx]=max(float(td_error[i])+0.01,0.01)
            # Polyak update for target network
            if s._target_net is not None:
                with torch.no_grad():
                    for p,tp in zip(s.net.parameters(), s._target_net.parameters()):
                        tp.mul_(1-s.tau).add_(p, alpha=s.tau)
                s._target_update_counter+=1
                if s._target_update_counter>=s._target_hard_update_interval:
                    s._target_net.load_state_dict(s.net.state_dict())
                    s._target_update_counter=0
            s._model_revision += 1
            s._aem_encoded_cache_sig=None; s._aem_encoded_cache=None
            return True
        finally:
            s.net.eval()

    def _bc_train_on_solution(s, raw_frames, action_indices, batch_size, epochs):
        """Supervised behavior cloning on a direction-only solution trajectory."""
        if len(raw_frames) < 2 or len(raw_frames) != len(action_indices):
            return None
        s.net.train()
        device = s.device
        # Random translation augmentation preserves the coarse board layout while
        # keeping BC inputs aligned with the replay/inference encoder.
        bg_color = int(np.bincount(
            np.ascontiguousarray(raw_frames[0], dtype=np.uint8).ravel(),
            minlength=16).argmax())
        shift_dx = random.randint(-1, 1)
        shift_dy = random.randint(-1, 1)
        do_shift = (shift_dx != 0 or shift_dy != 0) and random.random() < 0.5
        tensors = []
        for frame in raw_frames:
            frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
            if do_shift:
                pad = np.pad(frame_c, ((1, 1), (1, 1)), mode='constant',
                             constant_values=bg_color)
                frame_c = pad[1 + shift_dy:65 + shift_dy,
                              1 + shift_dx:65 + shift_dx]
            tensors.append(s._encode_frame_tensor(frame_c).to(device, non_blocking=True))
        n = len(tensors)
        indices = list(range(n))
        total_loss = 0.0
        step_count = 0
        try:
            for _ in range(epochs):
                random.shuffle(indices)
                for start in range(0, n, batch_size):
                    bidx = indices[start:start + batch_size]
                    states = torch.stack([tensors[i] for i in bidx])
                    targets = torch.tensor([action_indices[i] for i in bidx], device=device, dtype=torch.long)
                    s.opt.zero_grad(set_to_none=True)
                    with s._amp_context():
                        logits = s.net.forward_actions(states)
                        loss = F.cross_entropy(logits, targets)
                    if s._grad_scaler is not None:
                        s._grad_scaler.scale(loss).backward()
                        s._grad_scaler.unscale_(s.opt)
                        torch.nn.utils.clip_grad_norm_(s.net.parameters(), max_norm=1.0)
                        s._grad_scaler.step(s.opt)
                        s._grad_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(s.net.parameters(), max_norm=1.0)
                        s.opt.step()
                    total_loss += loss.item()
                    step_count += 1
        except Exception as e:
            s.net.eval()
            logger.warning(f"BC training failed: {e}")
            return None
        s.net.eval()
        s._model_revision += 1
        s._aem_encoded_cache_sig = None
        s._aem_encoded_cache = None
        return total_loss / max(1, step_count)

    def _maybe_train(s, max_steps=1, force=False):
        # Training is useful but it can stall action selection.  Gate it by
        # action count and cap the burst length so play keeps moving.
        if not s._wd or len(s.buf) < s.bsz or s.net is None or s.opt is None:
            return 0
        if not force and (s.action_counter - s._last_train_action) < s._train_min_gap:
            return 0
        steps=0
        for _ in range(max(1, int(max_steps))):
            if not s._train():
                break
            steps += 1
        if steps:
            s._last_train_action=s.action_counter
        return steps

    def _get_aem_tensors(s):
        M=len(s._aem_diffs)
        if M<2:return None,None,None

        # AEM is useful, but the old path re-encoded up to 256 diffs every
        # inference.  Use only the most recent/action-relevant window and move
        # it to the device in one batched transfer.
        K=min(M, getattr(s, '_aem_max_active', 128))
        last_id=id(s._aem_diffs[-1]) if M else 0
        sig=(M, K, len(s._aem_actions), len(s._aem_rewards), last_id, s.device.type)
        if s._aem_cache_sig == sig:
            return s._aem_cache

        diffs_l=list(s._aem_diffs)[-K:]
        acts_l=list(s._aem_actions)[-K:]
        rews_l=list(s._aem_rewards)[-K:]

        # Stack on CPU first; assigning one small tensor at a time into a CUDA
        # tensor causes many tiny transfers/synchronization points.
        diffs_np=np.stack([d.astype(np.float32, copy=False) for d in diffs_l], axis=0)
        diffs=torch.as_tensor(diffs_np, dtype=torch.float32, device=s.device).view(1,K,1,64,64)
        acts=torch.as_tensor([min(int(a),4) for a in acts_l], dtype=torch.long, device=s.device).view(1,K)
        rews=torch.as_tensor([float(r) for r in rews_l], dtype=torch.float32, device=s.device).view(1,K)

        s._aem_cache_sig=sig
        s._aem_cache=(diffs,acts,rews)
        return s._aem_cache

    def _get_aem_encoded(s, mem):
        """Reuse diff encoder output until replayed action-effect memory changes."""
        if mem[0] is None or s.net is None:
            return None
        sig=(s._aem_cache_sig,s._model_revision)
        if s._aem_encoded_cache_sig == sig and s._aem_encoded_cache is not None:
            return s._aem_encoded_cache
        encoded=s.net.aea.encode_memory(*mem)
        s._aem_encoded_cache_sig=sig
        s._aem_encoded_cache=encoded
        return encoded

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 6*3600-180
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Completion bonus: the previous action just advanced the level
                if s.pai is not None and s.pr is not None and lvl > s.cl:
                    s._add_replay(s.pr, s.pai, 15.0)
                    # Retroactively boost recent transitions that contributed
                    gamma = 0.95; bonus = 10.0
                    for i in range(len(s.buf_rewards) - 1, -1, -1):
                        s.buf_rewards[i] += bonus; bonus *= gamma
                        if i < len(s.buf_priorities):
                            s.buf_priorities[i] = float(s.buf_rewards[i]) + 0.01
                        if bonus < 0.1: break
                # Init BFS solver on first level
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # Try BFS for this level
                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl, lf=lf)

                # Init CNN fallback state.  Keep the same network/optimizer across
                # levels so learned features are not discarded and we avoid repeated CUDA
                # allocation + checkpoint probing on every level change.
                s._clear_replay()
                s.buf_h.clear()
                if s.net is None:
                    s.net = ForgeNet(s.IN, s.G).to(s.device)
                    if not s._weights_loaded:
                        for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                                   'pretrained_weights.pt']:
                            try:
                                if os.path.exists(wp):
                                    state=torch.load(wp,map_location=s.device,weights_only=True)
                                    ms=s.net.state_dict()
                                    loaded_keys=0
                                    for k in list(state.keys()):
                                        if k in ms and state[k].shape==ms[k].shape:
                                            ms[k]=state[k]; loaded_keys+=1
                                    s.net.load_state_dict(ms)
                                    s._weights_loaded=True
                                    logger.info(f"CNN weights loaded from {wp} ({loaded_keys}/{len(ms)} keys)")
                                    break
                                else:
                                    logger.info(f"CNN weights not found at {wp}")
                            except Exception as e:
                                logger.warning(f"CNN weights load failed from {wp}: {e}")
                        if not s._weights_loaded:
                            logger.info("CNN starting from random init (no pretrained weights)")
                    if s.device.type == 'cuda':
                        s.net=s.net.to(memory_format=torch.channels_last)
                    s.opt = s._make_optimizer()
                    s.scheduler = s._make_scheduler()
                    s._target_net = copy.deepcopy(s.net)
                    s._target_net.eval()
                    if s.device.type == 'cuda':
                        try:
                            import triton
                            s.net=torch.compile(s.net,mode='reduce-overhead',fullgraph=False)
                            logger.info("CNN compiled: mode=reduce-overhead")
                        except Exception:
                            pass
                else:
                    rebuilt_opt = False
                    if s.opt is None:
                        s.opt = s._make_optimizer()
                        rebuilt_opt = True
                    if (rebuilt_opt or s.scheduler is None) and s.opt is not None:
                        s.scheduler = s._make_scheduler()
                    if s._target_net is None:
                        s._target_net = copy.deepcopy(s.net)
                        s._target_net.eval()
                s.net.eval()
                s._clear_recent_action_state()
                s._semantic_target_coord=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None;s._wm_dev=None;s._wm_log_dev=None;s._wm_cache_key=None;s._aem_cache_sig=None;s._aem_cache=(None,None,None);s._aem_encoded_cache_sig=None;s._aem_encoded_cache=None
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                # FIX 1: Reset visited hashes on every level change
                s._visited_hashes = set()
                s._state_visit_counts = {}
                # FIX 4: Only reset epsilon if BFS didn't solve this level.
                # If BFS solved it, keep current eps so CNN fallback (if needed)
                # benefits from accumulated exploration knowledge.
                if not s._bfs_solution:
                    s._eps = 0.15
                    s._eps_steps = 0

                # BFS solution injection: replay the current level's solution as
                # expert demonstrations for CNN training, giving in-level behavioral
                # cloning signal that persists across levels via _clear_replay(keep_frac).
                if s._bfs_solution and len(s._bfs_solution) > 1:
                    sol = s._bfs_solution
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            prev_frame = _frame_view(r0.frame[-1], np.uint8)
                            bc_frames = []  # collect raw frames for BC training
                            bc_actions = []
                            for act_id, data in sol:
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                next_frame = _frame_view(result.frame[-1], np.uint8) if result and result.frame else None
                                # BC only trains the 5-way directional head.
                                if action_idx < 5:
                                    bc_frames.append(prev_frame.copy())
                                    bc_actions.append(action_idx)
                                # Inject each transition 3x into replay buffer for richer DQN sampling
                                for _ in range(3):
                                    s._add_replay(prev_frame, action_idx, 2.0, next_frame=next_frame)
                                if next_frame is not None:
                                    prev_frame = next_frame
                            # DQN pretraining on injected replay transitions
                            pretrain_bsz = min(s.bsz, len(s.buf))
                            if pretrain_bsz >= 4:
                                old_bsz = s.bsz; s.bsz = pretrain_bsz
                                dqn_steps = min(200, max(30, len(sol) * 10))
                                for _ in range(dqn_steps):
                                    if not s._train():
                                        break
                                s.bsz = old_bsz
                                logger.info(f"BFS solution injection: DQN {dqn_steps}x on {len(sol)} demos (3x replicated) from L{lvl}")
                            # BC (behavior cloning) — supervised cross-entropy on solution actions.
                            # Small batch sizes = more gradient steps per epoch, which matters when
                            # only a handful of solution states are available.
                            if len(bc_frames) >= 2:
                                bc_bsz = min(4, len(bc_frames))
                                bc_epochs = max(60, min(300, len(bc_frames) * 8))
                                _saved_bg = s._bg  # preserve current background color
                                bc_loss = MyAgent._bc_train_on_solution(s, bc_frames, bc_actions, bc_bsz, bc_epochs)
                                s._bg = _saved_bg
                                if bc_loss is not None:
                                    logger.info(f"BFS solution injection: BC {bc_epochs} epochs (bsz={bc_bsz}), final loss={bc_loss:.4f}")
                    except Exception as e:
                        logger.warning(f"BFS solution injection failed: {e}")

                # CLTI — inject BFS demos from previous level into CNN replay buffer
                # FIX 2: Use perform_action frame[-1] consistently with _raw(),
                # instead of get_pixels() which returns a different format.
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl - 1)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            # Start from the post-reset frame, consistent with _raw()
                            prev_frame = _frame_view(r0.frame[-1], np.uint8)
                            for act_id, data in prev_sol:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                s._add_replay(prev_frame, action_idx, 2.0)
                                # Advance prev_frame using the action result, not get_pixels()
                                if result.frame:
                                    prev_frame = _frame_view(result.frame[-1], np.uint8)
                            pretrain_bsz = min(s.bsz, len(s.buf))
                            if pretrain_bsz >= 4:
                                old_bsz = s.bsz; s.bsz = pretrain_bsz
                                pretrain_steps = min(25, max(5, len(prev_sol) * 3 // pretrain_bsz + 1))
                                for _ in range(pretrain_steps):
                                    if not s._train():
                                        break
                                s.bsz = old_bsz
                                logger.info(f"CLTI: pre-trained {pretrain_steps}x on {len(prev_sol)} demos from L{lvl-1}")
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")

                # BFS warm-fallthrough: seed CNN replay with BFS-discovered effective
                # actions from the current level's partial search.  These actions are
                # guaranteed to produce frame changes (discovered by _scan_actions),
                # giving the CNN immediate positive signal even when BFS didn't find
                # a full solution.  Replays each action on a clone to capture the
                # proper next_frame for TD bootstrapping.  Also performs BFS tree
                # replay to generate multi-step training data from explored states.
                if not s._bfs_solution and s._bfs and getattr(s._bfs, '_last_effective_actions', None):
                    eff = s._bfs._last_effective_actions
                    count = 0
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            root_frame = _frame_view(r0.frame[-1], np.uint8)
                            for act_id, data in eff[:500]:
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * s.G + data.get('x', 0) if data else 0)
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                g = s._bfs._clone_game(replay_game)
                                result = g.perform_action(ai, raw=True)
                                if result.frame:
                                    next_frame = _frame_view(result.frame[-1], np.uint8)
                                    s._add_replay(root_frame, action_idx, 0.8, next_frame=next_frame)
                                else:
                                    s._add_replay(root_frame, action_idx, 0.8)
                                if act_id == 6 and data and s._wm is not None:
                                    x, y = data.get('x', -1), data.get('y', -1)
                                    if 0 <= x < 64 and 0 <= y < 64:
                                        s._wm[y, x] = max(s._wm[y, x], 2.0)
                                count += 1
                    except Exception:
                        raw_frame = s._raw(lf)
                        for act_id, data in eff[:500]:
                            action_idx = (act_id - 1) if act_id <= 5 else (
                                5 + data.get('y', 0) * s.G + data.get('x', 0) if data else 0)
                            s._add_replay(raw_frame, action_idx, 0.8)
                            if act_id == 6 and data and s._wm is not None:
                                x, y = data.get('x', -1), data.get('y', -1)
                                if 0 <= x < 64 and 0 <= y < 64:
                                    s._wm[y, x] = max(s._wm[y, x], 2.0)
                            count += 1
                    # BFS tree replay: expand effective actions into a depth-limited
                    # action tree, generating multi-step training data from
                    # BFS-discovered action-effect patterns beyond the root state.
                    # Handles warm-up unlocked games (sc25-type) by detecting the
                    # warm-up action and starting expansion from the unlocked state.
                    tree_count = 0
                    if eff:
                        try:
                            tree_game = s._bfs.game_cls()
                            tree_game.set_level(lvl)
                            tree_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            r0 = tree_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            if r0.frame:
                                root_frame = _frame_view(r0.frame[-1], np.uint8)
                                # Find the BFS start state by probing direction actions
                                # until some eff action produces a visible frame change.
                                start_game = tree_game
                                start_frame = root_frame
                                found = False
                                for probe_state in [tree_game]:  # first check root
                                    for warmup_id in [0] + list(range(1, 5)):
                                        g_probe = s._bfs._clone_game(probe_state) if warmup_id > 0 else probe_state
                                        if warmup_id > 0:
                                            ai_warm = ActionInput(id=GameAction.from_id(warmup_id))
                                            rw = g_probe.perform_action(ai_warm, raw=True)
                                            if not rw or not rw.frame:
                                                continue
                                        probe_frame = _frame_view(rw.frame[-1], np.uint8) if warmup_id > 0 else root_frame
                                        # Check if any eff action works from this state
                                        for act_id, data in eff[:min(10, len(eff))]:
                                            g_test = s._bfs._clone_game(g_probe if warmup_id > 0 else probe_state)
                                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                            result = g_test.perform_action(ai, raw=True)
                                            if result and result.frame:
                                                test_frame = _frame_view(result.frame[-1], np.uint8)
                                                if np.any(test_frame != probe_frame):
                                                    start_game = g_probe if warmup_id > 0 else probe_state
                                                    start_frame = probe_frame
                                                    found = True
                                                    break
                                        if found:
                                            break
                                    if found:
                                        break
                                sorted_eff = sorted(eff, key=lambda a: s._bfs._action_priority.get(s._bfs._action_key(a[0], a[1]), 0), reverse=True)
                                top_eff = sorted_eff[:min(3, len(sorted_eff))]
                                frontier = deque()
                                frontier.append((start_game, start_frame, 0))
                                tree_visited = {s._fast_frame_hash(start_frame)}
                                bc_wf_frames = []
                                bc_wf_actions = []
                                tree_attempts = 0
                                tree_attempt_limit = 4000
                                max_depth = 12
                                while frontier and tree_attempts < tree_attempt_limit:
                                    parent_game, parent_frame, depth = frontier.popleft()
                                    if depth >= max_depth:
                                        continue
                                    children = []
                                    branch_eff = sorted_eff if depth == 0 else top_eff
                                    for act_id, data in branch_eff:
                                        if tree_attempts >= tree_attempt_limit:
                                            break
                                        tree_attempts += 1
                                        action_idx = (act_id - 1) if act_id <= 5 else (
                                            5 + data.get('y', 0) * s.G + data.get('x', 0) if data else 0)
                                        g = s._bfs._clone_game(parent_game)
                                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                        result = g.perform_action(ai, raw=True)
                                        if result and result.frame:
                                            child_frame = _frame_view(result.frame[-1], np.uint8)
                                            child_hash = s._fast_frame_hash(child_frame)
                                            if child_hash not in tree_visited and np.any(child_frame != parent_frame):
                                                tree_visited.add(child_hash)
                                                s._add_replay(parent_frame, action_idx, 0.8, next_frame=child_frame)
                                                tree_count += 1
                                                if action_idx < 5:
                                                    bc_wf_frames.append(parent_frame.copy())
                                                    bc_wf_actions.append(int(action_idx))
                                                if depth + 1 < max_depth:
                                                    children.append((g, child_frame, depth + 1))
                                            else:
                                                s._add_replay(parent_frame, action_idx, -0.1)
                                        else:
                                            s._add_replay(parent_frame, action_idx, -0.1)
                                    frontier.extend(children)
                        except Exception as e:
                            logger.warning(f"BFS tree replay failed: {e}")
                    total = count + tree_count
                    if total:
                        pre_steps = min(200, max(30, total * 8 // s.bsz + 1))
                        for _ in range(pre_steps):
                            if not s._train():
                                break
                        logger.info(f"BFS warm-fallthrough: {count} root + {tree_count} tree = {total} transitions, pre-trained {pre_steps}x")
                        # BC training from BFS tree replay: treats BFS-discovered
                        # effective actions as expert demonstrations for the CNN.
                        # More sample-efficient than TD bootstrapping for sparse
                        # rewards, especially on unsolved levels.
                        if len(bc_wf_frames) >= 4:
                            bf_saved = s._bg
                            bc_wf_bsz = min(8, len(bc_wf_frames))
                            bc_wf_epochs = max(30, min(150, len(bc_wf_frames) * 4 // bc_wf_bsz))
                            bc_wf_loss = MyAgent._bc_train_on_solution(
                                s, bc_wf_frames, bc_wf_actions,
                                bc_wf_bsz, bc_wf_epochs)
                            s._bg = bf_saved
                            if bc_wf_loss is not None:
                                logger.info(f"BFS warm-fallthrough BC: {bc_wf_epochs} epochs (bsz={bc_wf_bsz}), "
                                            f"final loss={bc_wf_loss:.4f}")

                # Fast BFS retry with the improved CNN (trained on warm-fallthrough
                # data).  This is much shorter than the adaptive timeout because the
                # CNN now has L1-specific training and guides action ordering better.
                # The initial BFS already used beam search (full→top-2 after 2000 states);
                # the retry uses a tighter beam (top-2 from the start) to reach depth ~13.
                sol_exhausted = (s._bfs_step >= len(s._bfs_solution)) if s._bfs_solution else True
                retry_net = s.net if s.net is not None else None
                if sol_exhausted and s._bfs and retry_net is not None:
                    try:
                        retry_timeout = min(20.0, s._adaptive_bfs_timeout(lvl) * 0.4)
                        logger.info(f"BFS L{lvl}: fast retry with improved CNN (timeout={retry_timeout:.1f}s)")
                        lf_tensor = s._tensor(lf) if lf is not None else None
                        retry_sol = s._bfs.solve_level(
                            lvl, prev_solution=None,
                            timeout=retry_timeout, net=s.net,
                            frame_tensor=lf_tensor)
                        if retry_sol:
                            s._bfs_solution = retry_sol
                            s._bfs_step = 0
                            logger.info(
                                f"BFS L{lvl}: SOLVED on fast retry "
                                f"({len(retry_sol)} actions, "
                                f"post warm-fallthrough training)")
                    except Exception as e:
                        logger.warning(f"BFS fast retry failed: {e}")

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                return s._finalize_control_action(
                    8,
                    "reset",
                    clear_recent=True,
                )

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = s._fresh_action(act_id, data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                tensor = s._tensor(lf)
                raw = s._raw(lf)
                ch = s._fast_frame_hash(raw)
                bfs_click_coord=(int(data.get('y', 0)), int(data.get('x', 0))) if int(act_id) == 6 and data else None
                s._refresh_semantic_target_coord(raw, fallback_coord=bfs_click_coord)
                s.fhist.append(raw.copy())
                if 1 <= int(act_id) <= 5:
                    action_idx = int(act_id) - 1
                elif int(act_id) == 6 and data:
                    action_idx = s._click_action_index(bfs_click_coord)
                else:
                    action_idx = None
                return s._finalize_action(
                    sel,
                    f"bfs:{s._bfs_step}/{len(s._bfs_solution)}",
                    tensor=tensor,
                    raw=raw,
                    frame_hash=ch,
                    action_idx=action_idx,
                    remember_recent=True,
                )

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = s._fast_frame_hash(raw)
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = s._available_action_ids(avail)
            s._undo_avail = 7 in avail_ids
            modeled_avail = any(aid in (1, 2, 3, 4, 5, 6) for aid in avail_ids)

            if s.pt is not None and s.pr is not None:
                curr_objs=None; move_bonus=0.0; moved=0
                diff_map=(s.pr!=raw)&s._reward_mask;changed=bool(np.any(diff_map))
                prev_h = s.ph if s.ph is not None else s._fast_frame_hash(s.pr)
                r = None
                if s.pai is not None:
                    eh=(s._fast_frame_hash(s.pr),int(s.pai))
                    if eh not in s.buf_h:
                        r=s._reward(s.pr,raw,prev_h,ch,changed=changed,curr_objs=curr_objs,move_bonus=move_bonus,moved=moved)
                        s._add_replay(s.pr, s.pai, r, next_frame=raw, dedup_key=eh)
                        if changed:
                            s._aem_diffs.append(diff_map)
                            s._aem_actions.append(min(s.pai,4))
                            s._aem_rewards.append(r)
                            s._aem_cache_sig=None; s._aem_cache=(None,None,None); s._aem_encoded_cache_sig=None; s._aem_encoded_cache=None
                    else:
                        r=s._reward(s.pr,raw,prev_h,ch,changed=changed,curr_objs=curr_objs,move_bonus=move_bonus,moved=moved)
                else:
                    r=s._reward(s.pr,raw,prev_h,ch,changed=changed,curr_objs=curr_objs,move_bonus=move_bonus,moved=moved)
                if changed:s._ckpt_hash=ch;s._unproductive=0
                else:s._unproductive+=1

                # Action repeat: if the last action was a directional move that
                # produced a frame change, repeat it with moderate probability to
                # exploit consistent movement patterns (e.g. walking across a maze)
                # without requiring the CNN to learn to chain identical actions.
                if changed and s.pai is not None and 0 <= s.pai < 4:
                    if random.random() < 0.4:
                        repeat_id = s.pai + 1
                        blocked_click_coord=s._blocked_click_coord(raw)
                        semantic_dir=s._semantic_direction_action(raw, avail)
                        click_avail=6 in avail_ids
                        semantic_clicks=(
                            s._semantic_click_targets_compat(
                                raw,
                                limit=1,
                                blocked_click_coord=blocked_click_coord,
                            )
                            if click_avail else []
                        )
                        preferred_click=s._preferred_click_coord()
                        click_matches_preferred=False
                        click_exact_preferred=False
                        if semantic_clicks and preferred_click is not None:
                            click_match_dist=s._click_coord_distance(semantic_clicks[0], preferred_click)
                            click_matches_preferred=click_match_dist <= 2
                            click_exact_preferred=click_match_dist == 0
                        click_blocks_repeat=(bool(semantic_clicks) and
                                             s._semantic_click_bonus_scale(
                                                 raw,
                                                 blocked_click_coord=blocked_click_coord,
                                             ) >= 0.5 and
                                             (click_exact_preferred or not click_matches_preferred))
                        semantic_repeat_ok=((semantic_dir is not None and semantic_dir[0] == s.pai) or
                                            (semantic_dir is None and not click_blocks_repeat))
                        if semantic_repeat_ok and repeat_id in avail_ids:
                            s.fhist.append(raw.copy())
                            s._refresh_semantic_target_coord(raw)
                            return s._finalize_action(
                                s._fresh_action(repeat_id),
                                f"repeat:a{repeat_id}",
                                tensor=tensor,
                                raw=raw,
                                frame_hash=ch,
                                action_idx=s.pai,
                                remember_recent=True,
                            )

            if not modeled_avail:
                if s._undo_avail:
                    return s._finalize_control_action(
                        7,
                        "undo-only",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        remember_recent=True,
                    )
                return s._finalize_control_action(
                    8,
                    "no-action",
                    clear_recent=True,
                )

            if s._wm is None:s._wm=s._detect_template(raw)

            if s._undo_avail and s._unproductive>=30 and s._ckpt_hash:
                s._unproductive=0
                return s._finalize_control_action(
                    7,
                    "undo",
                    tensor=tensor,
                    raw=raw,
                    frame_hash=ch,
                    remember_recent=True,
                )

            if not s._wd:
                blocked_click_coord=s._blocked_click_coord(raw)
                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la,blocked_click_coord=blocked_click_coord)
                else:
                    s._wd=True
                    s._maybe_train(max_steps=min(2,len(s.buf)//s.bsz), force=True)

            if s._wd:
                a6_avail = 6 in avail_ids
                blocked_click_coord=s._blocked_click_coord(raw)
                if s.net is None:
                    aidx,coords=s._heuristic(raw,avail,s.la,blocked_click_coord=blocked_click_coord)
                elif random.random()<s._eps:
                    prior_logits=s._semantic_exploration_logits(
                        raw,
                        avail,
                        a6_avail,
                        blocked_click_coord=blocked_click_coord,
                    )
                    aidx,coords=s._sample(prior_logits,avail,temp=1.25)
                else:
                    with torch.inference_mode():
                        with s._amp_context():
                            net_input=tensor.unsqueeze(0)
                            if s.device.type == 'cuda':
                                net_input=net_input.contiguous(memory_format=torch.channels_last)
                            mem=s._get_aem_tensors()
                            encoded=s._get_aem_encoded(mem) if mem[0] is not None else None
                            if a6_avail:
                                if mem[0] is not None:logits=s.net(net_input,*mem,mem_encoded=encoded).squeeze(0)
                                else:logits=s.net(net_input).squeeze(0)
                            else:
                                if mem[0] is not None:logits=s.net.forward_actions(net_input,*mem,mem_encoded=encoded).squeeze(0)
                                else:logits=s.net.forward_actions(net_input).squeeze(0)
                            # Flip ensemble removed: 2x speedup, accuracy impact is
                            # negligible with beam search and warm-fallthrough training
                            # (the CNN learns translation-invariant features from BC augmentation).
                    aidx,coords=None,None
                    try:
                        K=5
                        avail_mask=s._legal_action_mask(logits, avail)
                        semantic_clicks={}
                        semantic_click_targets=[]
                        semantic_dirs=s._semantic_direction_bonuses(raw, avail)
                        blocked_direction_idx=s._blocked_direction_action_index(raw)
                        repeat_direction_bonus_idx=s._recent_direction_action_index(raw)
                        repeat_click_bonus_idx=s._recent_click_action_index(raw)
                        preferred_click_coord=s._preferred_click_coord()
                        click_scale=s._semantic_click_bonus_scale(
                            raw,
                            blocked_click_coord=blocked_click_coord,
                        )
                        blocked_click_idx=s._blocked_click_action_index(raw)
                        if a6_avail:
                            semantic_click_targets=s._semantic_click_targets_compat(
                                raw,
                                limit=6,
                                blocked_click_coord=blocked_click_coord,
                            )
                            semantic_clicks=s._semantic_click_bonus_map(
                                raw,
                                limit=6,
                                click_scale=click_scale,
                                click_targets=semantic_click_targets,
                            )
                        n_valid=(avail_mask>-float('inf')).sum().item()
                        if n_valid>0:
                            scored=logits+avail_mask
                            best_k=scored.topk(min(K,n_valid))
                            candidate_indices=[]
                            candidate_seen=set()
                            for idx in best_k.indices.tolist():
                                s._append_candidate_index(candidate_indices, candidate_seen, idx)
                            for idx in s._semantic_candidate_action_indices(
                                    raw,
                                    a6_avail,
                                    avail,
                                    direction_bonuses=semantic_dirs,
                                    click_targets=semantic_click_targets,
                                    blocked_click_coord=blocked_click_coord):
                                s._append_candidate_index(
                                    candidate_indices,
                                    candidate_seen,
                                    idx,
                                    scored=scored,
                                    avail_mask=avail_mask,
                                )
                            if (preferred_click_coord is not None and
                                    not s._blocked_click_matches_coord(
                                        raw,
                                        preferred_click_coord,
                                        blocked_click_coord=blocked_click_coord)):
                                preferred_click_idx=s._click_action_index(preferred_click_coord)
                                s._append_candidate_index(
                                    candidate_indices,
                                    candidate_seen,
                                    preferred_click_idx,
                                    scored=scored,
                                    avail_mask=avail_mask,
                                )
                            best_local=0;best_score=float('-inf')
                            for i,top_idx in enumerate(candidate_indices):
                                score=float(scored[top_idx].item())
                                if top_idx < 5:
                                    if blocked_direction_idx is not None and top_idx == blocked_direction_idx:
                                        score=float('-inf')
                                    else:
                                        score += semantic_dirs.get(top_idx, 0.0)
                                    if (repeat_direction_bonus_idx is not None and
                                            top_idx == repeat_direction_bonus_idx and
                                            top_idx != blocked_direction_idx):
                                        score += 0.08
                                    score += s._bfs_priority_bonus(top_idx + 1)
                                else:
                                    click_coord=s._click_coord_from_action_index(top_idx)
                                    click_data=s._click_action_data(click_coord)
                                    blocked_click_match=s._coord_matches_blocked_click(
                                        click_coord,
                                        blocked_click_coord,
                                    )
                                    if blocked_click_match:
                                        score=float('-inf')
                                    else:
                                        score += s._bfs_priority_bonus(6, click_data)
                                        if s._wm is not None:
                                            score += float(s._wm[click_data["y"], click_data["x"]]) * 0.05
                                        score += semantic_clicks.get(click_coord, 0.0)
                                        score += s._preferred_click_bonus(click_coord, preferred_click_coord)
                                        if repeat_click_bonus_idx is not None and top_idx == repeat_click_bonus_idx and not blocked_click_match:
                                            score += 0.08
                                        if blocked_click_idx is not None and top_idx == blocked_click_idx:
                                            score=float('-inf')
                                if score>best_score:best_score=score;best_local=i
                            top_idx=int(candidate_indices[best_local])
                            aidx,coords=s._decode_policy_action_index(top_idx)
                    except Exception as e:
                        logger.debug("CNN action rescoring unavailable: %s", e)
                    if aidx is None:aidx,coords=s._sample(logits,avail,temp=0.5)
                # Cosine annealing epsilon schedule: gradual decay then plateau
                s._eps_steps+=1; total_steps=5000
                frac=min(s._eps_steps/total_steps,1.0)
                s._eps=s._eps_min+(0.15-s._eps_min)*0.5*(1+np.cos(np.pi*frac))
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            if aidx<5:sel=s._fresh_action(aidx + 1);reasoning=f"cnn:a{aidx+1}"
            else:
                y,x=coords
                sel=s._click_action((y, x));reasoning=f"cnn:c({x},{y})"
            s._refresh_semantic_target_coord(
                raw,
                fallback_coord=coords if aidx >= 5 else None,
                blocked_click_coord=blocked_click_coord,
            )
            action_idx = aidx if aidx < 5 else s._click_action_index(coords)
            # Schedule: increase training frequency as more data accumulates.
            # On unsolved levels (exhausted BFS solution), train every step
            # to make the most of limited gameplay data.
            next_action_counter = s.action_counter + 1
            sol_exhausted = (s._bfs_step >= len(s._bfs_solution)) if s._bfs_solution else True
            if sol_exhausted:
                s.tfreq = 1
            else:
                progress = min(1.0, next_action_counter / 150)
                s.tfreq = max(1, 5 - int(progress * 4))
            if next_action_counter % s.tfreq == 0 and s._wd:
                s._maybe_train(max_steps=1)
            return s._finalize_action(
                sel,
                reasoning,
                tensor=tensor,
                raw=raw,
                frame_hash=ch,
                action_idx=action_idx,
                remember_recent=True,
            )

        except Exception as e:
            traceback.print_exc()
            try:
                raw = s._raw(lf)
                ch = s._fast_frame_hash(raw)
                tensor = s._tensor(lf)
            except Exception:
                raw = None
                ch = None
                tensor = None
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = s._available_action_ids(avail)
            blocked_dir = s._blocked_direction_action_index(raw) if raw is not None else None
            if raw is not None:
                semantic_dir=s._semantic_direction_action(raw, avail)
                if semantic_dir is not None:
                    aid=int(semantic_dir[0]) + 1
                    a = s._fresh_action(aid)
                    s._refresh_semantic_target_coord(raw)
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=aid - 1,
                        remember_recent=True,
                    )
                if 6 in avail_ids:
                    semantic_clicks=s._semantic_click_targets_compat(raw, limit=1)
                    if semantic_clicks:
                        coords=semantic_clicks[0]
                        a = s._click_action(coords)
                        s._refresh_semantic_target_coord(raw, fallback_coord=coords)
                        return s._finalize_action(
                            a,
                            f"err:{e}",
                            tensor=tensor,
                            raw=raw,
                            frame_hash=ch,
                            action_idx=s._click_action_index(coords),
                            remember_recent=True,
                        )
            deferred_direction=None
            for aid in avail_ids:
                if 1 <= aid <= 5:
                    if aid <= 4 and blocked_dir is not None and (aid - 1) == blocked_dir:
                        if deferred_direction is None:
                            deferred_direction = aid
                        continue
                    a = s._fresh_action(aid)
                    if raw is not None:
                        s._refresh_semantic_target_coord(raw)
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=aid - 1,
                        remember_recent=raw is not None,
                    )
                if aid == 6:
                    coords=(32, 32)
                    used_semantic_click=False
                    blocked_click_coord=None
                    if raw is not None:
                        blocked_click_coord=s._blocked_click_coord(raw)
                        semantic_clicks=s._semantic_click_targets_compat(
                            raw,
                            limit=1,
                            blocked_click_coord=blocked_click_coord,
                        )
                        if semantic_clicks:
                            coords=semantic_clicks[0]
                            used_semantic_click=True
                        else:
                            if s._blocked_click_matches_coord(
                                    raw,
                                    coords,
                                    blocked_click_coord=blocked_click_coord):
                                for dy,dx in ((0,3), (3,0), (0,-3), (-3,0), (3,3), (-3,3), (3,-3), (-3,-3)):
                                    candidate=(max(0, min(s.G-1, coords[0] + dy)),
                                               max(0, min(s.G-1, coords[1] + dx)))
                                    if not s._blocked_click_matches_coord(
                                            raw,
                                            candidate,
                                            blocked_click_coord=blocked_click_coord):
                                        coords=candidate
                                        break
                    a = s._click_action(coords)
                    if raw is not None:
                        s._refresh_semantic_target_coord(
                            raw,
                            fallback_coord=coords if used_semantic_click else None,
                            blocked_click_coord=blocked_click_coord if 'blocked_click_coord' in locals() else None,
                        )
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=s._click_action_index(coords),
                        remember_recent=raw is not None,
                    )
                if aid == 7:
                    return s._finalize_control_action(
                        7,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        remember_recent=raw is not None,
                    )
            if deferred_direction is not None:
                a = s._fresh_action(deferred_direction)
                if raw is not None:
                    s._refresh_semantic_target_coord(raw)
                return s._finalize_action(
                    a,
                    f"err:{e}",
                    tensor=tensor,
                    raw=raw,
                    frame_hash=ch,
                    action_idx=deferred_direction - 1,
                    remember_recent=raw is not None,
                )
            return s._finalize_control_action(
                8,
                f"err:{e}",
                clear_recent=True,
            )
