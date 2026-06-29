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
import math
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
from itertools import islice

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
        s._framework_drives_action_counter=False
        # Replay stores compact uint8 frames plus parallel scalar arrays.  The
        # prior dict/int64 representation could exceed 1.6 GB at capacity.
        s.buf=[]; s.buf_actions=array('H'); s.buf_rewards=array('f'); s.buf_next_frames=[]; s.buf_has_next=array('b'); s.buf_priorities=array('f'); s.buf_keys=[]; s.buf_hashes=array('I'); s.buf_key_counts={}; s.buf_max=50000; s.buf_pos=0; s.buf_h=set()
        s._replay_buffer_version=0; s._replay_numeric_views_sig=None; s._replay_numeric_views_cache=None
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
        s._blocked_click_history=deque(maxlen=3)
        s._blocked_direction_history=deque(maxlen=3)
        s._eps=0.15; s._eps_min=0.02; s._eps_decay=0.9997; s._eps_steps=0
        s._prev_objs=None; s._obj_moved=0
        # FIX 1: Initialize _visited_hashes so _reward() deduplication works correctly
        s._visited_hashes = set()
        # Count-based intrinsic exploration bonus tracking
        s._state_visit_counts = {}
        # _tensor() static frame cache: avoids re-encoding 21 channels when frame unchanged
        s._tensor_last_frame_hash = None
        s._tensor_cached_static = None
        s._tensor_cached_full = None
        s._tensor_zero_tail_cache = {}
        # _replay_batch_tensor frame feature cache: avoids recomputing one-hot/edge/rarity
        s._frame_feature_cache = {}
        s._frame_feature_cache_max = 500
        s._replay_pos_cache = {}
        s._replay_zero_tail_cache = {}
        s._replay_tail_cache = {}
        s._legal_action_mask_cache = {}
        s._legal_direction_ids_cache = {}
        s._availability_summary_cache = {}
        s._bfs_priority_bonus_cache = {}
        # Semantic analysis caches: choose_action may query the same frame
        # several times through target ranking, click priors, and rescoring.
        s._semantic_components_cache_key=None
        s._semantic_components_cache_value=None
        s._semantic_detector_grid_cache_key=None
        s._semantic_detector_grid_cache_value=None
        s._semantic_target_candidates_cache_key=None
        s._semantic_target_candidates_cache_value=None
        s._semantic_click_targets_cache_key=None
        s._semantic_click_targets_cache_value=None
        s._semantic_click_candidate_indices_cache_key=None
        s._semantic_click_candidate_indices_cache_value=None
        s._heuristic_click_fallback_cache_key=None
        s._heuristic_click_fallback_cache_value=None
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

    def _normalized_palette_frame(s, frame):
        """Return a contiguous uint8 frame whose palette stays within 0..15."""
        frame=np.ascontiguousarray(frame, dtype=np.uint8)
        invalid=frame > 15
        if invalid.any():
            frame=frame.copy()
            frame[invalid]=0
        return frame

    def _replay_snapshot_frame(s, frame):
        """Store replay snapshots already normalized for later hot-path reuse."""
        normalized=s._normalized_palette_frame(frame)
        if normalized is frame:
            return normalized.copy()
        return normalized

    def _sanitize_frame_batch(s, frames_np):
        """Normalize a stacked replay batch only when a caller bypassed _add_replay."""
        if frames_np.dtype != np.uint8:
            frames_np=frames_np.astype(np.uint8, copy=False)
        invalid=frames_np > 15
        if invalid.any():
            frames_np=frames_np.copy()
            frames_np[invalid]=0
        return frames_np

    def _priority_from_reward(s, reward):
        """Convert a reward-like scalar into a finite positive replay priority."""
        try:
            reward=float(reward)
        except Exception:
            reward=0.0
        if not math.isfinite(reward):
            reward=0.0
        return max(abs(reward)+0.01, 0.01)

    def _sanitize_priority(s, priority, default=1.0):
        """Return a finite positive replay priority, falling back when needed."""
        try:
            priority=float(priority)
        except Exception:
            return float(default)
        if not math.isfinite(priority) or priority <= 0.0:
            return float(default)
        return priority

    def _packed_array_view(s, values, dtype, count=None):
        """Return a NumPy view over packed replay arrays when possible."""
        try:
            if count is None:
                return np.frombuffer(values, dtype=dtype)
            return np.frombuffer(values, dtype=dtype, count=count)
        except (TypeError, ValueError):
            if count is None:
                return np.asarray(values, dtype=dtype)
            return np.asarray(values[:count], dtype=dtype)

    def _bump_replay_buffer_version(s):
        """Invalidate cached NumPy views when replay arrays may have reallocated."""
        s._replay_buffer_version += 1
        s._replay_numeric_views_sig = None
        s._replay_numeric_views_cache = None

    def _release_replay_numeric_views(s):
        """Drop live NumPy buffer views before mutating packed replay arrays."""
        s._replay_numeric_views_sig = None
        s._replay_numeric_views_cache = None

    def _replay_numeric_views(s, count):
        """Reuse NumPy views over packed replay arrays until buffer shape changes."""
        sig=(s._replay_buffer_version, int(count))
        if s._replay_numeric_views_sig == sig and s._replay_numeric_views_cache is not None:
            return s._replay_numeric_views_cache
        views=(
            s._packed_array_view(s.buf_actions, np.uint16, count=count),
            s._packed_array_view(s.buf_rewards, np.float32, count=count),
            s._packed_array_view(s.buf_has_next, np.int8, count=count),
            s._packed_array_view(s.buf_priorities, np.float32, count=count),
        )
        s._replay_numeric_views_sig=sig
        s._replay_numeric_views_cache=views
        return views

    def _sampling_probabilities(s, n):
        """Return stable PER sampling probabilities for the current replay buffer."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if s.buf_priorities:
            _, _, _, priorities_view=s._replay_numeric_views(n)
            priorities=priorities_view
            if priorities.shape[0] < n:
                pad=np.full(n - priorities.shape[0], 1.0, dtype=np.float32)
                priorities=np.concatenate((priorities, pad), axis=0)
        else:
            priorities=np.ones(n, dtype=np.float32)
        invalid=(~np.isfinite(priorities)) | (priorities <= 0.0)
        if invalid.any():
            priorities=priorities.copy()
            priorities[invalid]=np.float32(1.0)
        probs=np.power(priorities, s._per_alpha, dtype=np.float32)
        total=float(probs.sum(dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            return np.full(n, 1.0 / float(n), dtype=np.float32)
        probs/=total
        if not np.all(np.isfinite(probs)):
            return np.full(n, 1.0 / float(n), dtype=np.float32)
        return probs

    def _update_sampled_priorities(s, indices, td_error):
        """Write sampled TD-error priorities back to the packed replay buffer."""
        if len(indices) == 0:
            return
        priorities_view=s._packed_array_view(s.buf_priorities, np.float32)
        td_error_np=np.asarray(td_error, dtype=np.float32)
        priority_values=np.abs(td_error_np, dtype=np.float32) + np.float32(0.01)
        invalid=(~np.isfinite(priority_values)) | (priority_values <= 0.0)
        if invalid.any():
            priority_values=priority_values.copy()
            priority_values[invalid]=np.float32(0.01)
        priorities_view[np.asarray(indices, dtype=np.int64)]=priority_values

    def _replay_pos_aug_batch(s, batch_size, like_tensor):
        """Reuse expanded positional channels for replay batches of the same size."""
        key=(like_tensor.device.type, like_tensor.device.index, int(batch_size))
        cached=s._replay_pos_cache.get(key)
        if cached is None or cached.device != like_tensor.device:
            if s._pos_aug_device is None or s._pos_aug_device.device!=like_tensor.device:
                s._pos_aug_device=s._pos_aug.to(like_tensor.device)
            cached=s._pos_aug_device.unsqueeze(0).expand(batch_size,-1,-1,-1)
            s._replay_pos_cache[key]=cached
        return cached

    def _replay_zero_tail_batch(s, batch_size, like_tensor):
        """Reuse zeroed dynamic-history channels for replay batches of the same size."""
        key=(like_tensor.device.type, like_tensor.device.index, int(batch_size), str(like_tensor.dtype))
        cached=s._replay_zero_tail_cache.get(key)
        if cached is None or cached.device != like_tensor.device or cached.dtype != like_tensor.dtype:
            cached=torch.zeros((batch_size,5,64,64), dtype=like_tensor.dtype, device=like_tensor.device)
            if like_tensor.device.type=='cuda':
                cached=cached.contiguous(memory_format=torch.channels_last)
            s._replay_zero_tail_cache[key]=cached
        return cached

    def _replay_tail_batch(s, batch_size, like_tensor):
        """Reuse the fixed 7 replay tail channels for a batch size/device/dtype."""
        key=(like_tensor.device.type, like_tensor.device.index, int(batch_size), str(like_tensor.dtype))
        cached=s._replay_tail_cache.get(key)
        if cached is None or cached.device != like_tensor.device or cached.dtype != like_tensor.dtype:
            pos=s._replay_pos_aug_batch(batch_size, like_tensor)
            zeros=s._replay_zero_tail_batch(batch_size, like_tensor)
            cached=torch.cat([pos, zeros], dim=1)
            if like_tensor.device.type=='cuda':
                cached=cached.contiguous(memory_format=torch.channels_last)
            s._replay_tail_cache[key]=cached
        return cached

    def _tensor_zero_tail(s, like_tensor):
        """Reuse the fixed 5 zero history channels for single-frame encoding."""
        key=(like_tensor.device.type, like_tensor.device.index, str(like_tensor.dtype))
        cached=s._tensor_zero_tail_cache.get(key)
        if cached is None or cached.device != like_tensor.device or cached.dtype != like_tensor.dtype:
            cached=torch.zeros((5,64,64), dtype=like_tensor.dtype, device=like_tensor.device)
            s._tensor_zero_tail_cache[key]=cached
        return cached

    def _pack_replay_feature_channels(s, oh, bg_m, rarity, edge):
        """Pack replay feature channels into one cached tensor."""
        return torch.cat([oh, bg_m, rarity, edge], dim=1)

    def _cached_replay_features(s, cached):
        """Support both legacy tuple cache entries and packed feature tensors."""
        if cached is None:
            return None
        if torch.is_tensor(cached):
            return cached
        oh_i,bg_i,ra_i,ed_i=cached
        packed=s._pack_replay_feature_channels(oh_i, bg_i, ra_i, ed_i)
        return packed

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
        return abs(coord_a[0] - coord_b[0]) + abs(coord_a[1] - coord_b[1])

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
        if s._semantic_target_coord is None or s._semantic_continuity_scale() <= 0.0:
            return None
        return (int(s._semantic_target_coord[0]), int(s._semantic_target_coord[1]))

    def _nearest_coord_within(s, coords, preferred_coord, max_distance):
        """Return the nearest `(y, x)` coord within a Manhattan distance threshold."""
        nearest_coord=None
        nearest_distance=None
        for coord in coords:
            dist=(abs(coord[0] - preferred_coord[0]) +
                  abs(coord[1] - preferred_coord[1]))
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance=dist
                nearest_coord=coord
        if nearest_coord is not None and nearest_distance is not None and nearest_distance <= max_distance:
            return nearest_coord
        return None

    def _prepend_nearest_preferred_coord(s, frame, candidates, coords, preferred_coord, seen, limit,
                                         blocked_click_coord=None):
        """Seed `coords` with the nearest preferred click candidate when it is nearby."""
        if preferred_coord is None:
            return False
        if s._semantic_continuity_scale() <= 0.5:
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
        if data:
            cache_key=(id(s._bfs), int(act_id), int(data.get("x", -1)), int(data.get("y", -1)))
        else:
            cache_key=(id(s._bfs), int(act_id), None, None)
        cached=s._bfs_priority_bonus_cache.get(cache_key)
        if cached is not None:
            return cached
        bfs_key=s._bfs._action_key(act_id, data)
        bonus=s._bfs._action_priority.get(bfs_key, 0) * 0.25
        s._bfs_priority_bonus_cache[cache_key]=bonus
        return bonus

    def _bfs_click_priority_bonus(s, click_coord):
        """Return the BFS-derived tie-break bonus for a click coordinate."""
        if s._bfs is None:
            return 0.0
        y,x=click_coord
        cache_key=(id(s._bfs), 6, int(x), int(y))
        cached=s._bfs_priority_bonus_cache.get(cache_key)
        if cached is not None:
            return cached
        data={"x": int(x), "y": int(y)}
        bfs_key=s._bfs._action_key(6, data)
        bonus=s._bfs._action_priority.get(bfs_key, 0) * 0.25
        s._bfs_priority_bonus_cache[cache_key]=bonus
        return bonus

    def _preferred_click_bonus(s, click_coord, preferred_click_coord):
        """Return the continuity bonus for clicks near the preferred semantic target."""
        if preferred_click_coord is None:
            return 0.0
        continuity_scale=s._semantic_continuity_scale()
        if continuity_scale <= 0.0:
            return 0.0
        click_pref_dist=s._click_coord_distance(click_coord, preferred_click_coord)
        if click_pref_dist == 0:
            return 0.08 * continuity_scale
        if click_pref_dist <= 2:
            return 0.04 * continuity_scale
        return 0.0

    def _preferred_direction_choice(s, preferred_dir, blocked, legal_action_ids):
        """Return the preferred direction index when it is still legal and unblocked."""
        if preferred_dir is None:
            return None
        preferred_action_id=preferred_dir + 1
        if preferred_dir == blocked or preferred_action_id not in legal_action_ids:
            return None
        return preferred_dir, None

    def _opposite_direction_index(s, direction_idx):
        """Return the opposite 0-based directional action index when available."""
        if direction_idx is None:
            return None
        return {0: 1, 1: 0, 2: 3, 3: 2}.get(int(direction_idx))

    def _preferred_click_target_choice(s, targets, preferred_coord, step):
        """Choose a click target by preferred continuity, then by step offset."""
        if preferred_coord is not None and s._semantic_continuity_scale() > 0.5:
            if preferred_coord in targets:
                return preferred_coord
            nearest_target=s._nearest_coord_within(targets, preferred_coord, 2)
            if nearest_target is not None:
                return nearest_target
        pidx=step-4
        if 0 <= pidx < len(targets):
            return targets[pidx]
        return None

    def _semantic_continuity_scale(s):
        """Decay sticky target continuity after repeated non-progress steps."""
        if s._unproductive >= 8:
            return 0.0
        if s._unproductive >= 6:
            return 0.35
        return 1.0

    def _preferred_click_continuity_active(s):
        """Return True while target continuity is still strong enough to bias candidate admission."""
        return s._preferred_click_coord() is not None and s._semantic_continuity_scale() > 0.5

    def _stale_wait_recovery(s, frame):
        """Return True when a recent ACTION5 already failed to change the scene."""
        return (
            s._unproductive >= 7 and
            s.pai is not None and
            int(s.pai) == 4 and
            s._frame_matches_previous(frame)
        )

    def _wait_recovery_bonus(s, frame, avail_ids, blocked_click_coord=None, frame_hash=None,
                             avail_summary=None):
        """Prefer ACTION5 when the agent is stuck and all semantic frontiers are exhausted."""
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids or ())
        if s._unproductive < 6 or not avail_summary["has_modeled"] or 5 not in (avail_ids or ()) or s._stale_wait_recovery(frame):
            return 0.0
        blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        for aid in avail_summary["legal_dirs"]:
            if not s._direction_matches_blocked_history(
                    aid - 1,
                    frame,
                    frame_hash=frame_hash,
                    blocked_direction=blocked_direction):
                return 0.0
        if avail_summary["has_click"]:
            semantic_clicks=s._semantic_click_targets_compat(
                frame,
                limit=2,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            if semantic_clicks:
                return 0.0
            fallback_clicks=s._heuristic_click_fallback_targets(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            if fallback_clicks:
                return 0.0
        return 0.3

    def _modeled_frontier_exhausted(s, frame, avail_ids, blocked_click_coord=None, frame_hash=None,
                                    avail_summary=None):
        """Return True when every modeled movement/click frontier is currently exhausted."""
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids or ())
        if 5 in avail_ids and not s._stale_wait_recovery(frame):
            return False
        blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        for aid in avail_summary["legal_dirs"]:
            if not s._direction_matches_blocked_history(
                    aid - 1,
                    frame,
                    frame_hash=frame_hash,
                    blocked_direction=blocked_direction):
                return False
        if not avail_summary["has_click"]:
            return True
        semantic_clicks=s._semantic_click_targets_compat(
            frame,
            limit=1,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if semantic_clicks:
            return False
        fallback_clicks=s._heuristic_click_fallback_targets(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if fallback_clicks:
            return False
        return True

    def _retry_blocked_direction_after_stale_wait(s, frame, avail_ids, blocked_click_coord=None, frame_hash=None,
                                                  avail_summary=None):
        """Return True when stale wait recovery should retry blocked directions as a last resort."""
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids or ())
        if not s._stale_wait_recovery(frame):
            return False
        legal_dirs=avail_summary["legal_dirs"]
        if not legal_dirs:
            return False
        blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        if any(not s._direction_matches_blocked_history(
                aid - 1,
                frame,
                frame_hash=frame_hash,
                blocked_direction=blocked_direction) for aid in legal_dirs):
            return False
        if avail_summary["has_click"]:
            semantic_clicks=s._semantic_click_targets_compat(
                frame,
                limit=1,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            if semantic_clicks:
                return False
            fallback_clicks=s._heuristic_click_fallback_targets(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            if fallback_clicks:
                return False
        return True

    def _should_exit_warmup_early(s, frame, avail_ids, blocked_click_coord=None, frame_hash=None,
                                  avail_summary=None):
        """Return True when heuristic warmup is clearly stuck and the learned policy should take over."""
        if s._unproductive < 6:
            return False
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids or ())
        if s._wait_recovery_bonus(
                frame,
                avail_ids,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                avail_summary=avail_summary) > 0.0:
            return True
        return s._modeled_frontier_exhausted(
            frame,
            avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )

    def _semantic_click_bonus_map(s, frame, limit, click_scale, click_targets=None):
        """Return ranked semantic click bonuses keyed by `(y, x)` coordinate."""
        bonuses={}
        if click_targets is None:
            click_targets=s._semantic_click_targets_compat(frame, limit=limit)
        for rank,(ty,tx) in enumerate(click_targets):
            bonuses[(int(ty), int(tx))]=max(0.0, 0.8 - 0.1 * rank) * click_scale
        return bonuses

    def _semantic_click_bonus(s, click_coord, click_scale, click_targets):
        """Return the ranked semantic click bonus for one `(y, x)` coordinate."""
        target_y=int(click_coord[0]); target_x=int(click_coord[1])
        for rank,(ty,tx) in enumerate(click_targets):
            if int(ty) == target_y and int(tx) == target_x:
                return max(0.0, 0.8 - 0.1 * rank) * click_scale
        return 0.0

    def _recent_frame_revisit_penalty(s, curr_h, prev_h):
        """Penalize short-horizon loops that bounce back into recent frames."""
        if not s.fhist:
            return 0.0
        recent_penalties=(0.35, 0.22, 0.12)
        for idx, frame in enumerate(islice(reversed(s.fhist), len(recent_penalties))):
            frame_h=s._fast_frame_hash(frame)
            if frame_h == prev_h:
                continue
            if frame_h == curr_h:
                return recent_penalties[idx]
        return 0.0

    def _recent_direction_progress_delta(s, frame, blocked_click_coord=None, frame_hash=None):
        """Return semantic target-distance improvement from the last effective direction."""
        def _baseline_goal_distance(goal_frame, goal_blocked_click_coord=None, goal_frame_hash=None):
            comps=s._semantic_components(goal_frame, frame_hash=goal_frame_hash)
            if not comps:
                return None
            player=None
            player_area=-1
            for key in ('4', '12'):
                for comp in comps.get(key) or ():
                    area=int(comp.get('cell_count', 0))
                    if area > player_area:
                        player=comp
                        player_area=area
            if player is None:
                return None
            center=player.get('center')
            if not center or len(center) != 2:
                return None
            py=float(center[0]); px=float(center[1])
            best=None
            blocked_click_known=goal_blocked_click_coord is not None
            for color, priority in ((14,0), (6,1), (11,2), (5,3), (9,4), (7,5), (13,6), (15,7)):
                for comp in comps.get(str(color)) or ():
                    tcenter=comp.get('center')
                    if not tcenter or len(tcenter) != 2:
                        continue
                    ty=float(tcenter[0]); tx=float(tcenter[1])
                    target_coord=(int(round(ty)), int(round(tx)))
                    if s._blocked_click_matches_coord(
                            goal_frame,
                            target_coord,
                            blocked_click_coord=goal_blocked_click_coord if blocked_click_known else None,
                            frame_hash=goal_frame_hash):
                        continue
                    dist=abs(ty-py)+abs(tx-px)
                    if dist < 1.0:
                        continue
                    area=int(comp.get('cell_count', 0))
                    if area <= 0 or area > 512:
                        continue
                    score_key=(priority, round(dist, 6), -area)
                    if best is None or score_key < best[0]:
                        best=(score_key, float(dist))
            if best is None:
                return None
            return best[1]

        if s.pr is None:
            return None
        recent_direction=s._recent_direction_action_index(frame, frame_hash=frame_hash)
        if recent_direction is None:
            return None
        prev_frame_hash=s.ph if s.ph is not None else s._fast_frame_hash(s.pr)
        prev_dist=_baseline_goal_distance(s.pr, goal_frame_hash=prev_frame_hash)
        curr_dist=_baseline_goal_distance(
            frame,
            goal_blocked_click_coord=blocked_click_coord,
            goal_frame_hash=frame_hash,
        )
        if prev_dist is None or curr_dist is None:
            return None
        return float(prev_dist - curr_dist)

    def _semantic_direct_click_choice(s, frame, avail=None, avail_ids=None,
                                      blocked_click_coord=None, frame_hash=None):
        """Commit to ACTION6 when a top semantic target is already directly clickable."""
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        if 6 not in (avail_ids or ()):
            return None
        if frame_hash is None:
            frame_hash=s._fast_frame_hash(frame)
        preferred_click_coord=s._preferred_click_coord()
        prefer_continuity_click=s._preferred_click_continuity_active()
        target_choice=s._semantic_target_choice(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if not target_choice:
            comps=s._semantic_components(frame, frame_hash=frame_hash) or {}
            has_player=bool((comps.get('4') or []) or (comps.get('12') or []))
            if not has_player:
                semantic_clicks=s._semantic_click_targets_compat(
                    frame,
                    limit=6,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash,
                )
                fallback_clicks=s._heuristic_click_fallback_targets(
                    frame,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash,
                )
                direct_click_candidates=list(semantic_clicks)
                for coord in fallback_clicks:
                    if coord not in direct_click_candidates:
                        direct_click_candidates.append(coord)
                if direct_click_candidates:
                    direct_click_coord=None
                    if (prefer_continuity_click and
                            not s._blocked_click_matches_coord(
                                frame,
                                preferred_click_coord,
                                blocked_click_coord=blocked_click_coord,
                                frame_hash=frame_hash,
                            )):
                        direct_click_coord=s._nearest_coord_within(direct_click_candidates, preferred_click_coord, 2)
                        if direct_click_coord is None:
                            direct_click_coord=preferred_click_coord
                    if direct_click_coord is None:
                        direct_click_coord=direct_click_candidates[0]
                    return 5, direct_click_coord
            return None
        if int(target_choice.get('priority', 99)) > 1 or float(target_choice.get('distance', 999.0)) > 2.5:
            return None
        target_coord=(
            int(round(target_choice['target_y'])),
            int(round(target_choice['target_x'])),
        )
        candidate_coords=[]
        if (prefer_continuity_click and
                not s._blocked_click_matches_coord(
                    frame,
                    preferred_click_coord,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash,
                )):
            candidate_coords.append(preferred_click_coord)
        candidate_coords.extend(s._semantic_click_targets_compat(
            frame,
            limit=3,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ))
        seen=set()
        for coord in candidate_coords:
            coord=(int(coord[0]), int(coord[1]))
            if coord in seen:
                continue
            seen.add(coord)
            if s._click_coord_distance(coord, target_coord) <= 2:
                return 5, coord
        return None

    def _count_action(s):
        if not s._framework_drives_action_counter:
            s.action_counter += 1

    def main(s, *args, **kwargs):
        """Let the framework own `action_counter` during real gameplay loops."""
        s._framework_drives_action_counter=True
        try:
            return super().main(*args, **kwargs)
        finally:
            s._framework_drives_action_counter=False

    def _clear_recent_action_state(s):
        """Drop the previous-frame/action cache used for reward shaping."""
        s.pt = None
        s.pai = None
        s.pr = None
        s.ph = None
        s._previous_frame_relation_cache = None
        if hasattr(s, '_blocked_click_history'):
            s._blocked_click_history.clear()
        if hasattr(s, '_blocked_direction_history'):
            s._blocked_direction_history.clear()

    def _remember_blocked_click_coord(s, coord):
        """Remember a blocked click region so future click ranking avoids dead ends."""
        if coord is None:
            return
        coord=(int(coord[0]), int(coord[1]))
        for seen in tuple(s._blocked_click_history):
            if s._coord_matches_blocked_click(coord, seen):
                s._blocked_click_history.remove(seen)
                s._blocked_click_history.append(coord)
                return
        s._blocked_click_history.append(coord)

    def _clear_blocked_click_history(s):
        """Forget stale blocked click regions after the scene changes."""
        s._blocked_click_history.clear()

    def _decay_blocked_click_history(s):
        """Age out only the oldest blocked click region after real progress."""
        if s._blocked_click_history:
            s._blocked_click_history.popleft()

    def _blocked_click_history_signature(s):
        """Return a stable cache-key signature for remembered blocked click regions."""
        return tuple((int(coord[0]), int(coord[1])) for coord in s._blocked_click_history)

    def _remember_blocked_direction_index(s, direction_idx):
        """Remember a blocked directional move to avoid short-horizon ping-pong."""
        if direction_idx is None or not (0 <= int(direction_idx) < 4):
            return
        direction_idx=int(direction_idx)
        if direction_idx in s._blocked_direction_history:
            s._blocked_direction_history.remove(direction_idx)
            s._blocked_direction_history.append(direction_idx)
            return
        s._blocked_direction_history.append(direction_idx)

    def _clear_blocked_direction_history(s):
        """Forget stale blocked directions after the scene changes."""
        s._blocked_direction_history.clear()

    def _decay_blocked_direction_history(s):
        """Age out only the oldest blocked direction after real progress."""
        if s._blocked_direction_history:
            s._blocked_direction_history.popleft()

    def _direction_matches_blocked_history(s, direction_idx, frame=None, frame_hash=None,
                                           blocked_direction=None):
        """Return True when a direction is blocked now or was blocked very recently."""
        if direction_idx is None or not (0 <= int(direction_idx) < 4):
            return False
        direction_idx=int(direction_idx)
        if blocked_direction is None and frame is not None:
            blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        if blocked_direction is not None and int(blocked_direction) == direction_idx:
            return True
        return direction_idx in s._blocked_direction_history

    def _snapshot_frame(s, raw):
        """Return an owned snapshot for history and previous-frame bookkeeping."""
        return raw.copy()

    def _remember_recent_action(s, tensor, raw, frame_hash, action_idx, raw_snapshot=None):
        """Store the current observation and chosen action for the next step."""
        s.pt = tensor
        s.pai = action_idx
        s.pr = raw_snapshot if raw_snapshot is not None else s._snapshot_frame(raw)
        s.ph = frame_hash
        s._previous_frame_relation_cache = None
        s.la += 1

    def _finalize_action(s, action, reasoning, *, tensor=None, raw=None, frame_hash=None,
                         action_idx=None, remember_recent=False, clear_recent=False,
                         raw_snapshot=None):
        """Attach reasoning and finish an action return with consistent bookkeeping."""
        action.reasoning = reasoning
        if clear_recent:
            s._clear_recent_action_state()
        elif remember_recent:
            s._remember_recent_action(tensor, raw, frame_hash, action_idx, raw_snapshot=raw_snapshot)
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
            raw_snapshot=None,
        )

    def _reset_level_runtime_state(s, lvl):
        """Reset per-level caches and counters while keeping learned network state."""
        s.net.eval()
        s._clear_recent_action_state()
        s._semantic_target_coord=None
        s.cl=lvl
        s.fhist.clear()
        s.la=0
        s._wd=False
        s._wm=None
        s._wm_dev=None
        s._wm_log_dev=None
        s._wm_cache_key=None
        s._bfs_priority_bonus_cache.clear()
        s._aem_cache_sig=None
        s._aem_cache=(None,None,None)
        s._aem_encoded_cache_sig=None
        s._aem_encoded_cache=None
        s._aem_diffs.clear()
        s._aem_actions.clear()
        s._aem_rewards.clear()
        s._prev_objs=None
        s._obj_moved=0
        s._ckpt_hash=None
        s._unproductive=0
        s._visited_hashes = set()
        s._state_visit_counts = {}
        if not s._bfs_solution:
            s._eps = 0.15
            s._eps_steps = 0

    def _ensure_click_template(s, raw):
        """Populate the click heatmap lazily for the current level."""
        if s._wm is None:
            s._wm=s._detect_template(raw)

    def _handle_non_modeled_availability(s, tensor, raw, frame_hash):
        """Return a control action when only non-modeled actions are currently legal."""
        if not s._undo_avail:
            return s._finalize_control_action(
                GameAction.RESET.value if hasattr(GameAction.RESET, "value") else int(GameAction.RESET),
                "no-action",
                clear_recent=True,
            )
        return s._finalize_control_action(
            7,
            "undo-only",
            tensor=tensor,
            raw=raw,
            frame_hash=frame_hash,
            remember_recent=True,
        )

    def _maybe_force_undo(s, tensor, raw, frame_hash):
        """Return UNDO after a long unproductive streak when it is legal."""
        if not (s._undo_avail and s._ckpt_hash):
            return None
        prev_h=s.ph if s.ph is not None else (s._fast_frame_hash(s.pr) if s.pr is not None else None)
        loop_revisit=(prev_h is not None and s._recent_frame_revisit_penalty(frame_hash, prev_h) > 0.0)
        if not (s._unproductive >= 30 or loop_revisit):
            return None
        s._unproductive=0
        return s._finalize_control_action(
            7,
            "undo",
            tensor=tensor,
            raw=raw,
            frame_hash=frame_hash,
            remember_recent=True,
        )

    def _prime_warmup_action(s, raw, avail, frame_hash=None):
        """Run heuristic warmup before the learned policy takes over."""
        blocked_click_coord=s._blocked_click_coord(raw, frame_hash=frame_hash)
        avail_ids=s._available_action_ids(avail)
        avail_summary=s._availability_summary(avail_ids)
        if s.la < 10 and not s._should_exit_warmup_early(
                raw,
                avail_ids,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                avail_summary=avail_summary):
            return s._heuristic(
                raw,
                avail,
                s.la,
                blocked_click_coord=blocked_click_coord,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )
        s._wd=True
        s._maybe_train(max_steps=min(2, len(s.buf) // s.bsz), force=True)
        return None

    def _choose_policy_action(s, tensor, raw, avail, avail_ids, blocked_click_coord, frame_hash=None):
        """Choose the next modeled action from heuristic, exploration, or CNN rescoring."""
        a6_avail=6 in avail_ids
        avail_summary=s._availability_summary(avail_ids)
        if s.net is None:
            return s._heuristic(
                raw,
                avail,
                s.la,
                blocked_click_coord=blocked_click_coord,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )
        direct_click_choice=s._semantic_direct_click_choice(
            raw,
            avail,
            avail_ids=avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if direct_click_choice is not None:
            return direct_click_choice
        if random.random() < s._eps:
            prior_logits=s._semantic_exploration_logits(
                raw,
                avail,
                a6_avail,
                blocked_click_coord=blocked_click_coord,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )
            return s._sample_semantic_exploration(
                prior_logits,
                raw,
                avail,
                avail_ids=avail_ids,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                temp=1.25,
            )
        with torch.inference_mode():
            with s._amp_context():
                net_input=tensor.unsqueeze(0)
                if s.device.type == 'cuda':
                    net_input=net_input.contiguous(memory_format=torch.channels_last)
                mem=s._get_aem_tensors()
                encoded=s._get_aem_encoded(mem) if mem[0] is not None else None
                if a6_avail:
                    if mem[0] is not None:
                        logits=s.net(net_input,*mem,mem_encoded=encoded).squeeze(0)
                    else:
                        logits=s.net(net_input).squeeze(0)
                else:
                    if mem[0] is not None:
                        logits=s.net.forward_actions(net_input,*mem,mem_encoded=encoded).squeeze(0)
                    else:
                        logits=s.net.forward_actions(net_input).squeeze(0)
        aidx,coords=None,None
        try:
            K=5
            avail_mask=s._legal_action_mask(logits, avail, avail_ids=avail_ids)
            semantic_click_targets=[]
            semantic_click_bonus_map={}
            semantic_dirs=s._semantic_direction_bonuses(raw, avail, avail_ids=avail_ids, frame_hash=frame_hash)
            blocked_direction_idx=s._blocked_direction_action_index(raw, frame_hash=frame_hash)
            repeat_direction_bonus_idx=s._recent_direction_action_index(raw, frame_hash=frame_hash)
            repeat_click_bonus_idx=s._recent_click_action_index(raw, frame_hash=frame_hash)
            preferred_click_coord=s._preferred_click_coord()
            prefer_continuity_click=s._preferred_click_continuity_active()
            target_choice=s._semantic_target_choice(
                raw,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            click_scale=s._semantic_click_bonus_scale(
                raw,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                target_choice=target_choice,
            )
            retry_blocked_direction=s._retry_blocked_direction_after_stale_wait(
                raw,
                avail_ids,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )
            blocked_click_idx=s._blocked_click_action_index(raw, frame_hash=frame_hash)
            if a6_avail:
                semantic_click_targets=s._semantic_click_targets_compat(
                    raw,
                    limit=6,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash,
                )
                semantic_click_bonus_map=s._semantic_click_bonus_map(
                    raw,
                    limit=6,
                    click_scale=click_scale,
                    click_targets=semantic_click_targets,
                )
            n_valid=s._legal_modeled_action_count(len(logits), avail_ids)
            if n_valid > 0:
                scored=logits + avail_mask
                best_k=scored.topk(min(K, n_valid))
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
                        blocked_click_coord=blocked_click_coord,
                        avail_ids=avail_ids,
                        frame_hash=frame_hash):
                    s._append_candidate_index(
                        candidate_indices,
                        candidate_seen,
                        idx,
                    )
                if (a6_avail and
                        prefer_continuity_click and
                        not s._blocked_click_matches_coord(
                            raw,
                            preferred_click_coord,
                            blocked_click_coord=blocked_click_coord,
                            frame_hash=frame_hash)):
                    preferred_click_idx=s._click_action_index(preferred_click_coord)
                    s._append_candidate_index(
                        candidate_indices,
                        candidate_seen,
                        preferred_click_idx,
                    )
                candidate_score_map=s._candidate_score_map(scored, candidate_indices)
                best_local=0
                best_score=float('-inf')
                for i,top_idx in enumerate(candidate_indices):
                    score=candidate_score_map.get(int(top_idx), float('-inf'))
                    if top_idx < 5:
                        if (top_idx == 4 and retry_blocked_direction):
                            score=float('-inf')
                        elif (not retry_blocked_direction and
                                s._direction_matches_blocked_history(
                                    top_idx,
                                    raw,
                                    frame_hash=frame_hash,
                                    blocked_direction=blocked_direction_idx)):
                            score=float('-inf')
                        else:
                            score += semantic_dirs.get(top_idx, 0.0)
                            if top_idx == 4:
                                score += s._wait_recovery_bonus(
                                    raw,
                                    avail_ids,
                                    blocked_click_coord=blocked_click_coord,
                                    frame_hash=frame_hash,
                                    avail_summary=avail_summary,
                                )
                        if (repeat_direction_bonus_idx is not None and
                                top_idx == repeat_direction_bonus_idx and
                                (retry_blocked_direction or
                                 not s._direction_matches_blocked_history(
                                     top_idx,
                                     raw,
                                     frame_hash=frame_hash,
                                     blocked_direction=blocked_direction_idx))):
                            score += 0.08
                        score += s._bfs_priority_bonus(top_idx + 1)
                    else:
                        click_coord=s._click_coord_from_action_index(top_idx)
                        click_y,click_x=click_coord
                        blocked_click_match=s._blocked_click_matches_coord(
                            raw,
                            click_coord,
                            blocked_click_coord=blocked_click_coord,
                            frame_hash=frame_hash,
                        )
                        if blocked_click_match:
                            score=float('-inf')
                        else:
                            score += s._bfs_click_priority_bonus(click_coord)
                            if s._wm is not None:
                                score += float(s._wm[click_y, click_x]) * 0.05
                            score += semantic_click_bonus_map.get(click_coord, 0.0)
                            score += s._preferred_click_bonus(click_coord, preferred_click_coord)
                            if (repeat_click_bonus_idx is not None and
                                    top_idx == repeat_click_bonus_idx and
                                    not blocked_click_match):
                                score += 0.08
                            if blocked_click_idx is not None and top_idx == blocked_click_idx:
                                score=float('-inf')
                    if score > best_score:
                        best_score=score
                        best_local=i
                top_idx=int(candidate_indices[best_local])
                aidx,coords=s._decode_policy_action_index(top_idx)
        except Exception as e:
            logger.debug("CNN action rescoring unavailable: %s", e)
        if aidx is None:
            aidx,coords=s._sample(logits, avail, temp=0.5, avail_ids=avail_ids)
        return aidx,coords

    def _training_frequency_for_next_action(s, next_action_counter):
        """Return the train cadence for the next modeled action."""
        sol_exhausted=(s._bfs_step >= len(s._bfs_solution)) if s._bfs_solution else True
        if sol_exhausted:
            return 1
        progress=min(1.0, next_action_counter / 150)
        return max(1, 5 - int(progress * 4))

    def _finalize_modeled_action(s, aidx, coords, tensor, raw, frame_hash, blocked_click_coord):
        """Build, bookkeep, and optionally train after a modeled action choice."""
        if aidx < 5:
            sel=s._fresh_action(aidx + 1)
            reasoning=f"cnn:a{aidx+1}"
        else:
            y,x=coords
            sel=s._click_action((y, x))
            reasoning=f"cnn:c({x},{y})"
        s._refresh_semantic_target_coord(
            raw,
            fallback_coord=coords if aidx >= 5 else None,
            blocked_click_coord=blocked_click_coord,
        )
        action_idx=aidx if aidx < 5 else s._click_action_index(coords)
        next_action_counter=s.action_counter + 1
        s.tfreq=s._training_frequency_for_next_action(next_action_counter)
        if next_action_counter % s.tfreq == 0 and s._wd:
            s._maybe_train(max_steps=1)
        return s._finalize_action(
            sel,
            reasoning,
            tensor=tensor,
            raw=raw,
            frame_hash=frame_hash,
            action_idx=action_idx,
            remember_recent=True,
        )

    def _try_repeat_direction_action(s, raw, avail, avail_ids, tensor, frame_hash):
        """Repeat a recently successful directional action when semantics still agree."""
        if s.pai is None or not (0 <= s.pai < 4):
            return None
        if random.random() >= 0.4:
            return None
        prev_h=s.ph if s.ph is not None else (s._fast_frame_hash(s.pr) if s.pr is not None else None)
        if prev_h is not None and s._recent_frame_revisit_penalty(frame_hash, prev_h) > 0.0:
            return None
        repeat_id=s.pai + 1
        blocked_click_coord=s._blocked_click_coord(raw, frame_hash=frame_hash)
        recent_progress_delta=s._recent_direction_progress_delta(
            raw,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if recent_progress_delta is not None and recent_progress_delta < -0.5:
            return None
        preferred_click=s._preferred_click_coord()
        direct_click_choice=s._semantic_direct_click_choice(
            raw,
            avail,
            avail_ids=avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if direct_click_choice is not None:
            direct_click_coord=direct_click_choice[1]
            if preferred_click is None:
                return None
            direct_click_match_dist=s._click_coord_distance(direct_click_coord, preferred_click)
            if direct_click_match_dist == 0 or direct_click_match_dist > 2:
                return None
        semantic_dir=s._semantic_direction_action(raw, avail, avail_ids=avail_ids, frame_hash=frame_hash)
        click_avail=6 in avail_ids
        semantic_clicks=(
            s._semantic_click_targets_compat(
                raw,
                limit=1,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            if click_avail else []
        )
        click_matches_preferred=False
        click_exact_preferred=False
        if semantic_clicks and preferred_click is not None:
            click_match_dist=s._click_coord_distance(semantic_clicks[0], preferred_click)
            click_matches_preferred=click_match_dist <= 2
            click_exact_preferred=click_match_dist == 0
        click_blocks_repeat=(
            bool(semantic_clicks) and
            s._semantic_click_bonus_scale(
                raw,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            ) >= 0.5 and
            (click_exact_preferred or not click_matches_preferred)
        )
        semantic_repeat_ok=(
            (semantic_dir is not None and semantic_dir[0] == s.pai) or
            (semantic_dir is None and not click_blocks_repeat)
        )
        if not semantic_repeat_ok or repeat_id not in avail_ids:
            return None
        raw_snapshot=s._snapshot_frame(raw)
        s.fhist.append(raw_snapshot)
        s._refresh_semantic_target_coord(raw)
        return s._finalize_action(
            s._fresh_action(repeat_id),
            f"repeat:a{repeat_id}",
            tensor=tensor,
            raw=raw,
            frame_hash=frame_hash,
            action_idx=s.pai,
            remember_recent=True,
            raw_snapshot=raw_snapshot,
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
                s.buf_priorities[idx] = s._priority_from_reward(s.buf_rewards[idx])

    def _track_replay_dedup_key(s, dedup_key):
        """Record one replay reference for a dedup key."""
        if dedup_key is None:
            return
        s.buf_key_counts[dedup_key] = s.buf_key_counts.get(dedup_key, 0) + 1
        s.buf_h.add(dedup_key)

    def _untrack_replay_dedup_key(s, dedup_key):
        """Drop one replay reference for a dedup key."""
        if dedup_key is None:
            return
        remaining = s.buf_key_counts.get(dedup_key, 0) - 1
        if remaining > 0:
            s.buf_key_counts[dedup_key] = remaining
            return
        s.buf_key_counts.pop(dedup_key, None)
        s.buf_h.discard(dedup_key)

    def _rebuild_replay_dedup_state(s):
        """Rebuild replay dedup membership from the retained key list."""
        counts = {}
        for key in s.buf_keys:
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1
        s.buf_key_counts = counts
        s.buf_h = set(counts)

    def _clear_replay(s, keep_frac=0.2):
        """Clear replay buffer, optionally retaining top-K highest-reward transitions
        for cross-level transfer of learned action-effect patterns.
        Small buffers (< keep_frac threshold) are preserved intact so expert
        demonstrations (BFS solutions, CLTI, etc.) persist across level changes."""
        if keep_frac > 0 and len(s.buf) <= s.bsz:
            return
        if keep_frac > 0 and len(s.buf) > s.bsz:
            s._release_replay_numeric_views()
            n_keep = max(s.bsz, int(len(s.buf) * keep_frac))
            rewards = np.array(s.buf_rewards, dtype=np.float32)
            if len(rewards) > n_keep:
                top_idx = np.argsort(rewards)[-n_keep:]
                s.buf = [s.buf[i] for i in top_idx]
                s.buf_actions = array('H', [s.buf_actions[i] for i in top_idx])
                s.buf_rewards = array('f', [s.buf_rewards[i] for i in top_idx])
                if s.buf_next_frames:
                    s.buf_next_frames = [s.buf_next_frames[i] for i in top_idx]
                if s.buf_has_next:
                    s.buf_has_next = array('b', [s.buf_has_next[i] for i in top_idx])
                if s.buf_priorities:
                    s.buf_priorities = array('f', [s.buf_priorities[i] for i in top_idx])
                if s.buf_keys:
                    s.buf_keys = [s.buf_keys[i] for i in top_idx]
                if s.buf_hashes:
                    s.buf_hashes = array('I', [s.buf_hashes[i] for i in top_idx])
            else:
                return  # buffer small enough — keep all entries intact
            # Always reset dedup hash and position when entries are pruned
            s._bump_replay_buffer_version()
            s._rebuild_replay_dedup_state(); s.buf_pos = 0
            return
        # Full clear (keep_frac <= 0 or buffer empty)
        s._release_replay_numeric_views()
        s.buf.clear(); s.buf_actions=array('H'); s.buf_rewards=array('f')
        s.buf_next_frames.clear(); s.buf_has_next=array('b'); s.buf_priorities=array('f'); s.buf_keys.clear(); s.buf_hashes=array('I'); s.buf_key_counts={}; s.buf_h.clear(); s.buf_pos=0
        s._bump_replay_buffer_version()

    def _add_replay(s, frame, action_idx, reward, next_frame=None, dedup_key=None):
        """Append a compact transition without per-entry dict or int64 overhead."""
        s._release_replay_numeric_views()
        snapshot=s._replay_snapshot_frame(frame)
        next_snapshot=s._replay_snapshot_frame(next_frame) if next_frame is not None else None
        snapshot_hash=s._fast_frame_hash(snapshot)
        action_idx=max(0,min(65535,int(action_idx)))
        reward=float(reward)
        priority=s._priority_from_reward(reward)
        if len(s.buf) < s.buf_max:
            s.buf.append(snapshot)
            s.buf_actions.append(action_idx)
            s.buf_rewards.append(reward)
            s.buf_next_frames.append(next_snapshot)
            s.buf_has_next.append(1 if next_snapshot is not None else 0)
            s.buf_priorities.append(priority)
            s.buf_keys.append(dedup_key)
            s.buf_hashes.append(snapshot_hash)
            s._track_replay_dedup_key(dedup_key)
            s._boost_recent_replay_rewards(reward, len(s.buf_rewards) - 1)
            s._bump_replay_buffer_version()
        else:
            i=s.buf_pos
            old_key = s.buf_keys[i] if i < len(s.buf_keys) else None
            s._untrack_replay_dedup_key(old_key)
            s.buf[i]=snapshot
            s.buf_actions[i]=action_idx
            s.buf_rewards[i]=reward
            s.buf_next_frames[i]=next_snapshot
            s.buf_has_next[i]=1 if next_snapshot is not None else 0
            s.buf_priorities[i]=priority
            s.buf_keys[i]=dedup_key
            s.buf_hashes[i]=snapshot_hash
            s._track_replay_dedup_key(dedup_key)
            s._boost_recent_replay_rewards(reward, i)
            s.buf_pos=(i+1)%s.buf_max
            s._bump_replay_buffer_version()

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
        frame=s._normalized_palette_frame(frame)
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
        frame=s._normalized_palette_frame(frame)
        fh=s._fast_frame_hash(frame)
        if (fh == s._tensor_last_frame_hash and
                s._tensor_cached_static is not None and
                s._tensor_cached_full is not None and
                s._tensor_cached_full.device == s.device):
            return s._tensor_cached_full
        if fh == s._tensor_last_frame_hash and s._tensor_cached_static is not None:
            static = s._tensor_cached_static
        else:
            static = s._encode_static_frame_cpu(frame, update_bg=True)
            s._tensor_last_frame_hash = fh
            s._tensor_cached_static = static
        static_device=static.to(s.device,non_blocking=True)
        out=torch.cat([static_device, s._tensor_zero_tail(static_device)], dim=0)
        s._tensor_cached_full=out
        return out

    def _tensor(s, fd):
        frame=s._raw(fd)
        return s._encode_frame_tensor(frame)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        active=(frame!=s._bg)
        col_act=np.sum(active,axis=0)
        col_nonzero=(col_act>0).astype(np.int16, copy=False)
        col_prefix=np.cumsum(col_nonzero, dtype=np.int16)
        col_suffix=np.cumsum(col_nonzero[::-1], dtype=np.int16)[::-1]
        for c in range(20,44):
            if (col_act[c]<=2 and
                    (col_prefix[c-1] if c > 0 else 0) >= 5 and
                    (col_suffix[c+1] if c + 1 < col_suffix.shape[0] else 0) >= 5):
                mask.view(64,64)[:, :c+1] = 0.05
                return mask
        row_act=np.sum(active,axis=1)
        row_nonzero=(row_act>0).astype(np.int16, copy=False)
        row_prefix=np.cumsum(row_nonzero, dtype=np.int16)
        row_suffix=np.cumsum(row_nonzero[::-1], dtype=np.int16)[::-1]
        for r in range(20,44):
            if (row_act[r]<=2 and
                    (row_prefix[r-1] if r > 0 else 0) >= 5 and
                    (row_suffix[r+1] if r + 1 < row_suffix.shape[0] else 0) >= 5):
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
        r -= s._recent_frame_revisit_penalty(curr_h, prev_h)
        # Count-based intrinsic exploration bonus: rewards novel states
        count = s._state_visit_counts.get(curr_h, 0)
        s._state_visit_counts[curr_h] = count + 1
        r += 0.3 / (count ** 0.5 + 1)
        return r

    def _sample(s, logits, avail=None, temp=1.0, avail_ids=None):
        has_click_logits = logits.numel() >= 4101
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        a6=has_click_logits and ((not avail_ids) or 6 in avail_ids)
        total_len=4101 if a6 else 5
        allp=torch.zeros(total_len, dtype=logits.dtype, device=logits.device)
        dir_logits=logits[:5]
        legal_dir_indices=[]
        if avail_ids:
            for aid in avail_ids:
                if 1 <= aid <= 5:
                    idx=aid - 1
                    legal_dir_indices.append(idx)
                    logit=dir_logits[idx]
                    if torch.isfinite(logit):
                        allp[idx]=torch.sigmoid(logit / temp)
        else:
            legal_dir_indices=[0, 1, 2, 3, 4]
            dir_finite=torch.isfinite(dir_logits)
            allp[:5]=torch.where(dir_finite, torch.sigmoid(dir_logits / temp), torch.zeros_like(dir_logits))
        finite_click_mask=None
        if a6:
            click_logits=logits[5:5+4096]
            template_log_bias=s._template_log_bias()
            if template_log_bias is not None:
                click_logits=click_logits + template_log_bias
            finite_click_mask=torch.isfinite(click_logits)
            click_probs=torch.sigmoid(click_logits / temp) / (s.G * s.G)
            allp[5:]=torch.where(finite_click_mask, click_probs, torch.zeros_like(click_probs))
        sm=allp.sum()
        if sm<1e-8:
            allp.zero_()
            finite_dir_found=False
            for idx in legal_dir_indices:
                if torch.isfinite(dir_logits[idx]):
                    allp[idx]=1.0
                    finite_dir_found=True
            finite_click_found=bool(a6 and finite_click_mask is not None and torch.any(finite_click_mask))
            if finite_click_found:
                allp[5:]=finite_click_mask.to(dtype=allp.dtype)
            if finite_dir_found or finite_click_found:
                allp=allp / allp.sum()
            else:
                allp.fill_(1.0 / len(allp))
        else:
            allp=allp / sm
        idx=int(torch.multinomial(allp, 1).item())
        return s._decode_policy_action_index(idx)

    def _sample_sparse_policy_indices(s, logits, avail_ids, candidate_indices, temp=1.0):
        """Sample from legal directions plus a sparse subset of click indices."""
        allp=torch.zeros(4101, dtype=logits.dtype, device=logits.device)
        dir_logits=logits[:5]
        legal_dir_indices=[]
        for aid in (avail_ids or ()):
            if 1 <= aid <= 5:
                idx=aid - 1
                legal_dir_indices.append(idx)
                logit=dir_logits[idx]
                if torch.isfinite(logit):
                    allp[idx]=torch.sigmoid(logit / temp)
        if candidate_indices:
            template_log_bias=s._template_log_bias()
            for idx in candidate_indices:
                idx=int(idx)
                if idx < 5 or idx >= logits.numel():
                    continue
                logit=logits[idx]
                if template_log_bias is not None:
                    logit=logit + template_log_bias[idx - 5]
                if torch.isfinite(logit):
                    allp[idx]=torch.sigmoid(logit / temp) / (s.G * s.G)
        sm=allp.sum()
        if sm<1e-8:
            finite_dir_found=False
            for idx in legal_dir_indices:
                if torch.isfinite(dir_logits[idx]):
                    allp[idx]=1.0
                    finite_dir_found=True
            finite_click_found=False
            for idx in candidate_indices:
                idx=int(idx)
                if 5 <= idx < logits.numel() and torch.isfinite(logits[idx]):
                    allp[idx]=1.0
                    finite_click_found=True
            if finite_dir_found or finite_click_found:
                allp=allp / allp.sum()
            else:
                allp.fill_(1.0 / len(allp))
        else:
            allp=allp / sm
        idx=int(torch.multinomial(allp, 1).item())
        return s._decode_policy_action_index(idx)

    def _sample_semantic_exploration(s, logits, frame, avail, avail_ids=None,
                                     blocked_click_coord=None, frame_hash=None, temp=1.0):
        """Sample epsilon-exploration actions from sparse semantic click candidates."""
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        if logits.numel() < 4101 or 6 not in (avail_ids or ()):
            return s._sample(logits, avail=avail, temp=temp, avail_ids=avail_ids)
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
        candidate_indices=s._semantic_click_candidate_indices(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if not candidate_indices:
            return s._sample(logits, avail=avail, temp=temp, avail_ids=avail_ids)
        return s._sample_sparse_policy_indices(
            logits,
            avail_ids,
            candidate_indices,
            temp=temp,
        )

    def _legal_action_mask(s, logits, avail, avail_ids=None):
        """Mask logits down to currently legal modeled actions."""
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        cache_key=(len(logits), tuple(avail_ids) if avail_ids is not None else None,
                   logits.device.type, getattr(logits.device, 'index', None))
        cached=s._legal_action_mask_cache.get(cache_key)
        if cached is not None and cached.device == logits.device:
            return cached
        mask=torch.full((len(logits),),-float('inf'),device=logits.device)
        if avail is None or len(avail)==0:
            mask.zero_()
            s._legal_action_mask_cache[cache_key]=mask
            return mask
        click_avail=False
        for aid in avail_ids:
            if 1<=aid<=5:
                mask[aid-1]=0.0
            elif aid==6 and len(logits)>5:
                click_avail=True
        if click_avail and len(logits)>5:
            mask[5:]=0.0
        s._legal_action_mask_cache[cache_key]=mask
        return mask

    def _legal_modeled_action_count(s, logits_len, avail_ids):
        """Count legal modeled actions without scanning a dense legality mask."""
        if not avail_ids:
            return int(logits_len)
        count=0
        click_added=False
        for aid in avail_ids:
            if 1 <= aid <= 5:
                count += 1
            elif aid == 6 and logits_len > 5 and not click_added:
                count += (logits_len - 5)
                click_added=True
        return count

    def _candidate_score_map(s, scored, candidate_indices):
        """Fetch candidate scores with one indexed tensor read."""
        if not candidate_indices:
            return {}
        index_tensor=torch.as_tensor(candidate_indices, dtype=torch.long, device=scored.device)
        candidate_scores=scored.index_select(0, index_tensor).detach().cpu().tolist()
        return {
            int(idx): float(score)
            for idx,score in zip(candidate_indices, candidate_scores)
        }

    def _legal_direction_ids(s, avail_ids):
        """Reuse the legal directional action id set for a given availability pattern."""
        if not avail_ids:
            return frozenset()
        cache_key=tuple(avail_ids)
        cached=s._legal_direction_ids_cache.get(cache_key)
        if cached is None:
            cached=frozenset(aid for aid in avail_ids if 1 <= aid <= 4)
            s._legal_direction_ids_cache[cache_key]=cached
        return cached

    def _availability_summary(s, avail_ids):
        """Reuse derived availability flags for a given action-id pattern."""
        if not avail_ids:
            return {
                "has_click": False,
                "has_undo": False,
                "has_modeled": False,
                "legal_dirs": frozenset(),
            }
        cache_key=tuple(avail_ids)
        cached=s._availability_summary_cache.get(cache_key)
        if cached is None:
            legal_dirs=s._legal_direction_ids(avail_ids)
            cached={
                "has_click": 6 in avail_ids,
                "has_undo": 7 in avail_ids,
                "has_modeled": any(1 <= aid <= 6 for aid in avail_ids),
                "legal_dirs": legal_dirs,
            }
            s._availability_summary_cache[cache_key]=cached
        return cached

    def _click_targets_from_components(s, frame, comps, preferred, preferred_coord,
                                       blocked_click_coord):
        """Rank click targets from cached semantic components when no player target is known."""
        if not comps:
            return []
        color_priority={14:0,6:1,9:2,11:3,5:4,7:5,13:6,15:7}
        continuity_scale=s._semantic_continuity_scale()
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
                distance=abs(cy-32)+abs(cx-32)
                if preferred is not None and continuity_scale > 0.0:
                    preferred_distance=abs(cy-preferred[0])+abs(cx-preferred[1])
                    continuity_distance=((continuity_scale * preferred_distance) +
                                         ((1.0 - continuity_scale) * distance))
                else:
                    continuity_distance=distance
                scored.append((
                    color_priority[color],
                    round(float(continuity_distance), 6),
                    -area,
                    distance,
                    cy,
                    cx,
                ))
        scored.sort()
        if not scored:
            return []
        scored_coords=[
            (item[4], item[5]) for item in scored
        ]
        return s._rank_click_target_coords(
            frame,
            scored_coords,
            preferred_coord,
            blocked_click_coord,
        )

    def _rank_click_target_coords(s, frame, scored_coords, preferred_coord, blocked_click_coord):
        """Return the full ranked click-target list for the current frame state."""
        if not scored_coords:
            return []
        coords=[]
        seen=set()
        full_limit=max(1, len(scored_coords))
        if s._prepend_nearest_preferred_coord(
                frame,
                scored_coords,
                coords,
                preferred_coord,
                seen,
                full_limit,
                blocked_click_coord=blocked_click_coord):
            return coords
        if s._append_unblocked_coords(
                frame,
                scored_coords,
                coords,
                seen,
                full_limit,
                blocked_click_coord=blocked_click_coord):
            return coords
        return coords

    def _semantic_click_targets(s, frame, limit=8, blocked_click_coord=None, frame_hash=None):
        """Rank likely interactive click targets from connected components."""
        frame=np.ascontiguousarray(frame, dtype=np.uint8)
        preferred=preferred_coord=s._preferred_click_coord()
        continuity_scale=s._semantic_continuity_scale()
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
        if frame_hash is None:
            frame_hash=s._fast_frame_hash(frame)
        cache_key=(
            frame_hash,
            None if preferred_coord is None else (int(preferred_coord[0]), int(preferred_coord[1])),
            None if blocked_click_coord is None else (int(blocked_click_coord[0]), int(blocked_click_coord[1])),
            s._blocked_click_history_signature(),
            round(float(continuity_scale), 3),
        )
        if s._semantic_click_targets_cache_key == cache_key:
            return s._semantic_click_targets_cache_value[:limit]
        scored=s._semantic_target_candidates(frame, blocked_click_coord=blocked_click_coord, frame_hash=frame_hash)
        if scored:
            scored_coords=[
                (int(round(item['target_y'])), int(round(item['target_x'])))
                for item in scored
            ]
            coords=s._rank_click_target_coords(
                frame,
                scored_coords,
                preferred_coord,
                blocked_click_coord,
            )
            s._semantic_click_targets_cache_key=cache_key
            s._semantic_click_targets_cache_value=coords
            return coords[:limit]
        comps=s._semantic_components(frame, frame_hash=frame_hash)
        coords=s._click_targets_from_components(
            frame,
            comps,
            preferred,
            preferred_coord,
            blocked_click_coord,
        )
        s._semantic_click_targets_cache_key=cache_key
        s._semantic_click_targets_cache_value=coords
        return coords[:limit]

    def _semantic_click_targets_compat(s, frame, limit=8, blocked_click_coord=None, frame_hash=None):
        """Call `_semantic_click_targets` with kwarg fallback for test doubles."""
        try:
            return s._semantic_click_targets(
                frame,
                limit=limit,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
        except TypeError as exc:
            msg=str(exc)
            if 'frame_hash' in msg:
                try:
                    return s._semantic_click_targets(
                        frame,
                        limit=limit,
                        blocked_click_coord=blocked_click_coord,
                    )
                except TypeError as inner_exc:
                    if 'blocked_click_coord' not in str(inner_exc):
                        raise
                    return s._semantic_click_targets(frame, limit=limit)
            if 'blocked_click_coord' not in msg:
                raise
            return s._semantic_click_targets(frame, limit=limit)

    def _semantic_components(s, frame, frame_hash=None):
        """Return semantic components when the sprite detector is available."""
        if frame_hash is None:
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
            grid_cache_key=(frame_hash, np.shape(frame))
            if s._semantic_detector_grid_cache_key == grid_cache_key:
                detector_grid=s._semantic_detector_grid_cache_value
            else:
                detector_grid=np.ascontiguousarray(frame, dtype=np.uint8).tolist()
                s._semantic_detector_grid_cache_key=grid_cache_key
                s._semantic_detector_grid_cache_value=detector_grid
            semantic=detector(detector_grid)
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

    def _semantic_target_candidates(s, frame, blocked_click_coord=None, frame_hash=None):
        """Rank semantic targets using class priority plus player-relative distance."""
        if frame_hash is None:
            frame_hash=s._fast_frame_hash(frame)
        preferred=s._semantic_target_coord
        continuity_scale=s._semantic_continuity_scale()
        recent_direction=s._recent_direction_action_index(frame, frame_hash=frame_hash)
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
        recent_progress_delta=s._recent_direction_progress_delta(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        cache_key=(
            frame_hash,
            None if preferred is None else (int(preferred[0]), int(preferred[1])),
            None if blocked_click_coord is None else (int(blocked_click_coord[0]), int(blocked_click_coord[1])),
            recent_direction,
            None if recent_progress_delta is None else round(float(recent_progress_delta), 3),
            round(float(continuity_scale), 3),
        )
        if s._semantic_target_candidates_cache_key == cache_key:
            return s._semantic_target_candidates_cache_value
        comps=s._semantic_components(frame, frame_hash=frame_hash)
        if not comps:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        player=None
        player_area=-1
        for key in ('4', '12'):
            for comp in comps.get(key) or ():
                area=int(comp.get('cell_count', 0))
                if area > player_area:
                    player=comp
                    player_area=area
        if player is None:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        center=player.get('center')
        if not center or len(center) != 2:
            s._semantic_target_candidates_cache_key=cache_key
            s._semantic_target_candidates_cache_value=[]
            return []
        py=float(center[0]); px=float(center[1])
        preferred_y=None if preferred is None else float(preferred[0])
        preferred_x=None if preferred is None else float(preferred[1])
        blocked_click_known=blocked_click_coord is not None
        target_specs=[]
        for color, priority in ((14,0), (6,1), (11,2), (5,3), (9,4), (7,5), (13,6), (15,7)):
            for comp in comps.get(str(color)) or []:
                tcenter=comp.get('center')
                if not tcenter or len(tcenter) != 2:
                    continue
                ty=float(tcenter[0]); tx=float(tcenter[1])
                target_coord=(int(round(ty)), int(round(tx)))
                if s._blocked_click_matches_coord(
                        frame,
                        target_coord,
                        blocked_click_coord=blocked_click_coord if blocked_click_known else None,
                        frame_hash=frame_hash):
                    continue
                dist=abs(ty-py)+abs(tx-px)
                if dist < 1.0:
                    continue
                area=int(comp.get('cell_count', 0))
                if area <= 0 or area > 512:
                    continue
                score=priority * 2.0 + dist / 6.0
                continuity_bonus=0.0
                if preferred_y is not None:
                    continuity_dist=abs(preferred_y-ty)+abs(preferred_x-tx)
                    continuity_bonus=(max(0.0, 0.6 - 0.1 * continuity_dist) *
                                      continuity_scale)
                    score -= continuity_bonus
                momentum_bonus=0.0
                counter_momentum_penalty=0.0
                if recent_progress_delta is None or recent_progress_delta >= -0.5:
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
                score_key=round(score, 6)
                target_specs.append({
                    'score': score,
                    'score_key': score_key,
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
        target_specs.sort(key=lambda item: (item['score_key'], -item['continuity_bonus'], item['counter_momentum_penalty'], -item['momentum_bonus'], -item['area']))
        s._semantic_target_candidates_cache_key=cache_key
        s._semantic_target_candidates_cache_value=target_specs
        return target_specs

    def _semantic_target_choice(s, frame, blocked_click_coord=None, frame_hash=None):
        """Return the best semantic target using class priority plus distance."""
        target_specs=s._semantic_target_candidates(frame, blocked_click_coord=blocked_click_coord, frame_hash=frame_hash)
        if not target_specs:
            return None
        return target_specs[0]

    def _semantic_direction_action(s, frame, avail, avail_ids=None, frame_hash=None):
        """Choose a directional move that heads toward a likely target."""
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        bonuses=s._semantic_direction_bonuses(frame, avail, avail_ids=avail_ids, frame_hash=frame_hash)
        if not bonuses:
            return None
        best_idx=None
        best_bonus=float('-inf')
        for idx,bonus in bonuses.items():
            idx=int(idx)
            bonus=float(bonus)
            if bonus <= 0.0:
                continue
            if bonus > best_bonus:
                best_bonus=bonus
                best_idx=idx
        if best_idx is None:
            return None
        return best_idx, None
        return None

    def _frame_matches_previous(s, frame, frame_hash=None):
        """Return True when the current raw frame matches the stored previous frame."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return False
        return relation['matches_previous']

    def _previous_frame_relation(s, frame, frame_hash=None):
        """Cache how the current frame relates to the stored previous frame."""
        if s.pr is None:
            return None
        try:
            curr_shape=np.shape(frame)
            prev_shape=np.shape(s.pr)
        except Exception:
            return None
        if curr_shape != prev_shape:
            return {
                'matches_previous': False,
                'changed_since_previous': True,
                'recent_direction_action_index': int(s.pai) if s.pai is not None and 0 <= int(s.pai) < 4 else None,
                'recent_direction_axis': ('vertical' if s.pai in (0, 1) else 'horizontal')
                if s.pai is not None and 0 <= int(s.pai) < 4 else None,
                'recent_click_action_index': int(s.pai) if s.pai is not None and 5 <= int(s.pai) < 5 + s.G * s.G else None,
                'blocked_direction_action_index': None,
                'blocked_click_coord': None,
                'blocked_click_action_index': None,
            }
        if frame_hash is None:
            frame_hash=s._fast_frame_hash(frame)
        key=(frame_hash, s.ph, s.pai, curr_shape, prev_shape)
        cached=s._previous_frame_relation_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            matches_previous=np.array_equal(frame, s.pr)
        except Exception:
            return None
        recent_direction_action_index=None
        recent_direction_axis=None
        recent_click_action_index=None
        blocked_direction_action_index=None
        blocked_click_coord=None
        blocked_click_action_index=None
        if s.pai is not None:
            pai=int(s.pai)
            if 0 <= pai < 4:
                if matches_previous:
                    blocked_direction_action_index=pai
                else:
                    recent_direction_action_index=pai
                    recent_direction_axis='vertical' if pai in (0, 1) else 'horizontal'
            else:
                click_base=5
                click_limit=click_base + s.G * s.G
                if click_base <= pai < click_limit:
                    if matches_previous:
                        blocked_click_coord=s._click_coord_from_action_index(pai)
                        blocked_click_action_index=pai
                    else:
                        recent_click_action_index=pai
        relation={
            'matches_previous': matches_previous,
            'changed_since_previous': not matches_previous,
            'recent_direction_action_index': recent_direction_action_index,
            'recent_direction_axis': recent_direction_axis,
            'recent_click_action_index': recent_click_action_index,
            'blocked_direction_action_index': blocked_direction_action_index,
            'blocked_click_coord': blocked_click_coord,
            'blocked_click_action_index': blocked_click_action_index,
        }
        s._previous_frame_relation_cache=(key, relation)
        return relation

    def _frame_changed_since_previous(s, frame, frame_hash=None):
        """Return True when the current frame safely differs from the stored previous frame."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return False
        return relation['changed_since_previous']

    def _recent_direction_action_index(s, frame, frame_hash=None):
        """Return the last directional action index when it changed the frame."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['recent_direction_action_index']

    def _recent_direction_axis(s, frame, frame_hash=None):
        """Return the axis implied by the most recent effective directional action."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['recent_direction_axis']

    def _recent_click_action_index(s, frame, frame_hash=None):
        """Return the last click action index when it changed the frame."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['recent_click_action_index']

    def _blocked_direction_action_index(s, frame, frame_hash=None):
        """Return the last directional action index if it left the state unchanged."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['blocked_direction_action_index']

    def _blocked_click_coord(s, frame, frame_hash=None):
        """Return the last click coordinate if it left the state unchanged."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['blocked_click_coord']

    def _blocked_click_action_index(s, frame, frame_hash=None):
        """Return the last click action index if it left the state unchanged."""
        relation=s._previous_frame_relation(frame, frame_hash=frame_hash)
        if relation is None:
            return None
        return relation['blocked_click_action_index']

    def _coord_matches_blocked_click(s, coord, blocked_click_coord):
        """Treat nearby click jitter as the same blocked click region."""
        return (blocked_click_coord is not None and
                (abs(coord[0]-blocked_click_coord[0]) +
                 abs(coord[1]-blocked_click_coord[1])) <= 2)

    def _blocked_click_matches_coord(s, frame, coord, blocked_click_coord=None, frame_hash=None):
        """Treat nearby click jitter as the same blocked click region."""
        if blocked_click_coord is None:
            blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
        if s._coord_matches_blocked_click(coord, blocked_click_coord):
            return True
        for blocked_coord in s._blocked_click_history:
            if s._coord_matches_blocked_click(coord, blocked_coord):
                return True
        return False

    def _semantic_click_candidate_indices(s, frame, click_targets=None,
                                          blocked_click_coord=None, frame_hash=None):
        """Reuse the ordered semantic/fallback click shortlist as action indices."""
        preferred_click_coord=s._preferred_click_coord()
        cache_key=(
            None if frame_hash is None else int(frame_hash),
            None if blocked_click_coord is None else (int(blocked_click_coord[0]), int(blocked_click_coord[1])),
            None if preferred_click_coord is None else (int(preferred_click_coord[0]), int(preferred_click_coord[1])),
            tuple(click_targets) if click_targets is not None else None,
            s._blocked_click_history_signature(),
            round(float(s._semantic_continuity_scale()), 3),
        )
        if s._semantic_click_candidate_indices_cache_key == cache_key:
            return s._semantic_click_candidate_indices_cache_value
        if click_targets is None:
            click_targets=s._semantic_click_targets_compat(
                frame,
                limit=6,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
        candidate_indices=[]
        candidate_seen=set()
        for ty,tx in click_targets:
            s._append_candidate_index(
                candidate_indices,
                candidate_seen,
                s._click_action_index((ty, tx)),
            )
        for ty,tx in s._heuristic_click_fallback_targets(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash):
            s._append_candidate_index(
                candidate_indices,
                candidate_seen,
                s._click_action_index((ty, tx)),
            )
        if (s._preferred_click_continuity_active() and
                not s._blocked_click_matches_coord(
                    frame,
                    preferred_click_coord,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash,
                )):
            s._append_candidate_index(
                candidate_indices,
                candidate_seen,
                s._click_action_index(preferred_click_coord),
            )
        s._semantic_click_candidate_indices_cache_key=cache_key
        s._semantic_click_candidate_indices_cache_value=candidate_indices
        return candidate_indices

    def _semantic_direction_bonuses(s, frame, avail=None, avail_ids=None, frame_hash=None,
                                    avail_summary=None):
        """Soft directional preferences derived from semantic targets."""
        if avail_ids is None and avail is not None:
            avail_ids=s._available_action_ids(avail)
        if avail_summary is None and avail_ids is not None:
            avail_summary=s._availability_summary(avail_ids)
        recent_direction=s._recent_direction_action_index(frame, frame_hash=frame_hash)
        opposite_recent=s._opposite_direction_index(recent_direction)
        blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
        retry_blocked_direction=s._retry_blocked_direction_after_stale_wait(
            frame,
            avail_ids if avail_ids is not None else (),
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )
        recent_progress_delta=s._recent_direction_progress_delta(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        legal_dirs=None
        if avail_summary is not None:
            legal_dirs=avail_summary["legal_dirs"]
            if not legal_dirs:
                return {}
        preferred_axis=s._recent_direction_axis(frame, frame_hash=frame_hash)
        fallback_bonuses=None
        for choice in s._semantic_target_candidates(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash):
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
                for blocked_idx in range(4):
                    if (not retry_blocked_direction and
                            s._direction_matches_blocked_history(
                                blocked_idx,
                                frame,
                                frame_hash=frame_hash,
                                blocked_direction=blocked_direction)):
                        if blocked_idx in bonuses:
                            bonuses[blocked_idx] = min(bonuses[blocked_idx], -0.12)
                        else:
                            bonuses[blocked_idx] = -0.12
                if (opposite_recent is not None and
                        opposite_recent in bonuses and
                        len(bonuses) > 1 and
                        (recent_progress_delta is None or recent_progress_delta >= -0.5)):
                    bonuses[opposite_recent] -= 0.30
                if any(float(bonus) > 0.0 for bonus in bonuses.values()):
                    return bonuses
                if (fallback_bonuses is None or
                        max(float(bonus) for bonus in bonuses.values()) >
                        max(float(bonus) for bonus in fallback_bonuses.values())):
                    fallback_bonuses=bonuses
        return fallback_bonuses or {}

    def _semantic_exploration_logits(s, frame, avail, include_clicks, blocked_click_coord=None, avail_ids=None,
                                     frame_hash=None, avail_summary=None):
        """Bias exploratory sampling toward semantic movement/click targets."""
        size=4101 if include_clicks else 5
        logits=torch.zeros(size, device=s.device)
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids)
        retry_blocked_direction=s._retry_blocked_direction_after_stale_wait(
            frame,
            avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )
        current_blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        target_choice=s._semantic_target_choice(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        for action_idx, bonus in s._semantic_direction_bonuses(
                frame,
                avail,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary).items():
            if 0 <= int(action_idx) < 5:
                logits[int(action_idx)] = float(bonus)
        wait_bonus=s._wait_recovery_bonus(
            frame,
            avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )
        if wait_bonus > 0.0:
            logits[4] = max(float(logits[4].item()), wait_bonus)
        elif retry_blocked_direction:
            logits[4] = -float('inf')
        for blocked_idx in range(4):
            if (not retry_blocked_direction and
                    s._direction_matches_blocked_history(
                        blocked_idx,
                        frame,
                        frame_hash=frame_hash,
                        blocked_direction=current_blocked_direction)):
                logits[int(blocked_idx)] = -float('inf')
        if include_clicks:
            if blocked_click_coord is None:
                blocked_click_coord=s._blocked_click_coord(frame, frame_hash=frame_hash)
            click_scale=s._semantic_click_bonus_scale(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
                target_choice=target_choice,
            )
            preferred_click_coord=s._preferred_click_coord()
            for rank,(ty,tx) in enumerate(s._semantic_click_targets_compat(
                    frame,
                    limit=6,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash)):
                idx=s._click_action_index((ty, tx))
                if 5 <= idx < logits.numel():
                    logits[idx] = max(float(logits[idx].item()), max(0.0, 0.8 - 0.1 * rank) * click_scale)
            for rank,(ty,tx) in enumerate(s._heuristic_click_fallback_targets(
                    frame,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash)[:6]):
                idx=s._click_action_index((ty, tx))
                if 5 <= idx < logits.numel():
                    logits[idx] = max(float(logits[idx].item()), max(0.0, 0.35 - 0.05 * rank) * click_scale)
            if (s._preferred_click_continuity_active() and
                    not s._blocked_click_matches_coord(
                        frame,
                        preferred_click_coord,
                        blocked_click_coord=blocked_click_coord,
                        frame_hash=frame_hash)):
                preferred_idx=s._click_action_index(preferred_click_coord)
                if 5 <= preferred_idx < logits.numel():
                    logits[preferred_idx] = max(float(logits[preferred_idx].item()), 0.08 * click_scale)
            blocked_click=s._blocked_click_action_index(frame, frame_hash=frame_hash)
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
            for blocked_coord in s._blocked_click_history:
                by,bx=int(blocked_coord[0]), int(blocked_coord[1])
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
                                           blocked_click_coord=None, avail_ids=None, frame_hash=None):
        """Semantic action indices that should always participate in rescoring."""
        candidates=[]
        seen=set()
        if avail_ids is None and avail is not None:
            avail_ids=s._available_action_ids(avail)
        if direction_bonuses is None:
            direction_bonuses=s._semantic_direction_bonuses(frame, avail, avail_ids=avail_ids, frame_hash=frame_hash)
        for action_idx in direction_bonuses.keys():
            idx=int(action_idx)
            if 0 <= idx < 5 and idx not in seen:
                seen.add(idx)
                candidates.append(idx)
        if s._wait_recovery_bonus(
                frame,
                avail_ids or (),
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash) > 0.0 and 4 not in seen:
            seen.add(4)
            candidates.append(4)
        if include_clicks:
            for idx in s._semantic_click_candidate_indices(
                    frame,
                    click_targets=click_targets,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=frame_hash):
                if idx not in seen and 5 <= idx < 4101:
                    seen.add(idx)
                    candidates.append(idx)
        return candidates

    def _semantic_goal_distance(s, frame, blocked_click_coord=None, frame_hash=None, target_choice=None):
        """Estimated player-to-target Manhattan distance from semantic detections."""
        choice=target_choice
        if choice is None:
            choice=s._semantic_target_choice(frame, blocked_click_coord=blocked_click_coord, frame_hash=frame_hash)
        if not choice:
            return None
        return float(choice['distance'])

    def _semantic_click_bonus_scale(s, frame, blocked_click_coord=None, frame_hash=None, target_choice=None):
        """Reduce click priors when the semantic target is far from the player."""
        goal_distance=s._semantic_goal_distance(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            target_choice=target_choice,
        )
        if goal_distance is None:
            return 1.0
        return max(0.25, min(1.0, 4.0 / max(float(goal_distance), 1.0)))

    def _refresh_semantic_target_coord(s, frame, fallback_coord=None, blocked_click_coord=None, frame_hash=None):
        """Track the current semantic target so later tie-breaks keep pursuing it."""
        choice=s._semantic_target_choice(frame, blocked_click_coord=blocked_click_coord, frame_hash=frame_hash)
        if choice is not None:
            s._semantic_target_coord=(int(round(choice['target_y'])), int(round(choice['target_x'])))
        elif (fallback_coord is not None and
              not s._blocked_click_matches_coord(
                  frame,
                  fallback_coord,
                  blocked_click_coord=blocked_click_coord,
                  frame_hash=frame_hash)):
            s._semantic_target_coord=(int(fallback_coord[0]), int(fallback_coord[1]))
        else:
            s._semantic_target_coord=None

    def _heuristic_click_fallback_targets(s, frame, blocked_click_coord=None, frame_hash=None):
        """Return cached heuristic click fallback targets for the current frame."""
        if frame_hash is None:
            frame_hash=s._fast_frame_hash(frame)
        cache_key=(
            frame_hash,
            int(s._bg),
            None if blocked_click_coord is None else (int(blocked_click_coord[0]), int(blocked_click_coord[1])),
            s._blocked_click_history_signature(),
        )
        if s._heuristic_click_fallback_cache_key == cache_key:
            return s._heuristic_click_fallback_cache_value
        cnt=np.bincount(frame.ravel(), minlength=16)
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
                        blocked_click_coord=blocked_click_coord,
                        frame_hash=frame_hash):
                    continue
                targets.append((coord[1],coord[0],len(ys)))
        targets.sort(key=lambda t:t[2])
        fallback_targets=[(int(ty), int(tx)) for tx,ty,_ in targets]
        s._heuristic_click_fallback_cache_key=cache_key
        s._heuristic_click_fallback_cache_value=fallback_targets
        return fallback_targets

    def _heuristic(s, frame, avail, step, blocked_click_coord=None, avail_ids=None, frame_hash=None,
                   avail_summary=None):
        if avail_ids is None:
            avail_ids=s._available_action_ids(avail)
        if avail_summary is None:
            avail_summary=s._availability_summary(avail_ids)
        av=avail_summary["legal_dirs"]
        direct_click_choice=s._semantic_direct_click_choice(
            frame,
            avail,
            avail_ids=avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        )
        if direct_click_choice is not None:
            return direct_click_choice
        semantic_dir=s._semantic_direction_action(frame, avail, avail_ids=avail_ids, frame_hash=frame_hash)
        if semantic_dir is not None:
            return semantic_dir
        blocked_direction=s._blocked_direction_action_index(frame, frame_hash=frame_hash)
        preferred_dir=int(s.pai) if s.pai is not None and 0 <= int(s.pai) < 4 else None
        preferred_coord=s._preferred_click_coord()
        if step < 4:
            preferred_choice=s._preferred_direction_choice(
                preferred_dir if not s._direction_matches_blocked_history(
                    preferred_dir,
                    frame,
                    frame_hash=frame_hash,
                    blocked_direction=blocked_direction) else None,
                blocked_direction,
                av,
            )
            if preferred_choice is not None:
                return preferred_choice
        for d in [1, 2, 3, 4]:
            if s._direction_matches_blocked_history(
                    d-1,
                    frame,
                    frame_hash=frame_hash,
                    blocked_direction=blocked_direction):
                continue
            if d in av and step < 4:
                return d - 1, None
        if avail_summary["has_click"]:
            semantic_targets=s._semantic_click_targets_compat(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            semantic_target_choice=s._preferred_click_target_choice(semantic_targets, preferred_coord, step)
            if semantic_target_choice is not None:
                return 5, semantic_target_choice
            fallback_targets=s._heuristic_click_fallback_targets(
                frame,
                blocked_click_coord=blocked_click_coord,
                frame_hash=frame_hash,
            )
            fallback_target_choice=s._preferred_click_target_choice(fallback_targets, preferred_coord, step)
            if fallback_target_choice is not None:
                return 5, fallback_target_choice
        choices=[a for a in avail_ids if 1<=a<=5]
        preferred_choice=s._preferred_direction_choice(
            preferred_dir if not s._direction_matches_blocked_history(
                preferred_dir,
                frame,
                frame_hash=frame_hash,
                blocked_direction=blocked_direction) else None,
            blocked_direction,
            choices,
        )
        if preferred_choice is not None:
            return preferred_choice
        if 5 in avail_ids and not any(a in av for a in (1, 2, 3, 4)):
            return 4, None
        directional_choices=[a for a in choices if 1<=a<=4]
        stale_wait=s._stale_wait_recovery(frame)
        if s._blocked_direction_history or blocked_direction is not None:
            unblocked_directional_choices=[
                a for a in directional_choices
                if not s._direction_matches_blocked_history(
                    a-1,
                    frame,
                    frame_hash=frame_hash,
                    blocked_direction=blocked_direction)
            ]
            if unblocked_directional_choices:
                directional_choices=unblocked_directional_choices
            elif 5 in avail_ids and not stale_wait:
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
        indices_np=np.asarray(indices, dtype=np.int64)
        frames_l=[s.buf[i] for i in indices]
        if len(s.buf_hashes) >= len(s.buf):
            frame_hashes_np=s._packed_array_view(s.buf_hashes, np.uint32, count=len(s.buf))[indices_np].astype(np.int64, copy=False)
        else:
            frame_hashes_np=np.asarray([s._fast_frame_hash(frame) for frame in frames_l], dtype=np.int64)
        if frame_hashes_np.size:
            unique_hashes, first_positions, inverse=np.unique(
                frame_hashes_np,
                return_index=True,
                return_inverse=True,
            )
        else:
            unique_hashes=np.empty(0, dtype=np.int64)
            first_positions=np.empty(0, dtype=np.int64)
            inverse=np.empty(0, dtype=np.int64)
        feature_by_unique=[None] * int(unique_hashes.size)
        uncached_unique_positions=[
            unique_pos for unique_pos, frame_hash in enumerate(unique_hashes)
            if int(frame_hash) not in s._frame_feature_cache
        ]
        # Group uncached frames and compute features in one batched pass
        if uncached_unique_positions:
            frames_np=np.stack([frames_l[int(first_positions[unique_pos])] for unique_pos in uncached_unique_positions],axis=0)
            frames_np=s._sanitize_frame_batch(frames_np)
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
            for j,unique_pos in enumerate(uncached_unique_positions):
                h=int(unique_hashes[unique_pos])
                features=s._pack_replay_feature_channels(
                    oh[j:j+1],
                    bg_m[j:j+1],
                    rarity[j:j+1],
                    edge[j:j+1],
                )
                s._frame_feature_cache[h]=features
                feature_by_unique[unique_pos]=features
            # Evict oldest entries when cache exceeds limit
            if len(s._frame_feature_cache)>s._frame_feature_cache_max:
                for _ in range(len(s._frame_feature_cache)-s._frame_feature_cache_max):
                    s._frame_feature_cache.pop(next(iter(s._frame_feature_cache)))
        # Gather packed features once per unique frame, then index back into the batch.
        for unique_pos,frame_hash in enumerate(unique_hashes):
            if feature_by_unique[unique_pos] is not None:
                continue
            h=int(frame_hash)
            packed=s._cached_replay_features(s._frame_feature_cache.get(h))
            if packed is None:
                rebuilt=s._replay_batch_tensor([indices[int(first_positions[unique_pos])]])
                packed=rebuilt[:, :19]
                s._frame_feature_cache[h]=packed
            elif not torch.is_tensor(s._frame_feature_cache.get(h)):
                s._frame_feature_cache[h]=packed
            feature_by_unique[unique_pos]=packed
        feature_bank=torch.cat(feature_by_unique,dim=0)
        gather_idx=torch.as_tensor(inverse, dtype=torch.long, device=feature_bank.device)
        feature_batch=feature_bank.index_select(0, gather_idx)
        B=len(indices)
        tail=s._replay_tail_batch(B, feature_batch)
        states=torch.cat([feature_batch,tail],dim=1)
        if s.device.type=='cuda':
            states=states.contiguous(memory_format=torch.channels_last)
        return states

    def _train(s):
        if len(s.buf)<s.bsz:return False
        # PER sampling: importance-weighted by priority
        n=len(s.buf)
        probs=s._sampling_probabilities(n)
        indices=np.random.choice(n,size=s.bsz,p=probs)
        actions_view,rewards_view,next_flags,_=s._replay_numeric_views(n)
        acts_np=actions_view[indices].astype(np.int64, copy=False)
        rews_np=rewards_view[indices]
        needs_click_head=bool((acts_np>=5).any())
        # Importance sampling weights
        sampled_probs=probs[indices]
        inv_weights=np.float32(1.0)/(np.float32(n)*sampled_probs)
        is_weights=np.power(inv_weights, np.float32(s._per_beta), dtype=np.float32)
        max_is_weight=float(is_weights.max())
        if max_is_weight > 0.0:
            is_weights/=np.float32(max_is_weight)
        else:
            is_weights.fill(np.float32(1.0))
        s._per_beta=min(1.0,s._per_beta+s._per_beta_step)
        states=s._replay_batch_tensor(indices)
        acts=torch.from_numpy(acts_np).to(s.device,non_blocking=True)
        rews=torch.from_numpy(rews_np).to(s.device,non_blocking=True)
        isw=torch.from_numpy(is_weights).to(s.device,non_blocking=True)
        s.net.train();s.opt.zero_grad(set_to_none=True)
        try:
            with s._amp_context():
                logits=s.net(states) if needs_click_head else s.net.forward_actions(states)
                acts_c=acts.clamp(0,logits.size(1)-1)
                q_sa=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
                # Munchausen DQN target: r + alpha*tau*log(pi(a|s)) + gamma*max_a' Q_target(s',a')
                td_target=rews.clone()
                if next_flags.shape[0] < n:
                    next_present=np.fromiter(
                        (idx<len(s.buf_next_frames) and s.buf_next_frames[idx] is not None for idx in indices),
                        dtype=np.bool_,
                        count=s.bsz,
                    )
                else:
                    next_present=next_flags[indices].astype(np.bool_, copy=False)
                has_next_mask=torch.from_numpy(next_present).to(s.device,non_blocking=True)
                if next_present.any() and s._target_net is not None:
                    next_indices=indices[next_present]
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
                s._update_sampled_priorities(indices, td_error)
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

        start=M-K
        diffs_l=list(islice(s._aem_diffs, start, None))

        # Stack on CPU first; assigning one small tensor at a time into a CUDA
        # tensor causes many tiny transfers/synchronization points.
        diffs_np=np.stack([d.astype(np.float32, copy=False) for d in diffs_l], axis=0)
        diffs=torch.as_tensor(diffs_np, dtype=torch.float32, device=s.device).view(1,K,1,64,64)
        acts_np=np.fromiter((min(int(a),4) for a in islice(s._aem_actions, start, None)), dtype=np.int64, count=K)
        rews_np=np.fromiter((float(r) for r in islice(s._aem_rewards, start, None)), dtype=np.float32, count=K)
        acts=torch.as_tensor(acts_np, dtype=torch.long, device=s.device).view(1,K)
        rews=torch.as_tensor(rews_np, dtype=torch.float32, device=s.device).view(1,K)

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
                            s.buf_priorities[i] = s._priority_from_reward(s.buf_rewards[i])
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
                # FIX 1: Reset visited hashes on every level change.
                # FIX 4: Only reset epsilon if BFS did not solve this level.
                s._reset_level_runtime_state(lvl)

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
                    GameAction.RESET.value if hasattr(GameAction.RESET, "value") else int(GameAction.RESET),
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
                raw_snapshot=s._snapshot_frame(raw)
                s.fhist.append(raw_snapshot)
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
                    raw_snapshot=raw_snapshot,
                )

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = s._fast_frame_hash(raw)
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = s._available_action_ids(avail)
            avail_summary=s._availability_summary(avail_ids)
            s._undo_avail = avail_summary["has_undo"]
            modeled_avail = avail_summary["has_modeled"]

            if s.pt is not None and s.pr is not None:
                curr_objs=None; move_bonus=0.0; moved=0
                diff_map=(s.pr!=raw)&s._reward_mask;changed=bool(np.any(diff_map))
                prev_h = s.ph if s.ph is not None else s._fast_frame_hash(s.pr)
                r = None
                if s.pai is not None:
                    eh=(prev_h,int(s.pai))
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
                if changed:
                    s._ckpt_hash=ch
                    s._unproductive=0
                    s._decay_blocked_click_history()
                    s._decay_blocked_direction_history()
                else:
                    s._unproductive+=1
                    if s.pai is not None and s.pai >= 5:
                        s._remember_blocked_click_coord(s._click_coord_from_action_index(s.pai))
                    elif s.pai is not None and 0 <= int(s.pai) < 4:
                        s._remember_blocked_direction_index(s.pai)

                # Action repeat: if the last action was a directional move that
                # produced a frame change, repeat it with moderate probability to
                # exploit consistent movement patterns (e.g. walking across a maze)
                # without requiring the CNN to learn to chain identical actions.
                if changed:
                    repeated_action=s._try_repeat_direction_action(raw, avail, avail_ids, tensor, ch)
                    if repeated_action is not None:
                        return repeated_action

            if not modeled_avail:
                return s._handle_non_modeled_availability(tensor, raw, ch)

            blocked_click_coord=s._blocked_click_coord(raw, frame_hash=ch)
            if (s._undo_avail and s._ckpt_hash and
                    s._modeled_frontier_exhausted(
                        raw,
                        avail_ids,
                        blocked_click_coord=blocked_click_coord,
                        frame_hash=ch,
                        avail_summary=avail_summary)):
                return s._finalize_control_action(
                    7,
                    "undo-frontier",
                    tensor=tensor,
                    raw=raw,
                    frame_hash=ch,
                    remember_recent=True,
                )

            s._ensure_click_template(raw)

            forced_undo=s._maybe_force_undo(tensor, raw, ch)
            if forced_undo is not None:
                return forced_undo
            if not s._wd:
                warmup_choice=s._prime_warmup_action(raw, avail, frame_hash=ch)
                if warmup_choice is not None:
                    aidx,coords=warmup_choice

            if s._wd:
                aidx,coords=s._choose_policy_action(
                    tensor,
                    raw,
                    avail,
                    avail_ids,
                    blocked_click_coord,
                    frame_hash=ch,
                )
                # Cosine annealing epsilon schedule: gradual decay then plateau
                s._eps_steps+=1; total_steps=5000
                frac=min(s._eps_steps/total_steps,1.0)
                s._eps=s._eps_min+(0.15-s._eps_min)*0.5*(1+np.cos(np.pi*frac))
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            return s._finalize_modeled_action(
                aidx,
                coords,
                tensor,
                raw,
                ch,
                blocked_click_coord,
            )

        except Exception as e:
            logger.debug("choose_action fallback triggered: %s", traceback.format_exc())
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
            avail_summary=s._availability_summary(avail_ids)
            blocked_dir = s._blocked_direction_action_index(raw, frame_hash=ch) if raw is not None else None
            if raw is not None:
                blocked_click_coord=s._blocked_click_coord(raw, frame_hash=ch)
                if (avail_summary["has_undo"] and s._ckpt_hash and
                        s._modeled_frontier_exhausted(
                            raw,
                            avail_ids,
                            blocked_click_coord=blocked_click_coord,
                            frame_hash=ch,
                            avail_summary=avail_summary)):
                    return s._finalize_control_action(
                        7,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        remember_recent=True,
                    )
                direct_click_choice=s._semantic_direct_click_choice(
                    raw,
                    avail,
                    avail_ids=avail_ids,
                    blocked_click_coord=blocked_click_coord,
                    frame_hash=ch,
                )
                if direct_click_choice is not None:
                    _, coords=direct_click_choice
                    a = s._click_action(coords)
                    s._refresh_semantic_target_coord(
                        raw,
                        fallback_coord=coords,
                        blocked_click_coord=blocked_click_coord,
                        frame_hash=ch,
                    )
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=s._click_action_index(coords),
                        remember_recent=True,
                    )
                semantic_dir=s._semantic_direction_action(raw, avail, frame_hash=ch)
                if semantic_dir is not None:
                    aid=int(semantic_dir[0]) + 1
                    a = s._fresh_action(aid)
                    s._refresh_semantic_target_coord(raw, frame_hash=ch)
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=aid - 1,
                        remember_recent=True,
                    )
                if avail_summary["has_click"]:
                    semantic_clicks=s._semantic_click_targets_compat(raw, limit=1, frame_hash=ch)
                    if semantic_clicks:
                        coords=semantic_clicks[0]
                        a = s._click_action(coords)
                        s._refresh_semantic_target_coord(raw, fallback_coord=coords, frame_hash=ch)
                        return s._finalize_action(
                            a,
                            f"err:{e}",
                            tensor=tensor,
                            raw=raw,
                            frame_hash=ch,
                            action_idx=s._click_action_index(coords),
                            remember_recent=True,
                        )
                if 5 in avail_ids and s._wait_recovery_bonus(
                        raw,
                        avail_ids,
                        blocked_click_coord=blocked_click_coord,
                        frame_hash=ch,
                        avail_summary=avail_summary) > 0.0:
                    a=s._fresh_action(5)
                    s._refresh_semantic_target_coord(raw, frame_hash=ch)
                    return s._finalize_action(
                        a,
                        f"err:{e}",
                        tensor=tensor,
                        raw=raw,
                        frame_hash=ch,
                        action_idx=4,
                        remember_recent=True,
                    )
            deferred_direction=None
            for aid in avail_ids:
                if 1 <= aid <= 5:
                    if aid <= 4 and raw is not None and s._direction_matches_blocked_history(aid - 1, raw, frame_hash=ch):
                        if deferred_direction is None:
                            deferred_direction = aid
                        continue
                    if (aid == 5 and deferred_direction is not None and raw is not None and
                            s._stale_wait_recovery(raw)):
                        continue
                    a = s._fresh_action(aid)
                    if raw is not None:
                        s._refresh_semantic_target_coord(raw, frame_hash=ch)
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
                        blocked_click_coord=s._blocked_click_coord(raw, frame_hash=ch)
                        semantic_clicks=s._semantic_click_targets_compat(
                            raw,
                            limit=1,
                            blocked_click_coord=blocked_click_coord,
                            frame_hash=ch,
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
                            frame_hash=ch,
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
                GameAction.RESET.value if hasattr(GameAction.RESET, "value") else int(GameAction.RESET),
                f"err:{e}",
                clear_recent=True,
            )
