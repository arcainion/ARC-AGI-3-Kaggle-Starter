from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "agent" / "my_agent.py"


class _Action:
    def __init__(self, value: int):
        self.value = int(value)
        self.data = None
        self.reasoning = ""

    def set_data(self, data):
        self.data = dict(data)


class _GameAction:
    ACTION1 = _Action(1)
    ACTION2 = _Action(2)
    ACTION3 = _Action(3)
    ACTION4 = _Action(4)
    ACTION5 = _Action(5)
    ACTION6 = _Action(6)
    ACTION7 = _Action(7)
    RESET = _Action(0)

    @staticmethod
    def from_id(value: int):
        valid_actions = {
            0: _GameAction.RESET,
            1: _GameAction.ACTION1,
            2: _GameAction.ACTION2,
            3: _GameAction.ACTION3,
            4: _GameAction.ACTION4,
            5: _GameAction.ACTION5,
            6: _GameAction.ACTION6,
            7: _GameAction.ACTION7,
        }
        template = valid_actions[int(value)]
        return _Action(template.value)


class _GameState:
    WIN = object()
    NOT_PLAYED = object()
    GAME_OVER = object()
    PLAYING = object()


class _ActionInput:
    def __init__(self, id, data=None):
        self.id = id
        self.data = data


class _AgentBase:
    def __init__(self, *args, **kwargs):
        self.card_id = kwargs.get("card_id")
        self.game_id = kwargs.get("game_id", "bench-game")
        self.agent_name = kwargs.get("agent_name", "bench-agent")
        self.ROOT_URL = kwargs.get("ROOT_URL", "http://localhost")
        self.record = kwargs.get("record", False)
        self.arc_env = kwargs.get("arc_env")
        self.tags = kwargs.get("tags", [])
        self.frames = []
        self.guid = None
        self.is_playback = False
        self.recorder = None
        self.action_counter = 0


def _install_agent_stubs() -> None:
    agents_pkg = types.ModuleType("agents")
    agents_agent_mod = types.ModuleType("agents.agent")
    agents_agent_mod.Agent = _AgentBase
    agents_pkg.agent = agents_agent_mod
    sys.modules["agents"] = agents_pkg
    sys.modules["agents.agent"] = agents_agent_mod

    arcengine_mod = types.ModuleType("arcengine")
    arcengine_mod.FrameData = object
    arcengine_mod.GameAction = _GameAction
    arcengine_mod.GameState = _GameState
    arcengine_mod.ActionInput = _ActionInput
    sys.modules["arcengine"] = arcengine_mod


def load_my_agent_module():
    _install_agent_stubs()
    module_name = "benchmark_my_agent_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ReplayGame:
    def __init__(self):
        self.step = 0
        self._current_level_index = 0

    def set_level(self, level_idx: int):
        self._current_level_index = int(level_idx)

    def perform_action(self, action_input, raw=True):
        aid = action_input.id.value if hasattr(action_input.id, "value") else int(action_input.id)
        self.step += 1
        frame = np.full((64, 64), (self.step + aid) % 16, dtype=np.uint8)
        if aid == 6 and action_input.data:
            frame[int(action_input.data["y"]) % 64, int(action_input.data["x"]) % 64] = 14
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


def make_agent():
    mod = load_my_agent_module()
    agent = mod.MyAgent(
        card_id="bench",
        game_id="bench-game",
        agent_name="bench-agent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=None,
        tags=["bench"],
    )
    return agent


def make_semantic_frame():
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[20:24, 20:24] = 4
    frame[20:24, 36:40] = 14
    frame[30:34, 18:22] = 6
    frame[10:13, 48:52] = 11
    frame[40:44, 42:46] = 5
    return frame


def benchmark_case(name, fn, iterations, warmup):
    for _ in range(max(0, int(warmup))):
        fn()
    samples_ms = []
    for _ in range(max(1, int(iterations))):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    return {
        "name": name,
        "iterations": len(samples_ms),
        "mean_ms": round(mean_ms, 6),
        "median_ms": round(median_ms, 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
    }


def run_benchmarks(iterations=200, warmup=20):
    agent = make_agent()
    frame = make_semantic_frame()
    frame_hash = agent._fast_frame_hash(frame)
    avail = [_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION6]
    avail_ids = agent._available_action_ids(avail)
    avail_summary = agent._availability_summary(avail_ids)
    blocked_click_coord = agent._blocked_click_coord(frame, frame_hash=frame_hash)
    wait_avail_ids = [2, 4, 5, 6]
    wait_avail_summary = agent._availability_summary(wait_avail_ids)

    def fake_detector(_grid):
        return {
            "components_per_value": {
                "4": [{"center": (21.5, 21.5), "cell_count": 16}],
                "14": [{"center": (21.5, 37.5), "cell_count": 16}],
                "6": [{"center": (31.5, 19.5), "cell_count": 16}],
                "11": [{"center": (11.5, 49.5), "cell_count": 12}],
                "5": [{"center": (41.5, 43.5), "cell_count": 16}],
            }
        }

    agent._semantic_detector = fake_detector
    agent._semantic_target_coord = (22, 38)
    agent._unproductive = 6
    agent._bfs = types.SimpleNamespace(
        game_cls=_ReplayGame,
        _action_priority={
            (1,): 1,
            (2,): 2,
            (4,): 3,
            (6, 37, 21): 5,
        },
        _action_key=lambda act_id, data: (int(act_id),) if not data else (int(act_id), int(data.get("x", -1)), int(data.get("y", -1))),
    )
    click_targets = agent._semantic_click_targets_compat(
        frame,
        limit=6,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
    )
    target_choice = agent._semantic_target_choice(
        frame,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
    )
    tensor = agent._encode_frame_tensor(frame)
    click_scale = agent._semantic_click_bonus_scale(
        frame,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
        target_choice=target_choice,
    )
    fallback_targets = agent._heuristic_click_fallback_targets(
        frame,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
    )
    scored_coords = tuple(click_targets + fallback_targets)
    semantic_click_bonus_map = agent._semantic_click_bonus_map(
        frame,
        limit=6,
        click_scale=click_scale,
        click_targets=click_targets,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
    )
    candidate_indices = agent._semantic_click_candidate_indices(
        frame,
        click_targets=click_targets,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
    )
    semantic_dirs = agent._semantic_direction_bonuses(
        frame,
        avail,
        avail_ids=avail_ids,
        frame_hash=frame_hash,
        avail_summary=avail_summary,
    )
    wait_bonus = agent._wait_recovery_bonus(
        frame,
        wait_avail_ids,
        blocked_click_coord=blocked_click_coord,
        frame_hash=frame_hash,
        avail_summary=wait_avail_summary,
    )
    blocked_direction = agent._blocked_direction_action_index(frame, frame_hash=frame_hash)
    repeat_direction_idx = agent._recent_direction_action_index(frame, frame_hash=frame_hash)
    repeat_click_idx = agent._recent_click_action_index(frame, frame_hash=frame_hash)
    blocked_click_idx = agent._blocked_click_action_index(frame, frame_hash=frame_hash)
    preferred_click_coord = agent._preferred_click_coord()
    continuity_scale = agent._semantic_continuity_scale()
    sample_click_coord = click_targets[0] if click_targets else (22, 38)
    semantic_components = agent._semantic_components(frame, frame_hash=frame_hash)
    semantic_logits = agent._semantic_exploration_logits(
        frame,
        avail,
        True,
        blocked_click_coord=blocked_click_coord,
        avail_ids=avail_ids,
        frame_hash=frame_hash,
        avail_summary=avail_summary,
    )
    direction_logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
    scored_logits = semantic_logits.clone()
    scored_logits[1] = 0.5
    if candidate_indices:
        scored_logits[int(candidate_indices[0])] = 0.9

    recent_direction_agent = make_agent()
    recent_direction_agent.pr = frame.copy()
    recent_direction_agent.ph = recent_direction_agent._fast_frame_hash(recent_direction_agent.pr)
    recent_direction_agent.pai = 1
    recent_direction_frame = frame.copy()
    recent_direction_frame[0, 0] = 1
    recent_direction_frame_hash = recent_direction_agent._fast_frame_hash(recent_direction_frame)

    recent_click_agent = make_agent()
    recent_click_agent.pr = frame.copy()
    recent_click_agent.ph = recent_click_agent._fast_frame_hash(recent_click_agent.pr)
    recent_click_agent.pai = 5 + 18 * recent_click_agent.G + 11
    recent_click_frame = frame.copy()
    recent_click_frame[18, 11] = 7
    recent_click_frame_hash = recent_click_agent._fast_frame_hash(recent_click_frame)

    blocked_direction_agent = make_agent()
    blocked_direction_agent.pr = frame.copy()
    blocked_direction_agent.ph = blocked_direction_agent._fast_frame_hash(blocked_direction_agent.pr)
    blocked_direction_agent.pai = 1
    blocked_direction_frame = frame.copy()
    blocked_direction_frame_hash = blocked_direction_agent._fast_frame_hash(blocked_direction_frame)

    blocked_click_agent = make_agent()
    blocked_click_agent.pr = frame.copy()
    blocked_click_agent.ph = blocked_click_agent._fast_frame_hash(blocked_click_agent.pr)
    blocked_click_agent.pai = 5 + 18 * blocked_click_agent.G + 11
    blocked_click_frame = frame.copy()
    blocked_click_frame_hash = blocked_click_agent._fast_frame_hash(blocked_click_frame)

    stale_wait_agent = make_agent()
    stale_wait_agent.pr = frame.copy()
    stale_wait_agent.ph = stale_wait_agent._fast_frame_hash(stale_wait_agent.pr)
    stale_wait_agent.pai = 4
    stale_wait_agent._unproductive = 8
    stale_wait_agent._remember_blocked_direction_index(1)
    stale_wait_agent._remember_blocked_direction_index(3)
    stale_wait_agent._semantic_detector = lambda grid: {"components_per_value": {}}
    stale_wait_frame = frame.copy()
    stale_wait_frame_hash = stale_wait_agent._fast_frame_hash(stale_wait_frame)
    stale_wait_avail_ids = [2, 4, 5]
    stale_wait_avail_summary = stale_wait_agent._availability_summary(stale_wait_avail_ids)

    warmup_agent = make_agent()
    warmup_agent._semantic_detector = fake_detector
    warmup_agent._semantic_target_coord = (22, 38)
    warmup_agent._unproductive = 0
    warmup_agent.la = 0
    warmup_agent._maybe_train = lambda *args, **kwargs: None
    warmup_frame_hash = warmup_agent._fast_frame_hash(frame)

    refresh_agent = make_agent()
    refresh_agent._semantic_detector = fake_detector
    refresh_agent._semantic_target_coord = None
    refresh_frame_hash = refresh_agent._fast_frame_hash(frame)
    refresh_blocked_click_coord = refresh_agent._blocked_click_coord(frame, frame_hash=refresh_frame_hash)
    refresh_target_choice = refresh_agent._semantic_target_choice(
        frame,
        blocked_click_coord=refresh_blocked_click_coord,
        frame_hash=refresh_frame_hash,
    )
    refresh_fallback_agent = make_agent()
    refresh_fallback_frame_hash = refresh_fallback_agent._fast_frame_hash(frame)
    refresh_fallback_agent._semantic_target_choice = lambda *args, **kwargs: None

    control_agent = make_agent()
    control_frame_hash = control_agent._fast_frame_hash(frame)
    control_tensor = control_agent._encode_frame_tensor(frame)
    control_agent._undo_avail = True
    control_agent._ckpt_hash = 1
    control_agent._unproductive = 31
    control_agent.pr = frame.copy()
    control_agent.ph = control_frame_hash
    control_agent.fhist.append(frame.copy())

    revisit_agent = make_agent()
    revisit_curr_hash = revisit_agent._fast_frame_hash(frame)
    revisit_prev_frame = frame.copy()
    revisit_prev_frame[0, 0] = 3
    revisit_prev_hash = revisit_agent._fast_frame_hash(revisit_prev_frame)
    revisit_agent.fhist.append(revisit_prev_frame.copy())
    revisit_agent.fhist.append(frame.copy())

    policy_agent = make_agent()
    policy_agent._semantic_detector = fake_detector
    policy_agent._semantic_target_coord = (22, 38)
    policy_agent._eps = 0.0
    policy_frame_hash = policy_agent._fast_frame_hash(frame)
    policy_avail_ids = policy_agent._available_action_ids(avail)
    policy_blocked_click_coord = policy_agent._blocked_click_coord(frame, frame_hash=policy_frame_hash)
    policy_tensor = policy_agent._encode_frame_tensor(frame)

    results = []
    results.append(benchmark_case(
        "availability_summary",
        lambda: agent._availability_summary(avail_ids),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "previous_frame_relation",
        lambda: agent._previous_frame_relation(
            frame,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_target_candidates",
        lambda: agent._semantic_target_candidates(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "recent_direction_progress_delta",
        lambda: agent._recent_direction_progress_delta(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_goal_distance",
        lambda: agent._semantic_goal_distance(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            target_choice=target_choice,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_target_choice",
        lambda: agent._semantic_target_choice(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_direct_click_choice",
        lambda: agent._semantic_direct_click_choice(
            frame,
            avail_ids=avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_continuity_scale",
        lambda: agent._semantic_continuity_scale(),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "preferred_click_continuity_active",
        lambda: agent._preferred_click_continuity_active(),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_click_bonus_scale",
        lambda: agent._semantic_click_bonus_scale(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            target_choice=target_choice,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "template_log_bias",
        lambda: agent._template_log_bias(),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_components",
        lambda: agent._semantic_components(
            frame,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "legal_action_mask",
        lambda: agent._legal_action_mask(
            direction_logits,
            avail,
            avail_ids=avail_ids,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "top_legal_policy_indices",
        lambda: agent._top_legal_policy_indices(
            scored_logits,
            avail_ids,
            limit=4,
            click_candidate_indices=candidate_indices,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "candidate_scores",
        lambda: agent._candidate_scores(
            scored_logits,
            candidate_indices[:4],
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "candidate_score_map",
        lambda: agent._candidate_score_map(
            scored_logits,
            candidate_indices[:4],
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_direction_bonuses",
        lambda: agent._semantic_direction_bonuses(
            frame,
            avail,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_candidate_action_indices",
        lambda: agent._semantic_candidate_action_indices(
            frame,
            True,
            avail=avail,
            direction_bonuses=semantic_dirs,
            click_targets=click_targets,
            click_candidate_indices=candidate_indices,
            blocked_click_coord=blocked_click_coord,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            wait_recovery_bonus=wait_bonus,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "click_targets_from_components",
        lambda: agent._click_targets_from_components(
            frame,
            semantic_components,
            preferred_click_coord,
            preferred_click_coord,
            blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "rank_click_target_coords",
        lambda: agent._rank_click_target_coords(
            frame,
            scored_coords,
            preferred_click_coord,
            blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_exploration_logits",
        lambda: agent._semantic_exploration_logits(
            frame,
            avail,
            True,
            blocked_click_coord=blocked_click_coord,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "sample_sparse_policy_indices",
        lambda: agent._sample_sparse_policy_indices(
            semantic_logits,
            avail_ids,
            candidate_indices,
            temp=1.25,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "has_click_frontier",
        lambda: agent._has_click_frontier(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_click_targets",
        lambda: agent._semantic_click_targets_compat(
            frame,
            limit=6,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "heuristic_click_fallback_targets",
        lambda: agent._heuristic_click_fallback_targets(
            frame,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_click_candidate_indices",
        lambda: agent._semantic_click_candidate_indices(
            frame,
            click_targets=click_targets,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "blocked_click_matches_coord",
        lambda: agent._blocked_click_matches_coord(
            frame,
            sample_click_coord,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "blocked_click_coord",
        lambda: agent._blocked_click_coord(
            frame,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "frame_matches_previous",
        lambda: blocked_direction_agent._frame_matches_previous(
            blocked_direction_frame,
            frame_hash=blocked_direction_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "frame_changed_since_previous",
        lambda: recent_direction_agent._frame_changed_since_previous(
            recent_direction_frame,
            frame_hash=recent_direction_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "recent_direction_action_index",
        lambda: recent_direction_agent._recent_direction_action_index(
            recent_direction_frame,
            frame_hash=recent_direction_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "recent_direction_axis",
        lambda: recent_direction_agent._recent_direction_axis(
            recent_direction_frame,
            frame_hash=recent_direction_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "recent_click_action_index",
        lambda: recent_click_agent._recent_click_action_index(
            recent_click_frame,
            frame_hash=recent_click_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "blocked_direction_action_index",
        lambda: blocked_direction_agent._blocked_direction_action_index(
            blocked_direction_frame,
            frame_hash=blocked_direction_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "blocked_click_action_index",
        lambda: blocked_click_agent._blocked_click_action_index(
            blocked_click_frame,
            frame_hash=blocked_click_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "preferred_direction_choice",
        lambda: agent._preferred_direction_choice(1, None, [2, 4, 6]),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "preferred_click_target_choice",
        lambda: agent._preferred_click_target_choice(click_targets, preferred_click_coord, 6),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "click_candidate_context_map",
        lambda: agent._click_candidate_context_map(
            frame,
            candidate_indices,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            preferred_click_coord=preferred_click_coord,
            semantic_click_bonus_map=semantic_click_bonus_map,
            repeat_click_idx=repeat_click_idx,
            blocked_click_idx=blocked_click_idx,
            continuity_scale=continuity_scale,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "recent_frame_revisit_penalty",
        lambda: revisit_agent._recent_frame_revisit_penalty(revisit_curr_hash, revisit_prev_hash),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "nearest_coord_within",
        lambda: agent._nearest_coord_within(click_targets + fallback_targets, preferred_click_coord, 2),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "prepend_nearest_preferred_coord",
        lambda: agent._prepend_nearest_preferred_coord(
            frame,
            scored_coords,
            [],
            preferred_click_coord,
            set(),
            len(scored_coords),
            blocked_click_coord=blocked_click_coord,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "append_unblocked_coords",
        lambda: agent._append_unblocked_coords(
            frame,
            scored_coords,
            [],
            set(),
            len(scored_coords),
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_click_bonus",
        lambda: agent._semantic_click_bonus(sample_click_coord, click_scale, click_targets),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "stale_wait_recovery",
        lambda: stale_wait_agent._stale_wait_recovery(stale_wait_frame),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "modeled_frontier_exhausted",
        lambda: stale_wait_agent._modeled_frontier_exhausted(
            stale_wait_frame,
            stale_wait_avail_ids,
            blocked_click_coord=None,
            frame_hash=stale_wait_frame_hash,
            avail_summary=stale_wait_avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "retry_blocked_direction_after_stale_wait",
        lambda: stale_wait_agent._retry_blocked_direction_after_stale_wait(
            stale_wait_frame,
            stale_wait_avail_ids,
            blocked_click_coord=None,
            frame_hash=stale_wait_frame_hash,
            avail_summary=stale_wait_avail_summary,
            blocked_direction=1,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "should_exit_warmup_early",
        lambda: stale_wait_agent._should_exit_warmup_early(
            stale_wait_frame,
            stale_wait_avail_ids,
            blocked_click_coord=None,
            frame_hash=stale_wait_frame_hash,
            avail_summary=stale_wait_avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_direction_action",
        lambda: agent._semantic_direction_action(
            frame,
            avail,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "heuristic_action",
        lambda: agent._heuristic(
            frame,
            avail,
            6,
            blocked_click_coord=blocked_click_coord,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "prime_warmup_action",
        lambda: warmup_agent._prime_warmup_action(
            frame,
            avail,
            frame_hash=warmup_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "refresh_semantic_target_coord_choice",
        lambda: refresh_agent._refresh_semantic_target_coord(
            frame,
            blocked_click_coord=refresh_blocked_click_coord,
            frame_hash=refresh_frame_hash,
            target_choice=refresh_target_choice,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "refresh_semantic_target_coord_fallback",
        lambda: refresh_fallback_agent._refresh_semantic_target_coord(
            frame,
            fallback_coord=(22, 38),
            blocked_click_coord=None,
            frame_hash=refresh_fallback_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "handle_non_modeled_availability",
        lambda: control_agent._handle_non_modeled_availability(
            control_tensor,
            frame,
            control_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "maybe_force_undo",
        lambda: control_agent._maybe_force_undo(
            control_tensor,
            frame,
            control_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "choose_policy_action_heuristic_path",
        lambda: policy_agent._choose_policy_action(
            policy_tensor,
            frame,
            avail,
            policy_avail_ids,
            policy_blocked_click_coord,
            frame_hash=policy_frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "bfs_click_priority_bonus",
        lambda: agent._bfs_click_priority_bonus(sample_click_coord),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "bfs_priority_bonus",
        lambda: agent._bfs_priority_bonus(4),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "direction_candidate_context_map",
        lambda: agent._direction_candidate_context_map(
            frame,
            [1, 3, 4],
            frame_hash=frame_hash,
            blocked_direction=blocked_direction,
            semantic_dirs=semantic_dirs,
            repeat_direction_idx=repeat_direction_idx,
            wait_recovery_bonus=wait_bonus,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "semantic_click_bonus_map",
        lambda: agent._semantic_click_bonus_map(
            frame,
            limit=6,
            click_scale=click_scale,
            click_targets=click_targets,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "heuristic_click_bonus_map",
        lambda: agent._heuristic_click_bonus_map(
            frame,
            limit=6,
            click_scale=click_scale,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            fallback_targets=fallback_targets,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "wait_recovery_bonus",
        lambda: agent._wait_recovery_bonus(
            frame,
            wait_avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=wait_avail_summary,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "sample_semantic_exploration_sparse",
        lambda: agent._sample_semantic_exploration_sparse(
            frame,
            avail,
            avail_ids=avail_ids,
            blocked_click_coord=blocked_click_coord,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
            temp=1.25,
        ),
        iterations,
        warmup,
    ))
    results.append(benchmark_case(
        "engine_action_input_plain",
        lambda: agent._engine_action_input(1),
        iterations * 5,
        warmup,
    ))
    results.append(benchmark_case(
        "make_replay_game_and_frame",
        lambda: agent._make_replay_game_and_frame(0),
        iterations,
        warmup,
    ))
    return results


def main():
    parser = argparse.ArgumentParser(description="Deterministic local microbenchmarks for agent/my_agent.py hot paths.")
    parser.add_argument("--iterations", type=int, default=200, help="Measured iterations per benchmark case.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per benchmark case.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a text table.")
    args = parser.parse_args()

    results = run_benchmarks(iterations=args.iterations, warmup=args.warmup)
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"{'benchmark':34} {'mean_ms':>10} {'median_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    for item in results:
        print(
            f"{item['name']:34} "
            f"{item['mean_ms']:10.6f} "
            f"{item['median_ms']:10.6f} "
            f"{item['min_ms']:10.6f} "
            f"{item['max_ms']:10.6f}"
        )


if __name__ == "__main__":
    main()
