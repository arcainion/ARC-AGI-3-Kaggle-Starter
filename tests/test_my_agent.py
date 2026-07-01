from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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
        try:
            template = valid_actions[int(value)]
        except KeyError as exc:
            raise ValueError(f"No GameAction with id {value}") from exc
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
        self.game_id = kwargs.get("game_id", "unit-test")
        self.agent_name = kwargs.get("agent_name", "unit-test")
        self.ROOT_URL = kwargs.get("ROOT_URL", "http://localhost")
        self.record = kwargs.get("record", False)
        self.arc_env = kwargs.get("arc_env")
        self.tags = kwargs.get("tags", [])
        self.frames = []
        self.guid = None
        self.is_playback = False
        self.recorder = None
        self.action_counter = 0


def _load_my_agent_module():
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

    module_name = "test_my_agent_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_frame(fill: int = 0, *, state=_GameState.PLAYING, actions=None, levels=0):
    frame = np.full((64, 64), fill, dtype=np.uint8)
    return types.SimpleNamespace(
        frame=[frame],
        state=state,
        levels_completed=levels,
        score=levels,
        available_actions=list(actions or []),
    )


class _DummyNet:
    def eval(self):
        return self


class _FixedLogitNet(_DummyNet):
    def __init__(self, logits, device):
        import torch

        self._logits = torch.as_tensor(logits, dtype=torch.float32, device=device)

    def __call__(self, x, *args, **kwargs):
        return self._logits.unsqueeze(0)

    def forward_actions(self, x, *args, **kwargs):
        return self._logits[:5].unsqueeze(0)


class _ForwardOnlyNet(_DummyNet):
    def __init__(self, logits, device):
        import torch

        self._logits = torch.as_tensor(logits, dtype=torch.float32, device=device)
        self.forward_actions_calls = 0
        self.full_forward_calls = 0

    def __call__(self, x, *args, **kwargs):
        self.full_forward_calls += 1
        return self._logits.unsqueeze(0)

    def forward_actions(self, x, *args, **kwargs):
        self.forward_actions_calls += 1
        return self._logits.unsqueeze(0)


class _ReplayGame:
    def __init__(self):
        self.step = 0
        self._current_level_index = 0

    def set_level(self, level_idx: int):
        self._current_level_index = level_idx

    def perform_action(self, action_input, raw=True):
        self.step += 1
        frame = np.full((64, 64), self.step % 16, dtype=np.uint8)
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


class _HashGame:
    def __init__(self):
        self.visible = 1
        self.energy = 7
        self._private = 99


def _action_id(action_input):
    return action_input.id.value if hasattr(action_input.id, "value") else int(action_input.id)


class _ParallelScanGame:
    def __init__(self):
        self._current_level_index = 0
        self._available_actions = [1, 2, 5, 6]
        self.frame = np.zeros((64, 64), dtype=np.uint8)
        for x, y, value in ((1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5)):
            self.frame[y, x] = value

    def perform_action(self, action_input, raw=True):
        aid = _action_id(action_input)
        frame = self.frame.copy()
        if aid == 1:
            frame[0, 0] = 1
        elif aid == 2:
            frame[0, 1] = 2
        elif aid == 5:
            raise RuntimeError("direction boom")
        elif aid == 6:
            x = int(action_input.data["x"])
            y = int(action_input.data["y"])
            if (x, y) == (3, 3):
                raise RuntimeError("click boom")
            if (x, y) in {(1, 1), (2, 2)}:
                frame[10, 10] = 7
            elif (x, y) == (4, 4):
                frame[10, 11] = 8
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


class _ParallelSolveGame:
    def __init__(self):
        self._current_level_index = 0
        self.stage = 0
        self._available_actions = [1, 2, 3, 4]

    def set_level(self, level_idx):
        self._current_level_index = level_idx
        self.stage = 0

    def perform_action(self, action_input, raw=True):
        aid = _action_id(action_input)
        if aid == 0:
            self.stage = 0
            frame = np.zeros((64, 64), dtype=np.uint8)
        elif self.stage == 0 and aid == 1:
            self.stage = 1
            frame = np.full((64, 64), 1, dtype=np.uint8)
        elif self.stage == 1 and aid == 1:
            self._current_level_index += 1
            frame = np.full((64, 64), 9, dtype=np.uint8)
        else:
            frame = np.full((64, 64), aid % 16, dtype=np.uint8)
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


class _FirstWinningParallelGame:
    def __init__(self):
        self._current_level_index = 0
        self._available_actions = [1, 2, 3, 4]

    def set_level(self, level_idx):
        self._current_level_index = level_idx

    def perform_action(self, action_input, raw=True):
        aid = _action_id(action_input)
        if aid == 0:
            frame = np.zeros((64, 64), dtype=np.uint8)
        else:
            frame = np.full((64, 64), aid, dtype=np.uint8)
            if aid in (1, 4):
                self._current_level_index += 1
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


class _HiddenRetryParallelGame:
    def __init__(self):
        self._current_level_index = 0
        self.energy = 0
        self._available_actions = [1, 2, 3, 4]

    def set_level(self, level_idx):
        self._current_level_index = level_idx
        self.energy = 0

    def perform_action(self, action_input, raw=True):
        aid = _action_id(action_input)
        if aid == 0:
            self.energy = 0
        elif aid == 1:
            self.energy += 1
        elif aid == 3 and self.energy >= 1:
            self._current_level_index += 1
        frame = np.zeros((64, 64), dtype=np.uint8)
        return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)


class MyAgentUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_my_agent_module()

    def make_agent(self):
        return self.mod.MyAgent(
            card_id="unit",
            game_id="unit-game",
            agent_name="unit-agent",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=None,
            tags=["unit"],
        )

    def test_tensor_encoding_has_no_history_side_effects(self):
        agent = self.make_agent()
        frame = _make_frame(3)

        before = len(agent.fhist)
        tensor = agent._tensor(frame)

        self.assertEqual(len(agent.fhist), before)
        self.assertEqual(tuple(tensor.shape), (26, 64, 64))
        self.assertTrue(np.allclose(tensor[21:].cpu().numpy(), 0.0))

    def test_encode_frame_tensor_reuses_cached_full_tensor_for_same_frame(self):
        import torch

        agent = self.make_agent()
        frame = np.full((64, 64), 3, dtype=np.uint8)

        first = agent._encode_frame_tensor(frame)
        with mock.patch.object(agent, "_encode_static_frame_cpu", side_effect=AssertionError("should reuse cached static encoding")):
            second = agent._encode_frame_tensor(frame.copy())

        self.assertIsNotNone(agent._tensor_cached_full)
        self.assertEqual(tuple(second.shape), (26, 64, 64))
        self.assertTrue(torch.equal(first, second))

    def test_encode_frame_tensor_reuses_zero_tail_cache_across_new_frames(self):
        agent = self.make_agent()
        frame_a = np.full((64, 64), 3, dtype=np.uint8)
        frame_b = np.full((64, 64), 4, dtype=np.uint8)

        agent._encode_frame_tensor(frame_a)
        self.assertEqual(len(agent._tensor_zero_tail_cache), 1)
        first_tail = next(iter(agent._tensor_zero_tail_cache.values()))

        agent._encode_frame_tensor(frame_b)
        self.assertEqual(len(agent._tensor_zero_tail_cache), 1)
        second_tail = next(iter(agent._tensor_zero_tail_cache.values()))

        self.assertIs(first_tail, second_tail)
        self.assertEqual(tuple(second_tail.shape), (5, 64, 64))

    def test_try_bfs_solve_uses_frame_object_for_tensor_guidance(self):
        agent = self.make_agent()
        frame = _make_frame(1)
        sentinel = object()
        seen = {}

        class FakeBfs:
            def __init__(self):
                self.solutions = {}

            def solve_level(self, level_idx, prev_solution=None, timeout=None, net=None, frame_tensor=None):
                seen["frame_tensor"] = frame_tensor
                return None

        agent._bfs = FakeBfs()
        agent.net = object()

        def fake_tensor(arg):
            self.assertIs(arg, frame)
            return sentinel

        agent._tensor = fake_tensor
        agent._try_bfs_solve(0, lf=frame)

        self.assertIs(seen["frame_tensor"], sentinel)

    def test_try_bfs_solve_promotes_validated_cached_solution_to_agent_state(self):
        agent = self.make_agent()

        class CacheReplayGame:
            def __init__(self):
                self._current_level_index = 0

            def set_level(self, level_idx):
                self._current_level_index = level_idx

            def perform_action(self, action_input, raw=True):
                aid = action_input.id.value if hasattr(action_input.id, "value") else int(action_input.id)
                if aid == 1:
                    self._current_level_index += 1
                frame = np.full((64, 64), self._current_level_index, dtype=np.uint8)
                return types.SimpleNamespace(frame=[frame], levels_completed=self._current_level_index)

        class FakeBfs:
            def __init__(self):
                self.game_cls = CacheReplayGame
                self.solutions = {0: [(1, None), (2, None)]}

            def solve_level(self, *args, **kwargs):
                raise AssertionError("cached solution path should not call solve_level")

        agent._bfs = FakeBfs()
        agent._bfs_solution = None
        agent._bfs_step = 99

        result = agent._try_bfs_solve(0)

        self.assertEqual(result, [(1, None)])
        self.assertEqual(agent._bfs_solution, [(1, None)])
        self.assertEqual(agent._bfs_step, 0)

    def test_solution_injection_bc_filters_out_click_actions(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1, _GameAction.ACTION6], levels=0)
        captured = {}

        class FakeBfs:
            def __init__(self):
                self.game_cls = _ReplayGame
                self.solutions = {}
                self._last_effective_actions = None

        solution = [(1, None), (6, {"x": 4, "y": 7}), (2, None)]
        agent._bfs = FakeBfs()
        agent._bfs_tried = True
        agent._try_bfs_solve = lambda level_idx, lf=None: setattr(agent, "_bfs_solution", solution) or solution
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()
        agent._target_net = None
        agent._train = lambda: False

        def fake_bc(self_ref, raw_frames, action_indices, batch_size, epochs):
            captured["actions"] = list(action_indices)
            return 0.0

        original_bc = self.mod.MyAgent._bc_train_on_solution
        self.mod.MyAgent._bc_train_on_solution = fake_bc
        try:
            result = agent.choose_action([], frame)
        finally:
            self.mod.MyAgent._bc_train_on_solution = original_bc

        self.assertEqual(captured["actions"], [0, 1])
        self.assertEqual(result.value, 1)

    def test_error_fallback_uses_available_action(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION3])
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 3)
        self.assertIn("err:boom", result.reasoning)

    def test_reset_state_returns_reset_action(self):
        agent = self.make_agent()
        frame = _make_frame(0, state=_GameState.GAME_OVER, actions=[_GameAction.ACTION1])
        agent.cl = 0
        agent.pt = object()
        agent.pai = 2
        agent.pr = np.ones((64, 64), dtype=np.uint8)
        agent.ph = 123

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, _GameAction.RESET.value)
        self.assertEqual(result.reasoning, "reset")
        self.assertIsNone(agent.pt)
        self.assertIsNone(agent.pai)
        self.assertIsNone(agent.pr)
        self.assertIsNone(agent.ph)

    def test_bfs_solution_execution_preserves_payload_and_bookkeeping(self):
        agent = self.make_agent()
        frame = _make_frame(5, actions=[_GameAction.ACTION1, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent._bfs_solution = [(6, {"x": 11, "y": 13})]
        agent._bfs_step = 0

        result = agent.choose_action([], frame)
        result.reasoning = "mutated"
        other = agent._fresh_action(6, {"x": 11, "y": 13})

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 11, "y": 13})
        self.assertIsNot(result, other)
        self.assertEqual(agent._bfs_step, 1)
        self.assertEqual(agent.action_counter, 1)
        self.assertEqual(agent.la, 1)
        self.assertEqual(len(agent.fhist), 1)
        self.assertEqual(agent.pai, 5 + 13 * agent.G + 11)
        self.assertIsNotNone(agent.pt)
        self.assertEqual(agent.ph, agent._fast_frame_hash(frame.frame[-1]))
        self.assertTrue(np.array_equal(agent.pr, frame.frame[-1]))
        self.assertIs(agent.pr, agent.fhist[0])
        self.assertEqual(result.reasoning, "mutated")

    def test_bfs_solution_execution_refreshes_semantic_target_coord(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent._bfs_solution = [(1, None)]
        agent._bfs_step = 0

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(agent._semantic_target_coord, (20, 28))

    def test_bfs_click_execution_preserves_semantic_target_coord_without_player(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent._bfs_solution = [(6, {"x": 11, "y": 13})]
        agent._bfs_step = 0
        agent._semantic_detector = lambda grid: {"components_per_value": {"14": [{"center": (13.0, 11.0), "cell_count": 6}]}}

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(agent._semantic_target_coord, (13, 11))

    def test_refresh_semantic_target_coord_discards_blocked_click_fallback(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._semantic_target_coord = (30, 40)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        agent._refresh_semantic_target_coord(frame, fallback_coord=(19, 12))

        self.assertIsNone(agent._semantic_target_coord)

    def test_refresh_semantic_target_coord_discards_recent_blocked_click_fallback_when_current_block_differs(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._semantic_target_coord = (30, 40)
        agent._remember_blocked_click_coord((19, 12))

        agent._refresh_semantic_target_coord(
            frame,
            fallback_coord=(19, 12),
            blocked_click_coord=(0, 0),
            frame_hash=agent._fast_frame_hash(frame),
        )

        self.assertIsNone(agent._semantic_target_coord)

    def test_refresh_semantic_target_coord_uses_supplied_target_choice(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        target_choice = {
            "target_y": 22.0,
            "target_x": 40.0,
        }

        with mock.patch.object(agent, "_semantic_target_choice", side_effect=AssertionError("should use supplied target choice")):
            agent._refresh_semantic_target_coord(
                frame,
                frame_hash=agent._fast_frame_hash(frame),
                target_choice=target_choice,
            )

        self.assertEqual(agent._semantic_target_coord, (22, 40))

    def test_live_bfs_action_can_receive_level_completion_credit(self):
        agent = self.make_agent()
        start = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        next_level = _make_frame(1, actions=[_GameAction.ACTION1], levels=1)
        agent.cl = 0
        agent._bfs_solution = [(1, None)]
        agent._bfs_step = 0
        agent._bfs_tried = True
        agent._bfs = None
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()

        first = agent.choose_action([], start)
        second = agent.choose_action([], next_level)

        self.assertEqual(first.value, 1)
        self.assertEqual(second.value, 1)
        self.assertTrue(agent.buf_rewards)
        self.assertGreaterEqual(float(agent.buf_rewards[0]), 15.0)

    def test_wd_without_network_falls_back_to_heuristic_instead_of_error(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION3, _GameAction.ACTION4], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 3)
        self.assertEqual(result.reasoning, "cnn:a3")

    def test_level_change_resets_epsilon_and_schedule_when_bfs_fails(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = -1
        agent._bfs_tried = True
        agent._bfs = None
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()
        agent._eps = 0.03
        agent._eps_steps = 9999

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(agent._eps, 0.15)
        self.assertEqual(agent._eps_steps, 0)

    def test_level_change_preserves_epsilon_schedule_when_bfs_succeeds(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = -1
        agent._bfs_tried = True
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()
        agent._eps = 0.03
        agent._eps_steps = 9999

        class FakeBfs:
            def __init__(self):
                self.solutions = {}
                self.game_cls = _ReplayGame

        agent._bfs = FakeBfs()
        agent._try_bfs_solve = lambda level_idx, lf=None: setattr(agent, "_bfs_solution", [(1, None)]) or [(1, None)]

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(result.reasoning, "bfs:1/1")
        self.assertEqual(agent._eps, 0.03)
        self.assertEqual(agent._eps_steps, 9999)

    def test_level_change_restores_missing_optimizer_scheduler_and_target_net(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = -1
        agent._bfs_tried = True
        agent._bfs = None
        agent.net = torch.nn.Linear(1, 1)
        agent.opt = None
        agent.scheduler = None
        agent._target_net = None
        agent._make_optimizer = lambda: "OPT"
        agent._make_scheduler = lambda: "SCHED"

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(agent.opt, "OPT")
        self.assertEqual(agent.scheduler, "SCHED")
        self.assertIsNotNone(agent._target_net)
        self.assertIsNot(agent._target_net, agent.net)
        self.assertFalse(agent._target_net.training)

    def test_undo_only_availability_never_emits_modeled_action(self):
        agent = self.make_agent()
        frame = _make_frame(7, actions=[_GameAction.ACTION7], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo-only")
        self.assertIsNone(agent.pai)
        self.assertEqual(agent.la, 1)
        self.assertEqual(agent.ph, agent._fast_frame_hash(frame.frame[-1]))
        self.assertTrue(np.array_equal(agent.pr, frame.frame[-1]))

    def test_level_change_replaces_stale_scheduler_when_optimizer_is_rebuilt(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = -1
        agent._bfs_tried = True
        agent._bfs = None
        agent.net = torch.nn.Linear(1, 1)
        agent.opt = None
        agent.scheduler = "STALE"
        agent._target_net = None
        agent._make_optimizer = lambda: "OPT"
        agent._make_scheduler = lambda: "NEW_SCHED"

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(agent.opt, "OPT")
        self.assertEqual(agent.scheduler, "NEW_SCHED")

    def test_reset_only_availability_returns_no_action_reset(self):
        agent = self.make_agent()
        frame = _make_frame(9, actions=[_GameAction.RESET], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None
        agent.pt = object()
        agent.pai = 3
        agent.pr = np.ones((64, 64), dtype=np.uint8)
        agent.ph = 123

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, _GameAction.RESET.value)
        self.assertEqual(result.reasoning, "no-action")
        self.assertEqual(agent.action_counter, 1)
        self.assertEqual(agent.la, 0)
        self.assertIsNone(agent.pt)
        self.assertIsNone(agent.pai)
        self.assertIsNone(agent.pr)
        self.assertIsNone(agent.ph)

    def test_error_fallback_uses_click_then_undo_then_reset(self):
        click_agent = self.make_agent()
        click_frame = _make_frame(0, actions=[_GameAction.ACTION6])
        click_agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        click_result = click_agent.choose_action([], click_frame)
        self.assertEqual(click_result.value, 6)
        self.assertEqual(click_result.data, {"x": 32, "y": 32})
        self.assertIn("err:boom", click_result.reasoning)
        self.assertEqual(click_agent.pai, 5 + 32 * click_agent.G + 32)
        self.assertIsNotNone(click_agent.pt)
        self.assertTrue(np.array_equal(click_agent.pr, click_frame.frame[-1]))

        undo_agent = self.make_agent()
        undo_frame = _make_frame(0, actions=[_GameAction.ACTION7])
        undo_agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        undo_result = undo_agent.choose_action([], undo_frame)
        self.assertEqual(undo_result.value, 7)
        self.assertIn("err:boom", undo_result.reasoning)
        self.assertIsNone(undo_agent.pai)
        self.assertIsNotNone(undo_agent.pt)
        self.assertTrue(np.array_equal(undo_agent.pr, undo_frame.frame[-1]))

        reset_agent = self.make_agent()
        reset_frame = _make_frame(0, actions=[])
        reset_agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        reset_result = reset_agent.choose_action([], reset_frame)
        self.assertEqual(reset_result.value, _GameAction.RESET.value)
        self.assertIn("err:boom", reset_result.reasoning)
        self.assertIsNone(reset_agent.pt)
        self.assertIsNone(reset_agent.pai)
        self.assertIsNone(reset_agent.pr)
        self.assertIsNone(reset_agent.ph)

    def test_error_fallback_reset_clears_semantic_target_coord(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[])
        agent._semantic_target_coord = (10, 20)
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, _GameAction.RESET.value)
        self.assertIsNone(agent._semantic_target_coord)

    def test_error_fallback_click_prefers_semantic_target(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6])

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (10.0, 20.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 20, "y": 10})
        self.assertEqual(agent.pai, 5 + 10 * agent.G + 20)
        self.assertEqual(agent._semantic_target_coord, (10, 20))

    def test_error_fallback_click_avoids_repeating_blocked_center_default(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6])
        agent.pai = 5 + 32 * agent.G + 32
        agent.pr = frame.frame[-1].copy()
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 35, "y": 32})
        self.assertEqual(agent.pai, 5 + 32 * agent.G + 35)

    def test_error_fallback_direction_refreshes_previous_action_bookkeeping(self):
        agent = self.make_agent()
        frame = _make_frame(3, actions=[_GameAction.ACTION3], levels=0)
        agent.pt = object()
        agent.pai = 0
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._semantic_target_coord = (10, 20)
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 3)
        self.assertEqual(agent.pai, 2)
        self.assertIsNotNone(agent.pt)
        self.assertTrue(np.array_equal(agent.pr, frame.frame[-1]))
        self.assertEqual(agent.ph, agent._fast_frame_hash(frame.frame[-1]))
        self.assertIsNone(agent._semantic_target_coord)

    def test_error_fallback_direction_refreshes_semantic_target_from_current_frame(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION3], levels=0)
        frame.frame[-1][20:22, 20:22] = 4
        frame.frame[-1][20:22, 24:26] = 6
        agent._semantic_target_coord = (10, 20)
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 3)
        self.assertEqual(agent._semantic_target_coord, (20, 24))

    def test_error_fallback_direction_uses_semantic_direction_over_first_available_action(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        frame.frame[-1][20:22, 20:22] = 4
        frame.frame[-1][20:22, 28:30] = 14
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_prefers_semantic_click_over_first_available_direction(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        frame.frame[-1][20:22, 20:22] = 4
        frame.frame[-1][20:22, 28:30] = 14
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 28, "y": 20})
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_commits_to_direct_clickable_target_before_direction(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        frame.frame[-1][20:22, 20:22] = 4
        frame.frame[-1][20:22, 22:24] = 14
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 22, "y": 20})
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_avoids_known_blocked_direction_when_alternative_exists(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        agent.pai = 1
        agent.pr = frame.frame[-1].copy()
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_prefers_wait_over_only_blocked_direction(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION5], levels=0)
        agent.pai = 1
        agent.pr = frame.frame[-1].copy()
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 5)
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_retries_blocked_direction_after_stale_wait(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION5], levels=0)
        agent._unproductive = 7
        agent.pt = object()
        agent.pai = 4
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._remember_blocked_direction_index(1)
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_prefers_undo_when_frontier_is_exhausted(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION7], levels=0)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._ckpt_hash = 99
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "err:boom")

    def test_error_fallback_prefers_undo_after_stale_wait_recovery(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION5, _GameAction.ACTION7], levels=0)
        agent._unproductive = 7
        agent.pt = object()
        agent.pai = 4
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._ckpt_hash = 99
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "err:boom")

    def test_repeat_action_shortcut_reuses_last_direction(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION1, _GameAction.ACTION2], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "repeat:a2")
        self.assertEqual(agent.la, 1)
        self.assertEqual(len(agent.fhist), 1)
        self.assertTrue(np.array_equal(agent.pr, current.frame[-1]))
        self.assertIs(agent.pr, agent.fhist[0])

    def test_repeat_action_shortcut_skips_recent_frame_revisit_loop(self):
        agent = self.make_agent()
        raw = np.full((64, 64), 2, dtype=np.uint8)
        prev = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 1
        agent.pr = prev.copy()
        agent.ph = agent._fast_frame_hash(prev)
        agent.fhist.extend([raw.copy(), prev.copy()])

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent._try_repeat_direction_action(
                raw,
                [_GameAction.ACTION1, _GameAction.ACTION2],
                [1, 2],
                object(),
                agent._fast_frame_hash(raw),
            )

        self.assertIsNone(result)

    def test_repeat_action_shortcut_skips_when_recent_direction_increased_semantic_distance(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.zeros((64, 64), dtype=np.uint8)
        prev[20, 20] = 4
        curr[20, 21] = 4
        prev[20, 18] = 6
        curr[20, 18] = 6

        def fake_detector(grid):
            grid = np.asarray(grid, dtype=np.uint8)
            components = {}
            for color in (4, 6):
                ys, xs = np.where(grid == color)
                if ys.size:
                    components[str(color)] = [{
                        "center": (float(ys.mean()), float(xs.mean())),
                        "cell_count": int(ys.size),
                    }]
            return {"components_per_value": components}

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = prev
        agent.ph = agent._fast_frame_hash(prev)

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent._try_repeat_direction_action(
                curr,
                [_GameAction.ACTION3, _GameAction.ACTION4],
                [3, 4],
                object(),
                agent._fast_frame_hash(curr),
            )

        self.assertIsNone(result)

    def test_repeat_action_shortcut_refreshes_semantic_target_coord(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION1, _GameAction.ACTION2], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (22.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 2)
        self.assertEqual(agent._semantic_target_coord, (22, 23))

    def test_repeat_action_shortcut_without_click_availability_stays_lazy_on_click_only_helpers(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = False

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(agent, "_semantic_direct_click_choice", side_effect=AssertionError("should stay lazy without click")):
            with mock.patch.object(self.mod.random, "random", return_value=0.0):
                result = agent.choose_action([], current)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_preferred_click_coord_drops_stale_target_after_unproductive_streak(self):
        agent = self.make_agent()
        agent._semantic_target_coord = (18, 46)
        agent._unproductive = 8

        self.assertIsNone(agent._preferred_click_coord())

    def test_repeat_action_shortcut_yields_to_new_semantic_direction(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = False

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_repeat_action_shortcut_yields_to_semantic_click_objective(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent.la = 4
        agent._wd = False

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (10.0, 20.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 20, "y": 10})

    def test_repeat_action_shortcut_can_continue_semantic_direction_even_when_click_exists(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 3
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                    "5": [{"center": (10.0, 20.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "repeat:a4")

    def test_repeat_action_shortcut_ignores_semantic_clicks_when_click_is_illegal(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (10.0, 20.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "repeat:a2")

    def test_repeat_action_shortcut_ignores_distant_click_target_when_direction_unavailable(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "repeat:a2")

    def test_repeat_action_shortcut_ignores_jittered_click_match_for_tracked_target(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True
        agent._semantic_target_coord = (18, 34)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (19.0, 35.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "repeat:a2")

    def test_repeat_action_shortcut_yields_to_exact_tracked_click_target(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent.la = 4
        agent._wd = False
        agent._semantic_target_coord = (10, 20)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (10.0, 20.0), "cell_count": 6}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 20, "y": 10})

    def test_repeat_action_shortcut_yields_to_direct_click_choice_even_when_click_matches_preferred(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION2, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True
        agent._semantic_target_coord = (19, 35)
        agent._semantic_direct_click_choice = lambda *args, **kwargs: (5, (19, 35))
        agent._semantic_direction_action = lambda *args, **kwargs: None
        agent._semantic_click_targets_compat = lambda *args, **kwargs: [(19, 35)]

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 35, "y": 19})
        self.assertEqual(result.reasoning, "cnn:c(35,19)")

    def test_try_repeat_direction_action_returns_none_when_direct_click_choice_exists(self):
        agent = self.make_agent()
        raw = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 1
        agent.pr = np.full((64, 64), 2, dtype=np.uint8)
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._semantic_direct_click_choice = lambda *args, **kwargs: (5, (19, 35))

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent._try_repeat_direction_action(
                raw,
                [_GameAction.ACTION2, _GameAction.ACTION6],
                [2, 6],
                object(),
                agent._fast_frame_hash(raw),
            )

        self.assertIsNone(result)

    def test_no_click_path_uses_forward_actions_head(self):
        import torch

        agent = self.make_agent()
        logits = torch.tensor([-5.0, 3.0, -2.0, -3.0, -4.0], dtype=torch.float32)
        net = _ForwardOnlyNet(logits, agent.device)
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = net
        agent._bfs = None

        with mock.patch.object(self.mod.random, "random", return_value=1.0):
            result = agent.choose_action([], frame)

        self.assertEqual(result.value, 2)
        self.assertEqual(net.forward_actions_calls, 1)
        self.assertEqual(net.full_forward_calls, 0)

    def test_undo_shortcut_returns_action7(self):
        agent = self.make_agent()
        frame = _make_frame(4, actions=[_GameAction.ACTION7], levels=0)

        agent.cl = 0
        agent._wd = True
        agent._unproductive = 30
        agent._ckpt_hash = 99
        agent._semantic_target_coord = (20, 30)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo-only")
        self.assertIsNone(agent.pai)
        self.assertEqual(agent.la, 1)
        self.assertEqual(agent._unproductive, 30)
        self.assertIsNone(agent._semantic_target_coord)
        self.assertTrue(np.array_equal(agent.pr, frame.frame[-1]))

    def test_undo_shortcut_still_fires_when_modeled_actions_exist(self):
        agent = self.make_agent()
        frame = _make_frame(4, actions=[_GameAction.ACTION1, _GameAction.ACTION7], levels=0)

        agent.cl = 0
        agent._wd = True
        agent._unproductive = 30
        agent._ckpt_hash = 99
        agent._semantic_target_coord = (20, 30)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo")
        self.assertIsNone(agent.pai)
        self.assertEqual(agent.la, 1)
        self.assertEqual(agent._unproductive, 0)
        self.assertIsNone(agent._semantic_target_coord)
        self.assertTrue(np.array_equal(agent.pr, frame.frame[-1]))

    def test_maybe_force_undo_skips_revisit_penalty_when_highly_unproductive(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        tensor = object()

        agent._undo_avail = True
        agent._ckpt_hash = 99
        agent._unproductive = 30

        with mock.patch.object(
                agent,
                "_recent_frame_revisit_penalty",
                side_effect=AssertionError("high unproductive undo should skip revisit penalty")):
            result = agent._maybe_force_undo(tensor, frame, frame_hash)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo")
        self.assertEqual(agent._unproductive, 0)

    def test_undo_shortcut_fires_early_on_recent_frame_revisit_loop(self):
        agent = self.make_agent()
        frame = _make_frame(4, actions=[_GameAction.ACTION1, _GameAction.ACTION7], levels=0)

        agent.cl = 0
        agent._wd = True
        agent._unproductive = 2
        agent._ckpt_hash = 99
        agent.pt = object()
        agent.pai = 0
        previous = np.zeros((64, 64), dtype=np.uint8)
        agent.pr = previous.copy()
        agent.ph = agent._fast_frame_hash(previous)
        agent.fhist.extend([frame.frame[-1].copy(), previous.copy()])

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo")
        self.assertEqual(agent._unproductive, 0)

    def test_undo_transition_is_not_written_into_replay_or_aem(self):
        agent = self.make_agent()
        current = _make_frame(1, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None
        agent.pt = object()
        agent.pai = None
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0

        result = agent.choose_action([], current)

        self.assertEqual(result.value, 1)
        self.assertEqual(len(agent.buf), 0)
        self.assertEqual(list(agent.buf_actions), [])
        self.assertEqual(list(agent._aem_actions), [])

    def test_undo_transition_still_updates_novelty_bookkeeping(self):
        agent = self.make_agent()
        current = _make_frame(1, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None
        agent.pt = object()
        agent.pai = None
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0

        agent.choose_action([], current)

        curr_h = agent._fast_frame_hash(current.frame[-1])
        self.assertIn(curr_h, agent._visited_hashes)
        self.assertEqual(agent._state_visit_counts[curr_h], 1)
        self.assertIsNotNone(agent._prev_objs)

    def test_repeat_shortcut_skips_illegal_repeated_direction(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        agent.ph = 0
        agent._wd = True
        agent.net = None

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            result = agent.choose_action([], current)

        self.assertEqual(result.value, 1)
        self.assertEqual(result.reasoning, "cnn:a1")

    def test_action_counter_advances_on_repeated_cnn_fallback_actions(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION3, _GameAction.ACTION4], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None

        first = agent.choose_action([], frame)
        second = agent.choose_action([], frame)

        self.assertEqual(first.value, 3)
        self.assertEqual(second.value, 4)
        self.assertEqual(agent.action_counter, 2)

    def test_sample_respects_available_direction_mask(self):
        import torch

        agent = self.make_agent()
        logits = torch.tensor([9.0, 1.5, 8.0, 2.5, 7.0], dtype=torch.float32, device=agent.device)
        avail = [_GameAction.ACTION2, _GameAction.ACTION4]

        with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([3], device=agent.device)):
            action_idx, coords = agent._sample(logits, avail=avail, temp=1.0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_sample_returns_click_coordinates_when_action6_available(self):
        import torch

        agent = self.make_agent()
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        click_y, click_x = 12, 34
        click_index = 5 + click_y * agent.G + click_x
        avail = [_GameAction.ACTION6]

        with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([click_index], device=agent.device)):
            action_idx, coords = agent._sample(logits, avail=avail, temp=1.0)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (click_y, click_x))

    def test_sample_applies_click_heatmap_without_shape_error(self):
        import torch

        agent = self.make_agent()
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        agent._wm = torch.ones((64, 64), dtype=torch.float32)
        agent._wm[8, 6] = 10.0
        avail = [_GameAction.ACTION6]

        with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([5 + 8 * agent.G + 6], device=agent.device)):
            action_idx, coords = agent._sample(logits, avail=avail, temp=1.0)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (8, 6))

    def test_sample_zero_mass_fallback_respects_legal_direction_mask(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((5,), -float("inf"), dtype=torch.float32, device=agent.device)
        avail = [_GameAction.ACTION2, _GameAction.ACTION4]

        with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([3], device=agent.device)):
            action_idx, coords = agent._sample(logits, avail=avail, temp=1.0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_sample_zero_mass_fallback_excludes_blocked_click_region(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((4101,), -1000.0, dtype=torch.float32, device=agent.device)
        safe_y, safe_x = 22, 40
        safe_idx = 5 + safe_y * agent.G + safe_x
        blocked_idx = 5 + 20 * agent.G + 23
        logits[blocked_idx] = -float("inf")
        avail = [_GameAction.ACTION6]

        with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([safe_idx], device=agent.device)):
            action_idx, coords = agent._sample(logits, avail=avail, temp=1.0)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (safe_y, safe_x))

    def test_legal_action_mask_keeps_directions_when_click_is_listed_first(self):
        import torch

        agent = self.make_agent()
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)

        mask = agent._legal_action_mask(logits, [_GameAction.ACTION6, _GameAction.ACTION3])

        self.assertEqual(mask[2].item(), 0.0)
        self.assertEqual(mask[5].item(), 0.0)
        self.assertTrue(torch.isneginf(mask[0]))

    def test_legal_action_mask_reuses_cached_tensor_for_same_availability(self):
        import torch

        agent = self.make_agent()
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        avail = [_GameAction.ACTION6, _GameAction.ACTION3]

        first = agent._legal_action_mask(logits, avail)
        second = agent._legal_action_mask(logits, avail)

        self.assertIs(first, second)

    def test_legal_modeled_action_count_matches_directional_and_click_legal_space(self):
        agent = self.make_agent()

        self.assertEqual(agent._legal_modeled_action_count(5, [2, 4]), 2)
        self.assertEqual(agent._legal_modeled_action_count(4101, [2, 4, 6]), 4098)
        self.assertEqual(agent._legal_modeled_action_count(4101, []), 4101)

    def test_top_legal_policy_indices_matches_dense_mask_topk(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((4101,), -20.0, dtype=torch.float32, device=agent.device)
        logits[1] = 4.0
        logits[3] = 3.0
        logits[5 + 30 * agent.G + 40] = 9.0
        logits[5 + 18 * agent.G + 18] = 8.5
        logits[5 + 10 * agent.G + 10] = 7.25
        avail_ids = [2, 4, 6]

        mask = agent._legal_action_mask(logits, None, avail_ids=avail_ids)
        dense_expected = (
            (logits + mask)
            .topk(5)
            .indices
            .detach()
            .cpu()
            .tolist()
        )
        sparse_actual = agent._top_legal_policy_indices(logits, avail_ids, 5)

        self.assertEqual(sparse_actual, [int(idx) for idx in dense_expected])

    def test_top_legal_policy_indices_can_use_sparse_click_shortlist(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((4101,), -20.0, dtype=torch.float32, device=agent.device)
        allowed_click = 5 + 22 * agent.G + 40
        unrelated_hot_click = 5 + 10 * agent.G + 10
        logits[0] = 0.4
        logits[allowed_click] = 0.9
        logits[unrelated_hot_click] = 5.0

        top_indices = agent._top_legal_policy_indices(
            logits,
            [1, 6],
            2,
            click_candidate_indices=[allowed_click],
        )

        self.assertEqual(top_indices, [allowed_click, 0])

    def test_legal_direction_ids_reuse_cached_set_for_same_availability(self):
        agent = self.make_agent()

        first = agent._legal_direction_ids([2, 4, 6])
        second = agent._legal_direction_ids([2, 4, 6])

        self.assertIs(first, second)
        self.assertEqual(first, frozenset({2, 4}))

    def test_availability_summary_reuses_cached_flags_for_same_availability(self):
        agent = self.make_agent()

        first = agent._availability_summary([2, 4, 6, 7])
        second = agent._availability_summary([2, 4, 6, 7])

        self.assertIs(first, second)
        self.assertTrue(first["has_click"])
        self.assertTrue(first["has_undo"])
        self.assertTrue(first["has_modeled"])
        self.assertEqual(first["legal_dirs"], frozenset({2, 4}))

    def test_cnn_rescoring_honors_directions_even_when_click_appears_first(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6, _GameAction.ACTION3], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[2] = 8.0
        logits[5 + 32 * agent.G + 32] = 7.0

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = types.SimpleNamespace(
            _action_key=lambda act_id, data: (act_id, None if not data else (data.get("x"), data.get("y"))),
            _action_priority={},
        )
        agent._wm = np.ones((64, 64), dtype=np.float32)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 3)
        self.assertEqual(result.reasoning, "cnn:a3")

    def test_is_done_true_for_win_and_false_while_playing(self):
        agent = self.make_agent()
        win_frame = _make_frame(0, state=_GameState.WIN)
        play_frame = _make_frame(0, state=_GameState.PLAYING)

        self.assertTrue(agent.is_done([], win_frame))
        self.assertFalse(agent.is_done([], play_frame))

    def test_reward_tracks_visited_hashes_and_visit_counts(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.ones((64, 64), dtype=np.uint8)

        reward_first = agent._reward(prev, curr, "prev", "curr", changed=True, curr_objs=[], move_bonus=0.0, moved=0)
        reward_second = agent._reward(prev, curr, "prev", "curr", changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        self.assertGreater(reward_first, reward_second)
        self.assertIn("curr", agent._visited_hashes)
        self.assertEqual(agent._state_visit_counts["curr"], 2)

    def test_semantic_target_choice_can_prefer_nearer_useful_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        choice = agent._semantic_target_choice(frame)
        dist = agent._semantic_goal_distance(frame)

        self.assertEqual(choice["priority"], 1)
        self.assertEqual(choice["distance"], 3.0)
        self.assertEqual(dist, 3.0)

    def test_semantic_target_choice_prefers_previous_target_when_scores_are_close(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (19.0, 24.0), "cell_count": 4},
                        {"center": (21.0, 24.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (21, 24)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (21, 24))

    def test_heuristic_commits_to_direct_clickable_target_before_walking(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 22, "y": 20})
        self.assertEqual(result.reasoning, "cnn:c(22,20)")

    def test_heuristic_commits_to_click_target_without_player_component(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (20.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 22, "y": 20})
        self.assertEqual(result.reasoning, "cnn:c(22,20)")

    def test_semantic_direct_click_choice_skips_preferred_click_lookup_for_distant_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._preferred_click_coord = lambda: (_ for _ in ()).throw(AssertionError("preferred click lookup should be lazy"))
        agent._preferred_click_continuity_active = lambda: (_ for _ in ()).throw(AssertionError("continuity lookup should be lazy"))

        choice = agent._semantic_direct_click_choice(frame, [_GameAction.ACTION6])

        self.assertIsNone(choice)

    def test_policy_commits_to_direct_clickable_target_before_direction_logit(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[3] = 9.0

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = types.SimpleNamespace(
            _action_key=lambda act_id, data: (act_id, None if not data else (data.get("x"), data.get("y"))),
            _action_priority={},
        )

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 22, "y": 20})
        self.assertEqual(result.reasoning, "cnn:c(22,20)")

    def test_policy_commits_to_click_target_without_player_component(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[3] = 9.0

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (20.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = types.SimpleNamespace(
            _action_key=lambda act_id, data: (act_id, None if not data else (data.get("x"), data.get("y"))),
            _action_priority={},
        )

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 22, "y": 20})
        self.assertEqual(result.reasoning, "cnn:c(22,20)")

    def test_heuristic_commits_to_preferred_click_target_without_player_component(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        agent._semantic_target_coord = (32, 42)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [
                        {"center": (0.0, 0.0), "cell_count": 4},
                        {"center": (32.0, 42.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})
        self.assertEqual(result.reasoning, "cnn:c(42,32)")

    def test_direct_click_choice_no_player_drops_preferred_target_after_unproductive_streak(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_target_coord = (32, 42)
        agent._unproductive = 8
        agent._semantic_target_choice = lambda *args, **kwargs: None
        agent._semantic_components = lambda *args, **kwargs: {"14": [{"center": (0.0, 0.0), "cell_count": 4}]}
        agent._semantic_click_targets_compat = lambda *args, **kwargs: [(0, 0)]
        agent._heuristic_click_fallback_targets = lambda *args, **kwargs: []

        choice = agent._semantic_direct_click_choice(frame, avail_ids=[6], frame_hash=agent._fast_frame_hash(frame))

        self.assertEqual(choice, (5, (0, 0)))

    def test_heuristic_direct_click_without_player_skips_recent_blocked_preferred_target(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        agent.cl = 0
        agent._semantic_target_coord = (32, 42)
        agent._remember_blocked_click_coord((32, 42))

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [
                        {"center": (0.0, 0.0), "cell_count": 4},
                        {"center": (32.0, 42.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 0, "y": 0})
        self.assertEqual(result.reasoning, "cnn:c(0,0)")

    def test_semantic_direct_click_choice_skips_recent_blocked_preferred_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_target_coord = (20, 22)
        agent._remember_blocked_click_coord((20, 22))
        agent._semantic_target_choice = lambda *args, **kwargs: {
            "priority": 1,
            "distance": 2.0,
            "target_y": 20.0,
            "target_x": 23.0,
        }
        agent._semantic_click_targets_compat = lambda *args, **kwargs: [(20, 24)]

        choice = agent._semantic_direct_click_choice(frame, avail_ids=[6], frame_hash=agent._fast_frame_hash(frame))

        self.assertEqual(choice, (5, (20, 24)))

    def test_semantic_target_choice_skips_nearby_blocked_click_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (19.0, 12.0), "cell_count": 4},
                        {"center": (22.0, 40.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (22, 40))

    def test_semantic_target_choice_skips_recent_blocked_history_when_current_block_differs(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (19.0, 12.0), "cell_count": 4},
                        {"center": (22.0, 40.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._remember_blocked_click_coord((19, 12))

        choice = agent._semantic_target_choice(
            frame,
            blocked_click_coord=(0, 0),
            frame_hash=agent._fast_frame_hash(frame),
        )

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (22, 40))

    def test_semantic_target_choice_stays_on_previous_target_when_alternative_is_only_slightly_better(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 25.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (20, 40)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 40))
        self.assertEqual(choice["priority"], 0)

    def test_semantic_target_choice_prefers_target_along_recent_successful_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [
                        {"center": (20.0, 24.0), "cell_count": 4},
                        {"center": (24.0, 20.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = np.ones((64, 64), dtype=np.uint8)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 24))
        self.assertEqual(choice["momentum_bonus"], 0.12)

    def test_semantic_target_choice_penalizes_slightly_nearer_target_behind_recent_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [
                        {"center": (20.0, 17.0), "cell_count": 4},
                        {"center": (20.0, 24.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = np.ones((64, 64), dtype=np.uint8)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 24))
        self.assertEqual(choice["momentum_bonus"], 0.12)
        self.assertEqual(choice["counter_momentum_penalty"], 0.0)

    def test_semantic_target_choice_drops_momentum_bias_after_recent_direction_worsens_distance(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        frame = np.zeros((64, 64), dtype=np.uint8)
        prev[20, 20] = 4
        frame[20, 21] = 4

        def fake_detector(grid):
            grid = np.asarray(grid, dtype=np.uint8)
            player_x = 20.0 if grid[20, 20] == 4 else 21.0
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, player_x), "cell_count": 1}],
                    "6": [
                        {"center": (20.0, 18.0), "cell_count": 4},
                        {"center": (20.0, 25.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = prev
        agent.ph = agent._fast_frame_hash(prev)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 18))
        self.assertEqual(choice["momentum_bonus"], 0.0)
        self.assertEqual(choice["counter_momentum_penalty"], 0.0)

    def test_semantic_target_choice_breaks_equal_scores_toward_continuity(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (20, 29)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 28))
        self.assertGreater(choice["continuity_bonus"], 0.0)

    def test_semantic_target_choice_drops_continuity_after_unproductive_streak(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (20.0, 28.0), "cell_count": 4},
                        {"center": (20.0, 30.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (20, 30)
        agent._unproductive = 8

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 28))
        self.assertEqual(choice["continuity_bonus"], 0.0)

    def test_semantic_target_choice_softens_continuity_before_full_cutoff(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (20, 29)

        fresh_choice = agent._semantic_target_choice(frame)
        agent._unproductive = 6
        stale_choice = agent._semantic_target_choice(frame)

        self.assertGreater(fresh_choice["continuity_bonus"], stale_choice["continuity_bonus"])
        self.assertGreater(stale_choice["continuity_bonus"], 0.0)

    def test_semantic_target_choice_prefers_larger_equally_scored_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [
                        {"center": (20.0, 24.0), "cell_count": 1},
                        {"center": (20.0, 24.0), "cell_count": 9},
                    ],
                }
            }

        agent._semantic_detector = fake_detector

        choice = agent._semantic_target_choice(frame)

        self.assertEqual(choice["area"], 9)

    def test_semantic_click_bonus_scale_uses_supplied_target_choice(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        target_choice = {"distance": 8.0}

        with mock.patch.object(agent, "_semantic_target_choice", side_effect=AssertionError("should use supplied target choice")):
            scale = agent._semantic_click_bonus_scale(frame, target_choice=target_choice)

        self.assertAlmostEqual(scale, 0.5, places=5)

    def test_reward_boosts_semantic_progress_toward_target(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.ones((64, 64), dtype=np.uint8)

        def fake_goal_distance(frame):
            return 10.0 if frame is prev else 6.0

        agent._semantic_goal_distance = fake_goal_distance

        reward = agent._reward(prev, curr, "prev", "curr", changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        self.assertGreater(reward, 2.5)

    def test_reward_penalizes_semantic_retreat_from_target(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.ones((64, 64), dtype=np.uint8)

        def fake_goal_distance(frame):
            return 6.0 if frame is prev else 10.0

        agent._semantic_goal_distance = fake_goal_distance

        retreat = agent._reward(prev, curr, "prev", "curr", changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        agent2 = self.make_agent()
        agent2._semantic_goal_distance = lambda frame: None
        baseline = agent2._reward(prev, curr, "prev", "curr", changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        self.assertLess(retreat, baseline)

    def test_reward_penalizes_unchanged_state(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        reward = agent._reward(frame, frame, "same", "same", changed=False, curr_objs=[], move_bonus=0.0, moved=0)

        self.assertLess(reward, 0.3)
        self.assertEqual(agent._state_visit_counts["same"], 1)

    def test_reward_penalizes_recent_two_step_frame_revisit(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        older = np.full((64, 64), 2, dtype=np.uint8)
        curr = older.copy()
        agent.fhist.extend([older.copy(), prev.copy()])
        prev_h = agent._fast_frame_hash(prev)
        curr_h = agent._fast_frame_hash(curr)

        revisit = agent._reward(prev, curr, prev_h, curr_h, changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        baseline_agent = self.make_agent()
        baseline = baseline_agent._reward(prev, curr, prev_h, curr_h, changed=True, curr_objs=[], move_bonus=0.0, moved=0)

        self.assertLess(revisit, baseline)

    def test_recent_frame_revisit_penalty_skips_immediate_previous_frame_hash(self):
        agent = self.make_agent()
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr_h = agent._fast_frame_hash(prev)
        agent.fhist.append(prev.copy())

        penalty = agent._recent_frame_revisit_penalty(curr_h, curr_h)

        self.assertEqual(penalty, 0.0)

    def test_wait_recovery_bonus_uses_supplied_availability_summary(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._unproductive = 6
        agent._remember_blocked_direction_index(0)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(2)
        agent._remember_blocked_direction_index(3)
        agent.pai = 0
        agent.pr = frame.copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        avail_summary = {
            "has_click": False,
            "has_undo": False,
            "has_modeled": True,
            "legal_dirs": frozenset({1, 2, 3, 4}),
        }

        with mock.patch.object(agent, "_availability_summary", side_effect=AssertionError("should use supplied availability summary")):
            bonus = agent._wait_recovery_bonus(
                frame,
                [1, 2, 3, 4, 5],
                frame_hash=agent._fast_frame_hash(frame),
                avail_summary=avail_summary,
            )

        self.assertEqual(bonus, 0.3)

    def test_click_frontier_cache_is_reused_across_frontier_helpers(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._unproductive = 6
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)

        with mock.patch.object(agent, "_semantic_click_targets_compat", wraps=agent._semantic_click_targets_compat) as semantic_mock:
            self.assertEqual(
                agent._wait_recovery_bonus(frame, [2, 4, 5, 6], frame_hash=frame_hash),
                0.3,
            )
            self.assertTrue(
                agent._modeled_frontier_exhausted(frame, [2, 4, 6], frame_hash=frame_hash)
            )
            self.assertEqual(semantic_mock.call_count, 1)

    def test_choose_action_reward_uses_previous_hash_for_same_state(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1, _GameAction.ACTION2], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 1
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._wd = True

        captured = {}

        def fake_reward(prev_raw, curr_raw, prev_h, curr_h, **kwargs):
            captured["prev_h"] = prev_h
            captured["curr_h"] = curr_h
            return 0.0

        agent._reward = fake_reward
        agent._sample = lambda logits, avail=None, temp=1.0: (0, None)

        with mock.patch.object(self.mod.random, "random", return_value=1.0):
            agent.choose_action([], frame)

        self.assertEqual(captured["prev_h"], agent.ph)
        self.assertEqual(captured["curr_h"], agent._fast_frame_hash(frame.frame[-1]))

    def test_choose_action_progress_decays_blocked_histories_instead_of_clearing_them(self):
        agent = self.make_agent()
        previous = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        current = _make_frame(1, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent.pt = object()
        agent.pai = 0
        agent.pr = previous.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._remember_blocked_click_coord((10, 10))
        agent._remember_blocked_click_coord((20, 20))
        agent._remember_blocked_click_coord((30, 30))
        agent._remember_blocked_direction_index(0)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(2)
        agent._try_repeat_direction_action = lambda *args, **kwargs: None

        result = agent.choose_action([], current)

        self.assertEqual(result.value, 1)
        self.assertEqual(list(agent._blocked_click_history), [(20, 20), (30, 30)])
        self.assertEqual(list(agent._blocked_direction_history), [1, 2])

    def test_dedup_hit_still_updates_novelty_bookkeeping(self):
        agent = self.make_agent()
        current = _make_frame(2, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = 0
        agent._wd = True
        agent.net = None
        agent.pt = object()
        agent.pai = 0
        agent.pr = np.zeros((64, 64), dtype=np.uint8)
        prev_h = agent._fast_frame_hash(agent.pr)
        agent.ph = prev_h
        agent.buf_h = {(prev_h, 0)}

        result = agent.choose_action([], current)

        curr_h = agent._fast_frame_hash(current.frame[-1])
        self.assertEqual(result.value, 1)
        self.assertEqual(len(agent.buf), 0)
        self.assertIn(curr_h, agent._visited_hashes)
        self.assertEqual(agent._state_visit_counts[curr_h], 1)
        self.assertIsNotNone(agent._prev_objs)

    def test_adaptive_bfs_timeout_respects_level_caps(self):
        agent = self.make_agent()
        agent.start_time = self.mod.time.time()
        agent.total_time_budget = 1000.0
        agent.estimated_total_levels = 50

        early = agent._adaptive_bfs_timeout(0)
        mid = agent._adaptive_bfs_timeout(3)
        late = agent._adaptive_bfs_timeout(10)

        self.assertLessEqual(early, 60.0)
        self.assertLessEqual(mid, 40.0)
        self.assertLessEqual(late, 25.0)
        self.assertGreaterEqual(late, 10.0)

    def test_detect_template_masks_left_half_after_sparse_separator_column(self):
        agent = self.make_agent()
        frame = np.full((64, 64), 8, dtype=np.uint8)
        frame[:, 30] = 0
        frame[10, 30] = 8
        frame[50, 30] = 8
        agent._bg = 0

        mask = agent._detect_template(frame)

        self.assertEqual(mask.shape[0], 4096)
        mask_2d = mask.view(64, 64)
        self.assertTrue(np.allclose(mask_2d[:, :31].cpu().numpy(), 0.05))
        self.assertTrue(np.allclose(mask_2d[:, 31:].cpu().numpy(), 1.0))

    def test_detect_template_masks_upper_half_after_sparse_separator_row(self):
        agent = self.make_agent()
        frame = np.full((64, 64), 8, dtype=np.uint8)
        frame[29, :] = 0
        frame[29, 8] = 8
        frame[29, 55] = 8
        agent._bg = 0

        mask = agent._detect_template(frame)

        self.assertEqual(mask.shape[0], 4096)
        mask_2d = mask.view(64, 64)
        self.assertTrue(np.allclose(mask_2d[:30, :].cpu().numpy(), 0.05))
        self.assertTrue(np.allclose(mask_2d[30:, :].cpu().numpy(), 1.0))

    def test_heuristic_prefers_first_available_direction_in_opening(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        avail = [_GameAction.ACTION3, _GameAction.ACTION4, _GameAction.ACTION6]

        action_idx, coords = agent._heuristic(frame, avail, step=0)

        self.assertEqual(action_idx, 2)
        self.assertIsNone(coords)

    def test_heuristic_prefers_semantic_direction_even_in_opening(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4], step=0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_heuristic_reuses_semantic_target_choice_across_click_and_direction_checks(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        real_target_choice = agent._semantic_target_choice
        calls = []

        def counting_target_choice(*args, **kwargs):
            calls.append(1)
            return real_target_choice(*args, **kwargs)

        agent._semantic_target_choice = counting_target_choice

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4, _GameAction.ACTION6], step=0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)
        self.assertEqual(len(calls), 1)

    def test_semantic_direction_action_fresh_agent_skips_history_lookups(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        with mock.patch.object(agent, "_recent_direction_action_index", side_effect=AssertionError("fresh path should skip recent direction lookup")):
            with mock.patch.object(agent, "_blocked_click_coord", side_effect=AssertionError("fresh path should skip blocked click lookup")):
                with mock.patch.object(agent, "_blocked_direction_action_index", side_effect=AssertionError("fresh path should skip blocked direction lookup")):
                    with mock.patch.object(agent, "_retry_blocked_direction_after_stale_wait", side_effect=AssertionError("fresh path should skip stale-wait retry lookup")):
                        with mock.patch.object(agent, "_recent_direction_progress_delta", side_effect=AssertionError("fresh path should skip progress delta lookup")):
                            with mock.patch.object(agent, "_recent_direction_axis", side_effect=AssertionError("fresh path should skip recent axis lookup")):
                                action_idx, coords = agent._semantic_direction_action(
                                    frame,
                                    [_GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4],
                                )

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_semantic_direction_action_reuses_cached_best_direction_for_same_bonus_map(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        class _BonusMap:
            def __init__(self, items):
                self._items = list(items)
                self.calls = 0

            def __bool__(self):
                return bool(self._items)

            def items(self):
                self.calls += 1
                return list(self._items)

        bonuses = _BonusMap([(3, 0.45), (1, 0.18)])
        agent._semantic_direction_bonuses = lambda *args, **kwargs: bonuses

        first = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])
        second = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(first, (3, None))
        self.assertEqual(second, first)
        self.assertEqual(bonuses.calls, 1)

    def test_heuristic_click_branch_targets_semantic_object_before_smallest_blob(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:12, 20:22] = 3
        frame[30:35, 40:45] = 5
        agent._bg = 0
        avail = [_GameAction.ACTION6]

        action_idx, coords = agent._heuristic(frame, avail, step=4)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (32, 42))

    def test_semantic_click_targets_prioritize_interactive_colors(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[8:10, 8:10] = 10
        frame[20:22, 30:32] = 6
        frame[40:42, 50:52] = 14
        agent._bg = 0

        targets = agent._semantic_click_targets(frame, limit=4)

        self.assertGreaterEqual(len(targets), 2)
        self.assertEqual(targets[0], (40, 50))
        self.assertEqual(targets[1], (20, 30))

    def test_semantic_click_targets_use_sprite_detector_when_available(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        calls = {}

        def fake_detector(grid):
            calls["shape"] = (len(grid), len(grid[0]))
            return {
                "components_per_value": {
                    "5": [{"center": (18.0, 11.0), "cell_count": 6}],
                    "14": [{"center": (33.0, 41.0), "cell_count": 10}],
                    "10": [{"center": (5.0, 5.0), "cell_count": 50}],
                }
            }

        agent._semantic_detector = fake_detector

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(calls["shape"], (64, 64))
        self.assertEqual(targets[:2], [(33, 41), (18, 11)])

    def test_semantic_click_targets_prefer_previous_target_when_no_player_exists(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (18.0, 34.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (18, 34))
        self.assertEqual(targets[1], (18, 18))

    def test_semantic_click_targets_reuse_cached_components_without_player(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        calls = {"count": 0}

        def fake_detector(grid):
            calls["count"] += 1
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (18.0, 34.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        first = agent._semantic_click_targets(frame, limit=2)
        second = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(first, [(18, 34), (18, 18)])
        self.assertEqual(second, [(18, 34), (18, 18)])
        self.assertEqual(calls["count"], 1)

    def test_semantic_click_targets_cache_ranked_coords_across_limit_changes(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (18.0, 34.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        with mock.patch.object(agent, "_click_targets_from_components", wraps=agent._click_targets_from_components) as ranked:
            first = agent._semantic_click_targets(frame, limit=1)
            second = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(first, [(18, 34)])
        self.assertEqual(second, [(18, 34), (18, 18)])
        self.assertEqual(ranked.call_count, 1)

    def test_click_targets_from_components_reuses_cached_ranked_targets(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        comps = {
            "6": [
                {"center": (18.0, 18.0), "cell_count": 6},
                {"center": (18.0, 34.0), "cell_count": 6},
            ],
        }

        with mock.patch.object(agent, "_rank_click_target_coords", wraps=agent._rank_click_target_coords) as ranked:
            first = agent._click_targets_from_components(frame, comps, (18, 34), (18, 34), None)
            second = agent._click_targets_from_components(frame, comps, (18, 34), (18, 34), None)

        self.assertEqual(first, [(18, 34), (18, 18)])
        self.assertEqual(second, first)
        self.assertEqual(ranked.call_count, 1)
        self.assertIsNotNone(agent._click_targets_from_components_cache_key)
        self.assertEqual(agent._click_targets_from_components_cache_value, tuple(first))

    def test_rank_click_target_coords_reuses_cached_ordering(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        scored_coords = [(18, 34), (18, 18), (31, 41)]

        with mock.patch.object(
                agent,
                "_blocked_click_matches_coord",
                return_value=False) as blocked_mock:
            first = agent._rank_click_target_coords(
                frame,
                scored_coords,
                preferred_coord=(18, 34),
                blocked_click_coord=None,
            )
            second = agent._rank_click_target_coords(
                frame,
                scored_coords,
                preferred_coord=(18, 34),
                blocked_click_coord=None,
            )

        self.assertEqual(first, [(18, 34), (18, 18), (31, 41)])
        self.assertEqual(second, first)
        self.assertEqual(blocked_mock.call_count, 5)
        self.assertIsNotNone(agent._rank_click_target_coords_cache_key)
        self.assertEqual(agent._rank_click_target_coords_cache_value, tuple(first))

    def test_append_unblocked_coords_reuses_cached_suffix(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        candidates = [(18, 34), (18, 18), (31, 41)]
        frame_hash = agent._fast_frame_hash(frame)

        with mock.patch.object(
                agent,
                "_blocked_click_matches_coord",
                return_value=False) as blocked_mock:
            coords_first = []
            seen_first = set()
            first_done = agent._append_unblocked_coords(
                frame,
                candidates,
                coords_first,
                seen_first,
                len(candidates),
                blocked_click_coord=None,
                frame_hash=frame_hash,
            )
            coords_second = []
            seen_second = set()
            second_done = agent._append_unblocked_coords(
                frame,
                candidates,
                coords_second,
                seen_second,
                len(candidates),
                blocked_click_coord=None,
                frame_hash=frame_hash,
            )

        self.assertTrue(first_done)
        self.assertTrue(second_done)
        self.assertEqual(coords_first, candidates)
        self.assertEqual(coords_second, candidates)
        self.assertEqual(seen_first, set(candidates))
        self.assertEqual(seen_second, set(candidates))
        self.assertEqual(blocked_mock.call_count, 3)
        self.assertIsNotNone(agent._append_unblocked_coords_cache_key)
        self.assertEqual(agent._append_unblocked_coords_cache_value, tuple(candidates))

    def test_semantic_click_targets_uses_supplied_frame_hash_without_rehashing(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (18.0, 34.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        with mock.patch.object(agent, "_fast_frame_hash", side_effect=AssertionError("should reuse supplied frame hash")):
            targets = agent._semantic_click_targets(frame, limit=2, frame_hash=123)

        self.assertEqual(targets, [(18, 34), (18, 18)])

    def test_heuristic_click_fallback_targets_reuse_cached_scan(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:12, 20:22] = 6
        frame[30:33, 40:43] = 7

        with mock.patch.object(self.mod.np, "where", wraps=self.mod.np.where) as where_mock:
            first = agent._heuristic_click_fallback_targets(frame, frame_hash=123)
            second = agent._heuristic_click_fallback_targets(frame, frame_hash=123)

        self.assertEqual(first, second)
        self.assertEqual(where_mock.call_count, 2)

    def test_heuristic_click_fallback_targets_invalidate_cache_for_blocked_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:12, 20:22] = 6
        frame[30:33, 40:43] = 7

        first = agent._heuristic_click_fallback_targets(frame, frame_hash=123)
        agent._remember_blocked_click_coord((10, 20))
        second = agent._heuristic_click_fallback_targets(frame, frame_hash=123)

        self.assertEqual(first, [(10, 20), (31, 41)])
        self.assertEqual(second, [(31, 41)])

    def test_semantic_click_targets_without_player_keep_preferred_target_across_small_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (19.0, 35.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (19, 35))
        self.assertEqual(targets[1], (18, 18))

    def test_semantic_click_targets_can_prefer_nearer_useful_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (20, 23))
        self.assertEqual(targets[1], (22, 40))

    def test_semantic_click_targets_with_player_keep_preferred_target_across_small_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (19.0, 35.0), "cell_count": 4},
                        {"center": (20.0, 24.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (19, 35))
        self.assertEqual(targets[1], (20, 24))

    def test_semantic_click_targets_prefer_larger_equally_scored_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [
                        {"center": (20.0, 24.0), "cell_count": 1},
                        {"center": (20.0, 24.0), "cell_count": 9},
                    ],
                }
            }

        agent._semantic_detector = fake_detector

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (20, 24))
        self.assertEqual(len(targets), 1)

    def test_raw_semantic_components_extract_component_centers(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[20:22, 23:25] = 6

        comps = agent._raw_semantic_components(frame)

        self.assertIn("4", comps)
        self.assertIn("6", comps)
        self.assertEqual(comps["4"][0]["cell_count"], 4)
        self.assertEqual(comps["6"][0]["cell_count"], 4)
        self.assertEqual(comps["4"][0]["center"], (20.5, 20.5))
        self.assertEqual(comps["6"][0]["center"], (20.5, 23.5))

    def test_semantic_components_reuses_detector_grid_cache_for_same_frame(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[22:24, 40:42] = 14
        calls = []

        def fake_detector(grid):
            calls.append(grid)
            return {"components_per_value": {"4": [{"center": (20.5, 20.5), "cell_count": 4}]}}

        agent._semantic_detector = fake_detector
        first_hash = agent._fast_frame_hash(frame)
        first = agent._semantic_components(frame, frame_hash=first_hash)
        agent._semantic_components_cache_key = None
        agent._semantic_components_cache_value = None
        second = agent._semantic_components(frame.copy(), frame_hash=first_hash)

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], calls[1])
        self.assertEqual(first, second)

    def test_semantic_click_targets_use_raw_frame_fallback_without_detector(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[22:24, 40:42] = 14
        frame[20:22, 23:25] = 6
        agent._semantic_detector = None

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (20, 24))
        self.assertEqual(targets[1], (22, 40))

    def test_engine_action_input_reuses_plain_actions_and_copies_payloads(self):
        agent = self.make_agent()

        first_plain = agent._engine_action_input(1)
        second_plain = agent._engine_action_input(1)
        payload = {"x": 7, "y": 9}
        payload_action = agent._engine_action_input(6, data=payload)
        payload["x"] = 99

        self.assertIs(first_plain, second_plain)
        self.assertEqual(payload_action.data, {"x": 7, "y": 9})
        self.assertIsNot(payload_action, agent._engine_action_input(6, data={"x": 7, "y": 9}))

    def test_compile_demo_actions_precomputes_indices_and_inputs(self):
        agent = self.make_agent()

        compiled = agent._compile_demo_actions([
            (1, None),
            (6, {"x": 7, "y": 9}),
        ])

        self.assertEqual(compiled[0][0], 1)
        self.assertIsNone(compiled[0][1])
        self.assertEqual(compiled[0][2], 0)
        self.assertIs(compiled[0][3], agent._engine_action_input(1))
        self.assertEqual(compiled[1][0], 6)
        self.assertEqual(compiled[1][1], {"x": 7, "y": 9})
        self.assertEqual(compiled[1][2], 5 + 9 * agent.G + 7)
        self.assertEqual(compiled[1][3].data, {"x": 7, "y": 9})

    def test_click_bonus_maps_reuse_cached_frame_results(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[20:22, 23:25] = 6
        frame_hash = agent._fast_frame_hash(frame)

        semantic_calls = []
        fallback_calls = []

        def fake_semantic_click_targets(*args, **kwargs):
            semantic_calls.append((args, kwargs))
            return [(20, 24), (22, 40)]

        def fake_fallback_targets(*args, **kwargs):
            fallback_calls.append((args, kwargs))
            return [(18, 18), (30, 30)]

        agent._semantic_click_targets = fake_semantic_click_targets
        agent._heuristic_click_fallback_targets = fake_fallback_targets

        first_semantic = agent._semantic_click_bonus_map(
            frame,
            limit=2,
            click_scale=0.75,
            frame_hash=frame_hash,
        )
        second_semantic = agent._semantic_click_bonus_map(
            frame.copy(),
            limit=2,
            click_scale=0.75,
            frame_hash=frame_hash,
        )
        first_fallback = agent._heuristic_click_bonus_map(
            frame,
            limit=2,
            click_scale=0.5,
            frame_hash=frame_hash,
        )
        second_fallback = agent._heuristic_click_bonus_map(
            frame.copy(),
            limit=2,
            click_scale=0.5,
            frame_hash=frame_hash,
        )

        self.assertEqual(len(semantic_calls), 1)
        self.assertEqual(len(fallback_calls), 1)
        self.assertIs(first_semantic, second_semantic)
        self.assertIs(first_fallback, second_fallback)
        self.assertEqual(first_semantic[(20, 24)], 0.8 * 0.75)
        self.assertEqual(first_fallback[(18, 18)], 0.35 * 0.5)

    def test_semantic_click_targets_raw_fallback_prefers_previous_target_when_no_player_exists(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 6
        frame[18:20, 34:36] = 6
        agent._semantic_detector = None
        agent._semantic_target_coord = (18, 34)

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (18, 34))
        self.assertEqual(targets[1], (18, 18))

    def test_semantic_click_targets_raw_fallback_keeps_preferred_target_across_small_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 6
        frame[19:21, 35:37] = 6
        agent._semantic_detector = None
        agent._semantic_target_coord = (18, 34)

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (20, 36))
        self.assertEqual(targets[1], (18, 18))

    def test_semantic_click_targets_raw_fallback_drops_preferred_target_after_unproductive_streak(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 6
        frame[18:20, 46:48] = 6
        agent._semantic_detector = None
        agent._semantic_target_coord = (18, 46)
        agent._unproductive = 8

        targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(targets[0], (18, 18))
        self.assertEqual(targets[1], (18, 46))

    def test_semantic_click_targets_raw_fallback_softens_preferred_target_before_full_cutoff(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:12, 50:52] = 6
        frame[20:22, 24:26] = 6
        agent._semantic_detector = None
        agent._semantic_target_coord = (10, 50)

        fresh_targets = agent._semantic_click_targets(frame, limit=2)
        agent._unproductive = 6
        softened_targets = agent._semantic_click_targets(frame, limit=2)

        self.assertEqual(fresh_targets[0], (10, 50))
        self.assertEqual(softened_targets[0], (20, 24))

    def test_semantic_click_targets_skip_recent_blocked_click_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 6
        frame[18:20, 34:36] = 6
        frame[18:20, 46:48] = 6
        agent._semantic_detector = None
        agent._remember_blocked_click_coord((18, 18))
        agent._remember_blocked_click_coord((18, 34))

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (18, 46))
        self.assertNotEqual(targets[0], (18, 18))
        self.assertNotEqual(targets[0], (18, 34))

    def test_semantic_click_bonus_map_uses_supplied_targets(self):
        agent = self.make_agent()
        click_targets = [(20, 24), (22, 40), (18, 18)]

        bonus_map = agent._semantic_click_bonus_map(
            np.zeros((64, 64), dtype=np.uint8),
            limit=3,
            click_scale=0.5,
            click_targets=click_targets,
        )

        self.assertAlmostEqual(bonus_map[(20, 24)], 0.4, places=5)
        self.assertAlmostEqual(bonus_map[(22, 40)], 0.35, places=5)
        self.assertAlmostEqual(bonus_map[(18, 18)], 0.3, places=5)

    def test_semantic_click_bonus_matches_ranked_targets_without_bonus_map(self):
        agent = self.make_agent()
        click_targets = [(20, 24), (22, 40), (18, 18)]

        self.assertAlmostEqual(agent._semantic_click_bonus((20, 24), 0.5, click_targets), 0.4, places=5)
        self.assertAlmostEqual(agent._semantic_click_bonus((22, 40), 0.5, click_targets), 0.35, places=5)
        self.assertAlmostEqual(agent._semantic_click_bonus((0, 0), 0.5, click_targets), 0.0, places=5)

    def test_blocked_click_coord_returns_last_click_when_state_is_unchanged(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        blocked = agent._blocked_click_coord(frame)

        self.assertEqual(blocked, (18, 11))

    def test_blocked_click_matches_recent_block_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._remember_blocked_click_coord((18, 11))

        self.assertTrue(agent._blocked_click_matches_coord(frame, (18, 12)))
        self.assertFalse(agent._blocked_click_matches_coord(frame, (18, 15)))

    def test_blocked_click_history_refreshes_recency_for_nearby_repeat(self):
        agent = self.make_agent()

        agent._remember_blocked_click_coord((10, 10))
        agent._remember_blocked_click_coord((20, 20))
        agent._remember_blocked_click_coord((30, 30))
        agent._remember_blocked_click_coord((11, 10))
        agent._remember_blocked_click_coord((40, 40))

        self.assertEqual(list(agent._blocked_click_history), [(30, 30), (11, 10), (40, 40)])

    def test_decay_blocked_click_history_drops_only_oldest_entry(self):
        agent = self.make_agent()
        agent._remember_blocked_click_coord((10, 10))
        agent._remember_blocked_click_coord((20, 20))
        agent._remember_blocked_click_coord((30, 30))

        agent._decay_blocked_click_history()

        self.assertEqual(list(agent._blocked_click_history), [(20, 20), (30, 30)])

    def test_direction_matches_recent_block_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._remember_blocked_direction_index(3)

        self.assertTrue(agent._direction_matches_blocked_history(3, frame))
        self.assertFalse(agent._direction_matches_blocked_history(1, frame))

    def test_direction_matches_blocked_history_uses_supplied_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_blocked_direction_action_index", side_effect=AssertionError("should use supplied blocked direction")):
            self.assertTrue(agent._direction_matches_blocked_history(2, frame, blocked_direction=2))
            self.assertFalse(agent._direction_matches_blocked_history(1, frame, blocked_direction=2))

    def test_blocked_direction_history_refreshes_recency_for_repeat(self):
        agent = self.make_agent()

        agent._remember_blocked_direction_index(0)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(2)
        agent._remember_blocked_direction_index(0)
        agent._remember_blocked_direction_index(3)

        self.assertEqual(list(agent._blocked_direction_history), [2, 0, 3])

    def test_decay_blocked_direction_history_drops_only_oldest_entry(self):
        agent = self.make_agent()
        agent._remember_blocked_direction_index(0)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(2)

        agent._decay_blocked_direction_history()

        self.assertEqual(list(agent._blocked_direction_history), [1, 2])

    def test_previous_frame_relation_cache_reuses_single_equality_check(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()
        agent.ph = agent._fast_frame_hash(agent.pr)

        with mock.patch.object(self.mod.np, "array_equal", wraps=self.mod.np.array_equal) as array_equal:
            self.assertEqual(agent._blocked_click_coord(frame), (18, 11))
            self.assertEqual(agent._blocked_click_action_index(frame), 5 + 18 * agent.G + 11)
            self.assertIsNone(agent._recent_click_action_index(frame))
            self.assertTrue(agent._frame_matches_previous(frame))

        self.assertEqual(array_equal.call_count, 1)

    def test_previous_frame_relation_exposes_cached_axis_and_blocked_click_index(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pr = frame.copy()
        agent.ph = agent._fast_frame_hash(agent.pr)

        agent.pai = 1
        moved = frame.copy()
        moved[0, 0] = 1
        self.assertEqual(agent._recent_direction_axis(moved), "vertical")

        agent.pai = 5 + 18 * agent.G + 11
        self.assertEqual(agent._blocked_click_action_index(frame), 5 + 18 * agent.G + 11)

    def test_blocked_click_action_index_returns_last_click_slot(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        blocked_idx = agent._blocked_click_action_index(frame)

        self.assertEqual(blocked_idx, 5 + 18 * agent.G + 11)

    def test_semantic_click_targets_avoid_repeating_blocked_click(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (18.0, 11.0), "cell_count": 6}],
                    "6": [{"center": (33.0, 41.0), "cell_count": 10}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (33, 41))
        self.assertNotIn((18, 11), targets[:1])

    def test_semantic_click_targets_avoid_nearby_blocked_click_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [
                        {"center": (19.0, 12.0), "cell_count": 6},
                        {"center": (33.0, 41.0), "cell_count": 10},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (33, 41))
        self.assertNotIn((19, 12), targets[:1])

    def test_semantic_click_targets_preferred_continuity_skips_nearby_blocked_click_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [
                        {"center": (19.0, 12.0), "cell_count": 6},
                        {"center": (19.0, 35.0), "cell_count": 7},
                        {"center": (33.0, 41.0), "cell_count": 10},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 11)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        targets = agent._semantic_click_targets(frame, limit=3)

        self.assertEqual(targets[0], (19, 35))
        self.assertNotEqual(targets[0], (19, 12))

    def test_heuristic_click_branch_prefers_semantic_targets_over_smallest_blob(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:12, 20:22] = 3
        frame[30:34, 40:44] = 14
        agent._bg = 0

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=4)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (32, 42))

    def test_heuristic_click_branch_raw_blob_fallback_avoids_repeating_blocked_click(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 11:13] = 3
        frame[30:32, 40:42] = 5
        agent._bg = 0
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=4)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (30, 40))

    def test_heuristic_click_branch_raw_blob_fallback_avoids_nearby_blocked_click_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[19:21, 12:14] = 3
        frame[30:32, 40:42] = 5
        agent._bg = 0
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=4)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (30, 40))

    def test_heuristic_click_branch_raw_blob_fallback_keeps_preferred_target_across_small_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 3
        frame[19:21, 35:37] = 5
        agent._bg = 0
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._semantic_target_coord = (18, 34)

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=5)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (19, 35))

    def test_heuristic_click_branch_stays_on_preferred_semantic_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (18.0, 34.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=5)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (18, 34))

    def test_heuristic_click_branch_tracks_preferred_target_across_small_detector_jitter(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "6": [
                        {"center": (18.0, 18.0), "cell_count": 6},
                        {"center": (19.0, 35.0), "cell_count": 6},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (18, 34)

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION6], step=5)

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, (19, 35))

    def test_semantic_direction_action_moves_toward_target_on_dominant_axis(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION1, _GameAction.ACTION4])

        self.assertEqual(action, (3, None))

    def test_semantic_direction_action_can_follow_nearer_lower_priority_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (3, None))

    def test_semantic_direction_action_uses_raw_frame_fallback_without_detector(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[20:22, 23:25] = 6
        agent._semantic_detector = None

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (3, None))

    def test_semantic_direction_action_can_fallback_to_secondary_target_when_best_target_is_illegal(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (30.0, 20.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_action_does_not_invent_perpendicular_move_when_axis_aligned(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION1, _GameAction.ACTION2])

        self.assertIsNone(action)

    def test_semantic_direction_action_falls_back_to_secondary_axis_when_needed(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (30.0, 35.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION4])
        self.assertEqual(action, (3, None))

        fallback = agent._semantic_direction_action(frame, [_GameAction.ACTION2])
        self.assertEqual(fallback, (1, None))

    def test_semantic_direction_action_avoids_repeating_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (30.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = frame.copy()

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_action_avoids_recent_blocked_direction_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (30.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._remember_blocked_direction_index(3)

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_action_can_skip_fully_blocked_top_target_for_later_positive_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (30.0, 20.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._remember_blocked_direction_index(3)

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_action_breaks_diagonal_tie_with_recent_vertical_momentum(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (24.0, 24.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 1
        agent.pr = previous

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_action_avoids_immediate_reverse_when_secondary_axis_is_available(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (24.0, 16.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = previous

        action = agent._semantic_direction_action(frame, [_GameAction.ACTION2, _GameAction.ACTION3])

        self.assertEqual(action, (1, None))

    def test_semantic_direction_bonuses_penalize_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (30.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = frame.copy()

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertLess(bonuses[3], 0.0)
        self.assertGreater(bonuses[1], bonuses[3])

    def test_semantic_direction_bonuses_penalize_recent_blocked_direction_history(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (30.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._remember_blocked_direction_index(3)

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertLess(bonuses[3], 0.0)
        self.assertGreater(bonuses[1], bonuses[3])

    def test_semantic_direction_bonuses_penalize_immediate_reverse_when_secondary_axis_exists(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (24.0, 16.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = previous

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertGreater(bonuses[1], bonuses[2])

    def test_semantic_direction_bonuses_allow_reverse_when_recent_direction_worsened_distance(self):
        agent = self.make_agent()
        previous = np.zeros((64, 64), dtype=np.uint8)
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous[20, 20] = 4
        frame[20, 21] = 4

        def fake_detector(grid):
            grid = np.asarray(grid, dtype=np.uint8)
            player_x = 20.0 if grid[20, 20] == 4 else 21.0
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, player_x), "cell_count": 1}],
                    "14": [{"center": (24.0, 18.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = previous
        agent.ph = agent._fast_frame_hash(previous)

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertGreater(bonuses[2], 0.0)
        self.assertLess(bonuses[2], bonuses[1])

    def test_recent_direction_progress_delta_reuses_cached_result(self):
        agent = self.make_agent()
        previous = np.zeros((64, 64), dtype=np.uint8)
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous[20, 20] = 4
        frame[20, 21] = 4

        def fake_detector(grid):
            grid = np.asarray(grid, dtype=np.uint8)
            player_x = 20.0 if grid[20, 20] == 4 else 21.0
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, player_x), "cell_count": 1}],
                    "14": [{"center": (24.0, 18.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 3
        agent.pr = previous
        agent.ph = agent._fast_frame_hash(previous)
        frame_hash = agent._fast_frame_hash(frame)

        with mock.patch.object(agent, "_semantic_components", wraps=agent._semantic_components) as comps_mock:
            first = agent._recent_direction_progress_delta(frame, frame_hash=frame_hash)
            second = agent._recent_direction_progress_delta(frame, frame_hash=frame_hash)

        self.assertAlmostEqual(first, second, places=6)
        self.assertEqual(comps_mock.call_count, 2)

    def test_heuristic_opening_skips_blocked_direction_when_state_is_unchanged(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        agent.pai = 2
        agent.pr = frame.copy()

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION3, _GameAction.ACTION4], step=0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_heuristic_opening_reuses_previous_direction_when_legal(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1
        agent._bg = 0
        agent.pai = 3
        agent.pr = previous

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION4], step=0)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_heuristic_prefers_semantic_direction_when_opening_rule_does_not_apply(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (40.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION1, _GameAction.ACTION2], step=6)

        self.assertEqual(action_idx, 1)
        self.assertIsNone(coords)

    def test_heuristic_late_fallback_reuses_previous_direction_when_legal(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1
        agent._bg = 0
        agent.pai = 3
        agent.pr = previous

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION4], step=6)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_semantic_direction_bonuses_prioritize_primary_and_secondary_axes(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (28.0, 45.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertGreater(bonuses[3], bonuses[1])
        self.assertNotIn(0, bonuses)

    def test_semantic_direction_bonuses_break_diagonal_tie_with_recent_vertical_momentum(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (24.0, 24.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 1
        agent.pr = previous

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertGreater(bonuses[1], bonuses[3])

    def test_semantic_direction_bonuses_skip_perpendicular_axis_when_aligned(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 28.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        bonuses = agent._semantic_direction_bonuses(frame)

        self.assertEqual(bonuses, {3: 0.45})

    def test_semantic_direction_bonuses_can_use_legal_secondary_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (30.0, 20.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        bonuses = agent._semantic_direction_bonuses(frame, [_GameAction.ACTION2, _GameAction.ACTION3])

        self.assertIn(1, bonuses)
        self.assertNotIn(3, bonuses)

    def test_semantic_direction_bonuses_can_skip_fully_blocked_top_target_for_later_positive_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (30.0, 20.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent._remember_blocked_direction_index(3)

        bonuses = agent._semantic_direction_bonuses(frame, [_GameAction.ACTION2, _GameAction.ACTION4])

        self.assertGreater(bonuses[1], 0.0)
        self.assertNotEqual(max(bonuses, key=bonuses.get), 3)

    def test_semantic_direction_bonuses_reuse_cached_target_scan(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        calls = []

        def fake_targets(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"player_y": 20.0, "player_x": 20.0, "target_y": 20.0, "target_x": 30.0}]

        agent._semantic_target_candidates = fake_targets
        agent._clear_blocked_click_history()
        agent._clear_blocked_direction_history()

        first = agent._semantic_direction_bonuses(frame, frame_hash=frame_hash)
        second = agent._semantic_direction_bonuses(frame.copy(), frame_hash=frame_hash)

        self.assertEqual(len(calls), 1)
        self.assertIs(first, second)
        self.assertEqual(first, {3: 0.45})

    def test_semantic_exploration_logits_reuse_cached_tensor(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        avail = [_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION6]
        avail_ids = agent._available_action_ids(avail)
        avail_summary = agent._availability_summary(avail_ids)
        calls = []

        def fake_click_targets(*args, **kwargs):
            calls.append((args, kwargs))
            return [(20, 24), (22, 40)]

        agent._semantic_click_targets = fake_click_targets
        agent._clear_blocked_click_history()
        agent._clear_blocked_direction_history()

        first = agent._semantic_exploration_logits(
            frame,
            avail,
            True,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )
        second = agent._semantic_exploration_logits(
            frame.copy(),
            avail,
            True,
            avail_ids=avail_ids,
            frame_hash=frame_hash,
            avail_summary=avail_summary,
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(first, second)

    def test_semantic_exploration_logits_bias_direction_and_click_targets(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (40.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION6], True)

        self.assertGreater(logits[3].item(), logits[1].item())
        self.assertGreater(logits[5 + 22 * agent.G + 40].item(), 0.0)
        self.assertGreater(logits[5 + 40 * agent.G + 22].item(), 0.0)
        self.assertFalse(torch.isneginf(logits[3]))

    def test_semantic_exploration_logits_exclude_blocked_direction(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 3
        agent.pr = frame.copy()

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION2, _GameAction.ACTION4], False)

        self.assertTrue(torch.isneginf(logits[3]))
        self.assertFalse(torch.isneginf(logits[1]))

    def test_semantic_exploration_logits_without_clicks_stays_lazy_on_click_only_helpers(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_semantic_target_choice", side_effect=AssertionError("should stay lazy without clicks")):
            with mock.patch.object(agent, "_semantic_click_bonus_scale", side_effect=AssertionError("should stay lazy without clicks")):
                logits = agent._semantic_exploration_logits(
                    frame,
                    [_GameAction.ACTION2, _GameAction.ACTION4],
                    False,
                )

        self.assertEqual(tuple(logits.shape), (5,))

    def test_semantic_exploration_logits_reuses_local_preferred_click_continuity_state(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_semantic_target_choice", return_value=None):
            with mock.patch.object(agent, "_semantic_click_bonus_scale", return_value=1.0):
                with mock.patch.object(agent, "_preferred_click_coord", return_value=(22, 40)):
                    with mock.patch.object(agent, "_semantic_continuity_scale", return_value=1.0):
                        with mock.patch.object(agent, "_preferred_click_continuity_active", side_effect=AssertionError("should use local continuity state")):
                            logits = agent._semantic_exploration_logits(
                                frame,
                                [_GameAction.ACTION6],
                                True,
                                frame_hash=agent._fast_frame_hash(frame),
                            )

        self.assertEqual(tuple(logits.shape), (4101,))

    def test_semantic_exploration_logits_uses_supplied_availability_summary(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        avail_ids = [2, 4]
        avail_summary = {
            "has_click": False,
            "has_undo": False,
            "has_modeled": True,
            "legal_dirs": frozenset({2, 4}),
        }

        with mock.patch.object(agent, "_availability_summary", side_effect=AssertionError("should use supplied availability summary")):
            logits = agent._semantic_exploration_logits(
                frame,
                [_GameAction.ACTION2, _GameAction.ACTION4],
                False,
                avail_ids=avail_ids,
                frame_hash=agent._fast_frame_hash(frame),
                avail_summary=avail_summary,
            )

        self.assertEqual(tuple(logits.shape), (5,))
        self.assertFalse(torch.isneginf(logits[1]))

    def test_semantic_exploration_logits_penalize_blocked_click(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.pai = 5 + 20 * agent.G + 23
        agent.pr = frame.copy()

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)

        self.assertTrue(torch.isneginf(logits[5 + 20 * agent.G + 23]))
        self.assertGreater(logits[5 + 22 * agent.G + 40].item(), 0.0)

    def test_semantic_exploration_logits_penalize_nearby_blocked_click_jitter(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 5 + 20 * agent.G + 23
        agent.pr = frame.copy()

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)

        self.assertTrue(torch.isneginf(logits[5 + 19 * agent.G + 24]))
        self.assertTrue(torch.isneginf(logits[5 + 21 * agent.G + 23]))

    def test_semantic_exploration_logits_penalize_recent_blocked_click_history(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._remember_blocked_click_coord((18, 11))

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)

        self.assertTrue(torch.isneginf(logits[5 + 18 * agent.G + 11]))
        self.assertTrue(torch.isneginf(logits[5 + 19 * agent.G + 11]))
        self.assertFalse(torch.isneginf(logits[5 + 21 * agent.G + 14]))

    def test_semantic_exploration_logits_downweight_far_clicks_when_player_visible(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION4, _GameAction.ACTION6], True)
        click_idx = 5 + 22 * agent.G + 40

        self.assertGreater(logits[3].item(), logits[click_idx].item())

    def test_semantic_exploration_logits_include_preferred_click_target_outside_semantic_top_list(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (32, 42)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]

        self.assertGreater(logits[preferred_idx].item(), 0.0)
        self.assertAlmostEqual(logits[5 + 0 * agent.G + 0].item(), 0.8, places=6)

    def test_semantic_exploration_logits_do_not_boost_preferred_click_in_blocked_jitter_region(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (19, 12)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]

        self.assertTrue(torch.isneginf(logits[preferred_idx]))

    def test_semantic_exploration_logits_do_not_boost_preferred_click_from_recent_history_when_current_block_differs(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (19, 12)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent._remember_blocked_click_coord(preferred)

        logits = agent._semantic_exploration_logits(
            frame,
            [_GameAction.ACTION6],
            True,
            blocked_click_coord=(0, 0),
            frame_hash=agent._fast_frame_hash(frame),
        )
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]

        self.assertTrue(torch.isneginf(logits[preferred_idx]))

    def test_semantic_exploration_logits_do_not_boost_stale_preferred_click_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (32, 42)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent._unproductive = 6

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]

        self.assertEqual(logits[preferred_idx].item(), 0.0)
        self.assertAlmostEqual(logits[5 + 0 * agent.G + 0].item(), 0.8, places=6)

    def test_semantic_exploration_logits_include_raw_fallback_click_targets(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 3
        frame[30:32, 40:42] = 5
        agent._semantic_detector = lambda grid: {"components_per_value": {}}

        logits = agent._semantic_exploration_logits(frame, [_GameAction.ACTION6], True)

        self.assertGreater(logits[5 + 18 * agent.G + 18].item(), 0.0)
        self.assertGreater(logits[5 + 30 * agent.G + 40].item(), 0.0)

    def test_sample_semantic_exploration_sparse_returns_none_without_clicks_and_stays_lazy(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_availability_summary", side_effect=AssertionError("should exit before summary")):
            with mock.patch.object(agent, "_semantic_direction_bonuses", side_effect=AssertionError("should exit before semantic scan")):
                result = agent._sample_semantic_exploration_sparse(
                    frame,
                    [_GameAction.ACTION2, _GameAction.ACTION4],
                    avail_ids=[2, 4],
                )

        self.assertIsNone(result)

    def test_sample_semantic_exploration_sparse_can_choose_direction_candidate(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_semantic_direction_bonuses", return_value={3: 2.0}):
            with mock.patch.object(agent, "_wait_recovery_bonus", return_value=0.0):
                with mock.patch.object(agent, "_retry_blocked_direction_after_stale_wait", return_value=False):
                    with mock.patch.object(agent, "_blocked_direction_action_index", return_value=None):
                        with mock.patch.object(agent, "_direction_matches_blocked_history", return_value=False):
                            with mock.patch.object(agent, "_blocked_click_coord", return_value=None):
                                with mock.patch.object(agent, "_semantic_click_targets_compat", return_value=[(22, 40)]):
                                    with mock.patch.object(agent, "_semantic_click_candidate_indices", return_value=[5 + 22 * agent.G + 40]):
                                        with mock.patch.object(agent, "_semantic_target_choice", return_value=None):
                                            with mock.patch.object(agent, "_semantic_click_bonus_scale", return_value=1.0):
                                                with mock.patch.object(agent, "_semantic_click_bonus_map", return_value={(22, 40): 1.0}):
                                                    with mock.patch.object(agent, "_heuristic_click_bonus_map", return_value={}):
                                                        with mock.patch.object(agent, "_preferred_click_coord", return_value=None):
                                                            with mock.patch.object(agent, "_semantic_continuity_scale", return_value=0.0):
                                                                with mock.patch.object(self.mod.random, "random", return_value=0.0):
                                                                    action_idx, coords = agent._sample_semantic_exploration_sparse(
                                                                        frame,
                                                                        [_GameAction.ACTION4, _GameAction.ACTION6],
                                                                        avail_ids=[4, 6],
                                                                        frame_hash=agent._fast_frame_hash(frame),
                                                                    )

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_sample_semantic_exploration_sparse_can_choose_click_candidate_with_supplied_summary(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        avail_ids = [6]
        avail_summary = {
            "has_click": True,
            "has_undo": False,
            "has_modeled": True,
            "legal_dirs": frozenset(),
        }
        click_coord = (22, 40)

        with mock.patch.object(agent, "_availability_summary", side_effect=AssertionError("should use supplied availability summary")):
            with mock.patch.object(agent, "_semantic_direction_bonuses", return_value={}):
                with mock.patch.object(agent, "_wait_recovery_bonus", return_value=0.0):
                    with mock.patch.object(agent, "_retry_blocked_direction_after_stale_wait", return_value=False):
                        with mock.patch.object(agent, "_blocked_direction_action_index", return_value=None):
                            with mock.patch.object(agent, "_blocked_click_coord", return_value=None):
                                with mock.patch.object(agent, "_semantic_click_targets_compat", return_value=[click_coord]):
                                    with mock.patch.object(agent, "_semantic_click_candidate_indices", return_value=[5 + click_coord[0] * agent.G + click_coord[1]]):
                                        with mock.patch.object(agent, "_semantic_target_choice", return_value=None):
                                            with mock.patch.object(agent, "_semantic_click_bonus_scale", return_value=1.0):
                                                with mock.patch.object(agent, "_semantic_click_bonus_map", return_value={click_coord: 1.0}):
                                                    with mock.patch.object(agent, "_heuristic_click_bonus_map", return_value={}):
                                                        with mock.patch.object(agent, "_preferred_click_coord", return_value=None):
                                                            with mock.patch.object(agent, "_semantic_continuity_scale", return_value=0.0):
                                                                with mock.patch.object(self.mod.random, "random", return_value=0.0):
                                                                    action_idx, coords = agent._sample_semantic_exploration_sparse(
                                                                        frame,
                                                                        [_GameAction.ACTION6],
                                                                        avail_ids=avail_ids,
                                                                        frame_hash=frame_hash,
                                                                        avail_summary=avail_summary,
                                                                    )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, click_coord)

    def test_sample_semantic_exploration_sparse_reuses_cached_distribution(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        avail = [_GameAction.ACTION4, _GameAction.ACTION6]
        avail_ids = agent._available_action_ids(avail)
        avail_summary = agent._availability_summary(avail_ids)
        calls = []

        def fake_click_targets(*args, **kwargs):
            calls.append((args, kwargs))
            return [(22, 40)]

        agent._semantic_click_targets = fake_click_targets
        agent._clear_blocked_click_history()
        agent._clear_blocked_direction_history()

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            first = agent._sample_semantic_exploration_sparse(
                frame,
                avail,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )
            second = agent._sample_semantic_exploration_sparse(
                frame.copy(),
                avail,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)

    def test_sample_semantic_exploration_sparse_cached_hit_skips_decode(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        avail = [_GameAction.ACTION4, _GameAction.ACTION6]
        avail_ids = agent._available_action_ids(avail)
        avail_summary = agent._availability_summary(avail_ids)

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            first = agent._sample_semantic_exploration_sparse(
                frame,
                avail,
                avail_ids=avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )

        with mock.patch.object(
                agent,
                "_decode_policy_action_index",
                side_effect=AssertionError("cached hit should reuse decoded action")):
            with mock.patch.object(self.mod.random, "random", return_value=0.0):
                second = agent._sample_semantic_exploration_sparse(
                    frame.copy(),
                    avail,
                    avail_ids=avail_ids,
                    frame_hash=frame_hash,
                    avail_summary=avail_summary,
                )

        self.assertEqual(first, second)

    def test_semantic_goal_distance_uses_raw_frame_fallback_without_detector(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[20:22, 20:22] = 4
        frame[20:22, 23:25] = 6
        agent._semantic_detector = None

        dist = agent._semantic_goal_distance(frame)

        self.assertEqual(dist, 3.0)

    def test_semantic_candidate_action_indices_include_direction_and_click_targets(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (40.0, 22.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector

        indices = agent._semantic_candidate_action_indices(frame, True)

        self.assertIn(3, indices)
        self.assertIn(5 + 22 * agent.G + 40, indices)
        self.assertIn(5 + 40 * agent.G + 22, indices)

    def test_semantic_candidate_action_indices_include_preferred_click_target_outside_top_list(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (32, 42)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]

        indices = agent._semantic_candidate_action_indices(frame, True)

        self.assertIn(5 + preferred[0] * agent.G + preferred[1], indices)
        self.assertIn(5 + 0 * agent.G + 0, indices)

    def test_semantic_candidate_action_indices_skip_preferred_click_in_blocked_jitter_region(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (19, 12)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        indices = agent._semantic_candidate_action_indices(frame, True)

        self.assertNotIn(5 + preferred[0] * agent.G + preferred[1], indices)
        self.assertIn(5 + 0 * agent.G + 0, indices)

    def test_semantic_candidate_action_indices_skip_preferred_click_from_recent_history_when_current_block_differs(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (19, 12)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent._remember_blocked_click_coord(preferred)

        indices = agent._semantic_candidate_action_indices(
            frame,
            True,
            blocked_click_coord=(0, 0),
            frame_hash=agent._fast_frame_hash(frame),
        )

        self.assertNotIn(5 + preferred[0] * agent.G + preferred[1], indices)
        self.assertIn(5 + 0 * agent.G + 0, indices)

    def test_semantic_candidate_action_indices_skip_stale_preferred_click_target(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        preferred = (32, 42)
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]
        agent._unproductive = 6

        indices = agent._semantic_candidate_action_indices(frame, True)

        self.assertNotIn(5 + preferred[0] * agent.G + preferred[1], indices)
        self.assertIn(5 + 0 * agent.G + 0, indices)

    def test_semantic_candidate_action_indices_include_raw_fallback_click_targets(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 3
        frame[30:32, 40:42] = 5
        agent._semantic_detector = lambda grid: {"components_per_value": {}}

        indices = agent._semantic_candidate_action_indices(frame, True)

        self.assertIn(5 + 18 * agent.G + 18, indices)
        self.assertIn(5 + 30 * agent.G + 40, indices)

    def test_semantic_candidate_action_indices_reuse_cached_click_shortlist(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        calls = {"semantic": 0, "fallback": 0}

        agent._semantic_direction_bonuses = lambda *args, **kwargs: {3: 0.45}

        def fake_semantic_targets(raw, limit=6, blocked_click_coord=None, frame_hash=None):
            calls["semantic"] += 1
            return [(22, 40)]

        def fake_fallback_targets(raw, blocked_click_coord=None, frame_hash=None):
            calls["fallback"] += 1
            return [(30, 40)]

        agent._semantic_click_targets = fake_semantic_targets
        agent._heuristic_click_fallback_targets = fake_fallback_targets

        first = agent._semantic_candidate_action_indices(frame, True, frame_hash=frame_hash)
        second = agent._semantic_candidate_action_indices(frame, True, frame_hash=frame_hash)

        self.assertEqual(first, second)
        self.assertEqual(calls["semantic"], 1)
        self.assertEqual(calls["fallback"], 1)
        self.assertIn(5 + 22 * agent.G + 40, first)
        self.assertIn(5 + 30 * agent.G + 40, first)

    def test_semantic_candidate_action_indices_include_wait_for_stalled_recovery(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._unproductive = 6
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}

        indices = agent._semantic_candidate_action_indices(frame, False, avail_ids=[2, 4, 5], frame_hash=agent._fast_frame_hash(frame))

        self.assertIn(4, indices)

    def test_semantic_candidate_action_indices_uses_supplied_wait_recovery_bonus(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        with mock.patch.object(agent, "_wait_recovery_bonus", side_effect=AssertionError("should use supplied wait bonus")):
            indices = agent._semantic_candidate_action_indices(
                frame,
                False,
                avail_ids=[2, 4, 5],
                frame_hash=agent._fast_frame_hash(frame),
                wait_recovery_bonus=0.3,
            )

        self.assertIn(4, indices)

    def test_semantic_candidate_action_indices_uses_supplied_click_candidate_indices(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        click_idx = 5 + 22 * agent.G + 40

        with mock.patch.object(agent, "_semantic_click_candidate_indices", side_effect=AssertionError("should use supplied click candidate indices")):
            indices = agent._semantic_candidate_action_indices(
                frame,
                True,
                direction_bonuses={3: 0.45},
                click_candidate_indices=[click_idx],
                wait_recovery_bonus=0.0,
            )

        self.assertIn(3, indices)
        self.assertIn(click_idx, indices)

    def test_cnn_rescoring_prefers_semantic_direction_without_bfs(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, 8.25, -10.0, 8.0, -10.0], dtype=np.float32)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertIn(result.value, (2, 4))
        self.assertTrue(result.reasoning.startswith("cnn:a"))

    def test_cnn_rescoring_without_click_does_not_touch_click_only_semantic_helpers(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, 8.25, -10.0, 8.0, -10.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        with mock.patch.object(agent, "_refresh_semantic_target_coord", return_value=None):
            with mock.patch.object(agent, "_semantic_direct_click_choice", side_effect=AssertionError("should stay lazy without click")):
                with mock.patch.object(agent, "_preferred_click_coord", side_effect=AssertionError("should stay lazy without click")):
                    with mock.patch.object(agent, "_preferred_click_continuity_active", side_effect=AssertionError("should stay lazy without click")):
                        with mock.patch.object(agent, "_semantic_target_choice", side_effect=AssertionError("should stay lazy without click")):
                            with mock.patch.object(agent, "_semantic_click_bonus_scale", side_effect=AssertionError("should stay lazy without click")):
                                with mock.patch.object(agent, "_recent_click_action_index", side_effect=AssertionError("should stay lazy without click")):
                                    with mock.patch.object(agent, "_blocked_click_action_index", side_effect=AssertionError("should stay lazy without click")):
                                        result = agent.choose_action([], frame)

        self.assertIn(result.value, (2, 4))
        self.assertTrue(result.reasoning.startswith("cnn:a"))

    def test_cnn_rescoring_prefers_wait_when_stuck_and_frontiers_are_exhausted(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION5], levels=0)
        logits = np.array([-10.0, 9.0, -10.0, 8.5, -2.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent._unproductive = 6
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 5)
        self.assertEqual(result.reasoning, "cnn:a5")

    def test_retry_blocked_direction_after_stale_wait_reuses_supplied_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()
        avail_summary = agent._availability_summary([2, 5])

        with mock.patch.object(
                agent,
                "_blocked_direction_action_index",
                side_effect=AssertionError("should reuse supplied blocked direction")):
            result = agent._retry_blocked_direction_after_stale_wait(
                frame,
                [2, 5],
                frame_hash=frame_hash,
                avail_summary=avail_summary,
                blocked_direction=1,
            )

        self.assertTrue(result)

    def test_should_exit_warmup_early_skips_wait_recovery_bonus_during_stale_wait(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        avail_ids = [2, 4, 5]
        avail_summary = agent._availability_summary(avail_ids)

        with mock.patch.object(
                agent,
                "_wait_recovery_bonus",
                side_effect=AssertionError("should skip wait recovery bonus during stale wait")):
            result = agent._should_exit_warmup_early(
                frame,
                avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )

        self.assertTrue(result)

    def test_should_exit_warmup_early_stale_wait_avoids_per_direction_helper(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        avail_ids = [2, 4, 5]
        avail_summary = agent._availability_summary(avail_ids)

        with mock.patch.object(
                agent,
                "_direction_matches_blocked_history",
                side_effect=AssertionError("should use aggregated blocked-direction check")):
            result = agent._should_exit_warmup_early(
                frame,
                avail_ids,
                frame_hash=frame_hash,
                avail_summary=avail_summary,
            )

        self.assertTrue(result)

    def test_modeled_frontier_exhausted_requires_no_wait_and_no_unblocked_frontier(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)

        self.assertTrue(agent._modeled_frontier_exhausted(frame, [2, 4], frame_hash=agent._fast_frame_hash(frame)))
        self.assertFalse(agent._modeled_frontier_exhausted(frame, [2, 4, 5], frame_hash=agent._fast_frame_hash(frame)))

    def test_modeled_frontier_exhausted_treats_wait_as_exhausted_after_stale_wait(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()

        self.assertTrue(agent._modeled_frontier_exhausted(frame, [2, 4, 5], frame_hash=agent._fast_frame_hash(frame)))

    def test_modeled_frontier_exhausted_reuses_cached_result(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)

        first = agent._modeled_frontier_exhausted(frame, [2, 4], frame_hash=frame_hash)

        with mock.patch.object(
                agent,
                "_all_legal_dirs_blocked",
                side_effect=AssertionError("cached result should skip blocked-direction scan")):
            second = agent._modeled_frontier_exhausted(frame.copy(), [2, 4], frame_hash=frame_hash)

        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_choose_action_prefers_undo_when_modeled_frontier_is_exhausted(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION7], levels=0)
        logits = np.array([-10.0, 9.0, -10.0, 8.5, -2.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent._ckpt_hash = 99
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo-frontier")

    def test_choose_action_prefers_undo_after_stale_wait_recovery(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION5, _GameAction.ACTION7], levels=0)
        logits = np.array([-10.0, 9.0, -10.0, 8.5, 7.5], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent._ckpt_hash = 99
        agent._unproductive = 6
        agent.pt = object()
        agent.pai = 4
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 7)
        self.assertEqual(result.reasoning, "undo-frontier")

    def test_retry_blocked_direction_after_stale_wait_requires_no_other_frontier(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()
        agent._remember_blocked_direction_index(1)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}

        self.assertTrue(agent._retry_blocked_direction_after_stale_wait(frame, [2, 5], frame_hash=agent._fast_frame_hash(frame)))
        self.assertFalse(agent._retry_blocked_direction_after_stale_wait(frame, [2, 4, 5], frame_hash=agent._fast_frame_hash(frame)))

    def test_cnn_rescoring_retries_blocked_direction_after_stale_wait(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION5], levels=0)
        logits = np.array([-10.0, 8.0, -10.0, -10.0, 9.5], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent._unproductive = 6
        agent.pt = object()
        agent.pai = 4
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)
        agent._remember_blocked_direction_index(1)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "cnn:a2")

    def test_cnn_rescoring_breaks_equal_direction_ties_with_recent_momentum(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, 5.0, -10.0, 5.0, -10.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None
        agent.pt = object()
        agent.pai = 3
        agent.pr = np.ones((64, 64), dtype=np.uint8)
        agent.ph = 0

        with mock.patch.object(self.mod.random, "random", return_value=1.0):
            result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_cnn_rescoring_avoids_recently_blocked_direction_when_scores_are_close(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, 8.25, -10.0, 8.2, -10.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None
        agent.pt = object()
        agent.pai = 1
        agent.pr = frame.frame[-1].copy()
        agent.ph = 0

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)

    def test_cnn_rescoring_excludes_blocked_direction_even_with_hotter_logit(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, 8.22, -10.0, 8.2, -10.0], dtype=np.float32)

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pai = 1
        agent.pr = frame.frame[-1].copy()

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_cnn_rescoring_considers_semantic_direction_outside_raw_topk(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        for rank, (y, x) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]):
            logits[5 + y * agent.G + x] = 8.7 - rank * 0.1
        logits[3] = 8.2

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_cnn_rescoring_can_choose_legal_secondary_semantic_direction(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION3], levels=0)
        logits = np.array([-10.0, 7.9, 8.0, -10.0, -10.0], dtype=np.float32)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (20.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (30.0, 20.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 2)
        self.assertEqual(result.reasoning, "cnn:a2")

    def test_cnn_rescoring_all_negative_scores_still_picks_best_semantic_direction(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        logits = np.array([-10.0, -1.8, -10.0, -1.9, -10.0], dtype=np.float32)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_cnn_rescoring_prefers_semantic_click_without_bfs(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[5 + 10 * agent.G + 20] = 8.25
        logits[5 + 32 * agent.G + 42] = 8.0

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (32.0, 42.0), "cell_count": 8}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})
        self.assertEqual(agent._semantic_target_coord, (32, 42))

    def test_cnn_rescoring_avoids_recently_blocked_click_when_scores_are_close(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        blocked_idx = 5 + 20 * agent.G + 23
        better_idx = 5 + 22 * agent.G + 40
        logits[blocked_idx] = 8.4
        logits[better_idx] = 8.2

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                    "6": [{"center": (20.0, 23.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pai = blocked_idx
        agent.pr = frame.frame[-1].copy()

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})

    def test_cnn_rescoring_avoids_recent_blocked_click_history_even_with_hotter_logit(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        frame.frame[-1][18:20, 18:20] = 3
        frame.frame[-1][30:32, 40:42] = 5
        blocked_idx = 5 + 18 * agent.G + 18
        safe_idx = 5 + 30 * agent.G + 40
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[blocked_idx] = 9.0
        logits[safe_idx] = 8.0

        agent._remember_blocked_click_coord((18, 18))
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _ForwardOnlyNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 30})

    def test_cnn_rescoring_avoids_nearby_blocked_click_jitter_when_scores_are_close(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        near_blocked_idx = 5 + 19 * agent.G + 24
        better_idx = 5 + 22 * agent.G + 40
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[near_blocked_idx] = 8.35
        logits[better_idx] = 8.2

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [
                        {"center": (19.0, 24.0), "cell_count": 4},
                        {"center": (22.0, 40.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pai = 5 + 18 * agent.G + 23
        agent.pr = frame.frame[-1].copy()

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})

    def test_cnn_rescoring_excludes_blocked_click_jitter_even_with_hotter_logit(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        near_blocked_idx = 5 + 19 * agent.G + 24
        better_idx = 5 + 22 * agent.G + 40
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[near_blocked_idx] = 8.22
        logits[better_idx] = 8.2

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pai = 5 + 18 * agent.G + 23
        agent.pr = frame.frame[-1].copy()

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})

    def test_cnn_rescoring_does_not_boost_preferred_click_in_blocked_jitter_region(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        blocked_preferred = (19, 24)
        blocked_idx = 5 + blocked_preferred[0] * agent.G + blocked_preferred[1]
        better_idx = 5 + 22 * agent.G + 40
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[blocked_idx] = 8.31
        logits[better_idx] = 8.20

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent._semantic_target_coord = blocked_preferred
        agent._semantic_click_targets = lambda raw, limit=6: []
        agent.pai = 5 + 18 * agent.G + 23
        agent.pr = frame.frame[-1].copy()

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})

    def test_cnn_rescoring_considers_semantic_click_outside_raw_topk(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        hot_cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for rank, (y, x) in enumerate(hot_cells):
            logits[5 + y * agent.G + x] = 8.6 - rank * 0.1
        logits[5 + 32 * agent.G + 42] = 8.0

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (32.0, 42.0), "cell_count": 8}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})

    def test_cnn_rescoring_all_negative_scores_still_picks_best_semantic_click(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        logits = np.full(4101, -10.0, dtype=np.float32)
        hot_cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for rank, (y, x) in enumerate(hot_cells):
            logits[5 + y * agent.G + x] = -1.5 - rank * 0.1
        logits[5 + 32 * agent.G + 42] = -1.95

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "14": [{"center": (32.0, 42.0), "cell_count": 8}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})

    def test_cnn_rescoring_breaks_equal_click_ties_with_recent_click_momentum(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        first_idx = 5 + 10 * agent.G + 20
        second_idx = 5 + 32 * agent.G + 42
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[first_idx] = 5.0
        logits[second_idx] = 5.0

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pt = object()
        agent.pai = second_idx
        agent.pr = np.ones((64, 64), dtype=np.uint8)
        agent.ph = 0

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})

    def test_cnn_rescoring_penalizes_repeating_recent_click_when_unproductive(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        first_idx = 5 + 10 * agent.G + 20
        second_idx = 5 + 32 * agent.G + 42
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[first_idx] = 5.0
        logits[second_idx] = 5.0

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent._unproductive = 6
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent.pai = second_idx
        agent.pr = frame.frame[-1].copy()
        agent.ph = agent._fast_frame_hash(agent.pr)

        with mock.patch.object(agent, "_semantic_direct_click_choice", return_value=None):
            with mock.patch.object(agent, "_semantic_target_choice", return_value=None):
                with mock.patch.object(agent, "_semantic_click_bonus_scale", return_value=0.0):
                    with mock.patch.object(agent, "_semantic_click_bonus_map", return_value={}):
                        with mock.patch.object(agent, "_semantic_click_targets_compat", return_value=[(10, 20), (32, 42)]):
                            with mock.patch.object(agent, "_semantic_click_candidate_indices", return_value=[first_idx, second_idx]):
                                result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 20, "y": 10})

    def test_cnn_rescoring_considers_preferred_click_target_outside_semantic_top_list(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        preferred = (32, 42)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        logits = np.full(4101, -10.0, dtype=np.float32)
        for rank, (y, x) in enumerate(decoys):
            logits[5 + y * agent.G + x] = 7.20 - rank * 0.01
        logits[preferred_idx] = 8.04

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=6: decoys[:limit]

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})

    def test_cnn_rescoring_considers_raw_fallback_click_target_outside_top_list(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        frame.frame[-1][18:20, 18:20] = 3
        frame.frame[-1][30:32, 40:42] = 5
        preferred = (30, 40)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]
        decoys = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        logits = np.full(4101, -10.0, dtype=np.float32)
        for rank, (y, x) in enumerate(decoys):
            logits[5 + y * agent.G + x] = 7.20 - rank * 0.01
        logits[preferred_idx] = 7.16

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent._semantic_target_coord = preferred

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 30})

    def test_epsilon_exploration_uses_semantic_biases(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 1.0
        agent.net = _ForwardOnlyNet(np.zeros(5, dtype=np.float32), agent.device)
        agent._bfs = None

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            with mock.patch.object(self.mod.torch, "multinomial", return_value=torch.tensor([3], device=agent.device)):
                result = agent.choose_action([], frame)

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

    def test_sample_semantic_exploration_restricts_clicks_to_semantic_candidates(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        semantic_coord = (22, 40)
        decoy_coord = (10, 10)
        semantic_idx = 5 + semantic_coord[0] * agent.G + semantic_coord[1]
        decoy_idx = 5 + decoy_coord[0] * agent.G + decoy_coord[1]
        logits[semantic_idx] = 0.7
        logits[decoy_idx] = 0.9

        agent._semantic_click_targets = lambda raw, limit=8, blocked_click_coord=None, frame_hash=None: [semantic_coord]

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 1)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
            action_idx, coords = agent._sample_semantic_exploration(
                logits,
                frame,
                [_GameAction.ACTION6],
                avail_ids=[6],
            )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, semantic_coord)

    def test_sample_semantic_exploration_includes_raw_fallback_click_candidates(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[18:20, 18:20] = 3
        frame[30:32, 40:42] = 5
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        fallback_coord = (30, 40)
        other_fallback_coord = (18, 18)
        decoy_coord = (10, 10)
        fallback_idx = 5 + fallback_coord[0] * agent.G + fallback_coord[1]
        other_fallback_idx = 5 + other_fallback_coord[0] * agent.G + other_fallback_coord[1]
        decoy_idx = 5 + decoy_coord[0] * agent.G + decoy_coord[1]
        logits[fallback_idx] = 0.7
        logits[other_fallback_idx] = -float("inf")
        logits[decoy_idx] = 0.9
        agent._semantic_detector = lambda grid: {"components_per_value": {}}

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            action_idx, coords = agent._sample_semantic_exploration(
                logits,
                frame,
                [_GameAction.ACTION6],
                avail_ids=[6],
            )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, fallback_coord)

    def test_sample_semantic_exploration_excludes_preferred_click_from_recent_history_when_current_block_differs(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        preferred = (19, 12)
        decoy_coord = (10, 10)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]
        decoy_idx = 5 + decoy_coord[0] * agent.G + decoy_coord[1]
        logits[preferred_idx] = 0.9
        logits[decoy_idx] = 0.7

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=8, blocked_click_coord=None, frame_hash=None: [decoy_coord]
        agent._remember_blocked_click_coord(preferred)

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 1)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
            action_idx, coords = agent._sample_semantic_exploration(
                logits,
                frame,
                [_GameAction.ACTION6],
                avail_ids=[6],
                blocked_click_coord=(0, 0),
                frame_hash=agent._fast_frame_hash(frame),
            )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, decoy_coord)

    def test_sample_semantic_exploration_excludes_stale_preferred_click_target(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        logits = torch.zeros(4101, dtype=torch.float32, device=agent.device)
        preferred = (32, 42)
        decoy_coord = (10, 10)
        preferred_idx = 5 + preferred[0] * agent.G + preferred[1]
        decoy_idx = 5 + decoy_coord[0] * agent.G + decoy_coord[1]
        logits[preferred_idx] = 0.9
        logits[decoy_idx] = 0.7

        agent._semantic_target_coord = preferred
        agent._semantic_click_targets = lambda raw, limit=8, blocked_click_coord=None, frame_hash=None: [decoy_coord]
        agent._unproductive = 6

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 1)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
            action_idx, coords = agent._sample_semantic_exploration(
                logits,
                frame,
                [_GameAction.ACTION6],
                avail_ids=[6],
                frame_hash=agent._fast_frame_hash(frame),
            )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, decoy_coord)

    def test_sample_sparse_policy_indices_limits_click_mass_to_candidates(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((4101,), -float("inf"), dtype=torch.float32, device=agent.device)
        allowed_click = (22, 40)
        blocked_click = (10, 10)
        allowed_idx = 5 + allowed_click[0] * agent.G + allowed_click[1]
        blocked_idx = 5 + blocked_click[0] * agent.G + blocked_click[1]
        logits[allowed_idx] = 0.9
        logits[blocked_idx] = 0.95

        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            action_idx, coords = agent._sample_sparse_policy_indices(
                logits,
                [6],
                [allowed_idx],
            )

        self.assertEqual(action_idx, 5)
        self.assertEqual(coords, allowed_click)

    def test_sample_sparse_policy_indices_reuses_cached_distribution(self):
        import torch

        agent = self.make_agent()
        logits = torch.full((4101,), -float("inf"), dtype=torch.float32, device=agent.device)
        click_coord = (22, 40)
        click_idx = 5 + click_coord[0] * agent.G + click_coord[1]
        logits[1] = 0.5
        logits[click_idx] = 0.9

        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            first = agent._sample_sparse_policy_indices(
                logits,
                [2, 6],
                [click_idx],
            )
            second = agent._sample_sparse_policy_indices(
                logits,
                [2, 6],
                [click_idx],
            )

        self.assertEqual(first, second)
        self.assertIsNotNone(agent._sample_sparse_policy_cache_key)
        self.assertIsNotNone(agent._sample_sparse_policy_cache_value)

    def test_candidate_score_map_batches_candidate_reads(self):
        import torch

        agent = self.make_agent()
        scored = torch.tensor([0.2, -1.0, 3.5, 0.7], dtype=torch.float32, device=agent.device)

        score_map = agent._candidate_score_map(scored, [2, 0, 3])

        self.assertEqual(set(score_map.keys()), {2, 0, 3})
        self.assertAlmostEqual(score_map[2], 3.5, places=6)
        self.assertAlmostEqual(score_map[0], 0.2, places=6)
        self.assertAlmostEqual(score_map[3], 0.7, places=6)

    def test_candidate_score_map_reuses_cached_mapping_and_invalidates_on_tensor_change(self):
        import torch

        agent = self.make_agent()
        scored = torch.tensor([0.2, -1.0, 3.5, 0.7], dtype=torch.float32, device=agent.device)

        first = agent._candidate_score_map(scored, [2, 0, 3])
        self.assertIsNotNone(agent._candidate_score_map_cache_key)
        self.assertIs(first, agent._candidate_score_map_cache_value)

        second = agent._candidate_score_map(scored, [2, 0, 3])
        self.assertIs(first, second)

        scored[2] = 4.5
        third = agent._candidate_score_map(scored, [2, 0, 3])
        self.assertIsNot(first, third)
        self.assertAlmostEqual(third[2], 4.5, places=6)

    def test_candidate_scores_preserve_candidate_order(self):
        import torch

        agent = self.make_agent()
        scored = torch.tensor([0.2, -1.0, 3.5, 0.7], dtype=torch.float32, device=agent.device)

        candidate_scores = agent._candidate_scores(scored, [2, 0, 3])

        self.assertEqual(len(candidate_scores), 3)
        self.assertAlmostEqual(candidate_scores[0], 3.5, places=6)
        self.assertAlmostEqual(candidate_scores[1], 0.2, places=6)
        self.assertAlmostEqual(candidate_scores[2], 0.7, places=6)

    def test_candidate_scores_reuse_cached_values_and_invalidate_on_tensor_change(self):
        import torch

        agent = self.make_agent()
        scored = torch.tensor([0.2, -1.0, 3.5, 0.7], dtype=torch.float32, device=agent.device)

        first = agent._candidate_scores(scored, [2, 0, 3])
        self.assertIsNotNone(agent._candidate_scores_cache_key)
        self.assertEqual(tuple(first), agent._candidate_scores_cache_value)

        second = agent._candidate_scores(scored, [2, 0, 3])
        self.assertEqual(first, second)

        scored[2] = 4.5
        third = agent._candidate_scores(scored, [2, 0, 3])
        self.assertAlmostEqual(third[0], 4.5, places=6)

    def test_click_candidate_context_map_batches_coords_and_static_click_bonuses(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        click_a = 5 + 18 * agent.G + 12
        click_b = 5 + 22 * agent.G + 40
        agent._wm = np.ones((64, 64), dtype=np.float32)
        semantic_bonus_map = {(18, 12): 0.3, (22, 40): 0.15}

        with mock.patch.object(
                agent,
                "_blocked_click_matches_coord",
                side_effect=[True, False]) as blocked_mock:
            with mock.patch.object(
                    agent,
                    "_bfs_click_priority_bonus",
                    side_effect=[0.25, 0.5]) as bfs_mock:
                context = agent._click_candidate_context_map(
                    frame,
                    [2, click_a, click_b, click_a],
                    blocked_click_coord=(18, 12),
                    frame_hash=agent._fast_frame_hash(frame),
                    preferred_click_coord=(22, 40),
                    semantic_click_bonus_map=semantic_bonus_map,
                    repeat_click_idx=click_b,
                    blocked_click_idx=click_a,
                    continuity_scale=1.0,
                )
        self.assertEqual(context[click_a]["coord"], (18, 12))
        self.assertTrue(context[click_a]["blocked"])
        self.assertAlmostEqual(context[click_a]["bfs_bonus"], 0.25, places=6)
        self.assertAlmostEqual(context[click_a]["preferred_bonus"], 0.0, places=6)
        self.assertAlmostEqual(context[click_a]["semantic_bonus"], 0.3, places=6)
        self.assertAlmostEqual(context[click_a]["repeat_bonus"], 0.0, places=6)
        self.assertAlmostEqual(context[click_a]["wm_bonus"], 0.05, places=6)
        self.assertTrue(context[click_a]["is_blocked_idx"])
        self.assertEqual(context[click_b]["coord"], (22, 40))
        self.assertFalse(context[click_b]["blocked"])
        self.assertAlmostEqual(context[click_b]["bfs_bonus"], 0.5, places=6)
        self.assertAlmostEqual(context[click_b]["preferred_bonus"], 0.08, places=6)
        self.assertAlmostEqual(context[click_b]["semantic_bonus"], 0.15, places=6)
        self.assertAlmostEqual(context[click_b]["repeat_bonus"], 0.08, places=6)
        self.assertAlmostEqual(context[click_b]["wm_bonus"], 0.05, places=6)
        self.assertFalse(context[click_b]["is_blocked_idx"])
        self.assertEqual(blocked_mock.call_count, 2)
        self.assertEqual(bfs_mock.call_count, 2)

    def test_click_candidate_context_map_reuses_cached_mapping(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        click_a = 5 + 18 * agent.G + 12
        click_b = 5 + 22 * agent.G + 40
        frame_hash = agent._fast_frame_hash(frame)
        agent._wm = np.ones((64, 64), dtype=np.float32)
        semantic_bonus_map = {(18, 12): 0.3, (22, 40): 0.15}

        with mock.patch.object(
                agent,
                "_blocked_click_matches_coord",
                side_effect=[True, False]) as blocked_mock:
            with mock.patch.object(
                    agent,
                    "_bfs_click_priority_bonus",
                    side_effect=[0.25, 0.5]) as bfs_mock:
                first = agent._click_candidate_context_map(
                    frame,
                    [2, click_a, click_b, click_a],
                    blocked_click_coord=(18, 12),
                    frame_hash=frame_hash,
                    preferred_click_coord=(22, 40),
                    semantic_click_bonus_map=semantic_bonus_map,
                    repeat_click_idx=click_b,
                    blocked_click_idx=click_a,
                    continuity_scale=1.0,
                )
                second = agent._click_candidate_context_map(
                    frame,
                    [2, click_a, click_b, click_a],
                    blocked_click_coord=(18, 12),
                    frame_hash=frame_hash,
                    preferred_click_coord=(22, 40),
                    semantic_click_bonus_map=semantic_bonus_map,
                    repeat_click_idx=click_b,
                    blocked_click_idx=click_a,
                    continuity_scale=1.0,
                )

        self.assertIs(first, second)
        self.assertEqual(blocked_mock.call_count, 2)
        self.assertEqual(bfs_mock.call_count, 2)
        self.assertIsNotNone(agent._click_candidate_context_cache_key)
        self.assertIs(first, agent._click_candidate_context_cache_value)

    def test_click_candidate_context_map_cached_hit_skips_click_coord_decode(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        click_a = 5 + 18 * agent.G + 12
        click_b = 5 + 22 * agent.G + 40
        agent._wm = np.ones((64, 64), dtype=np.float32)
        semantic_bonus_map = agent._semantic_click_bonus_map(
            frame,
            limit=6,
            click_scale=1.0,
            click_targets=[(18, 12), (22, 40)],
            blocked_click_coord=None,
            frame_hash=frame_hash,
        )

        first = agent._click_candidate_context_map(
            frame,
            [click_a, click_b],
            frame_hash=frame_hash,
            preferred_click_coord=(22, 40),
            semantic_click_bonus_map=semantic_bonus_map,
            repeat_click_idx=click_b,
            blocked_click_idx=click_a,
            continuity_scale=1.0,
        )

        with mock.patch.object(
                agent,
                "_click_coord_from_action_index",
                side_effect=AssertionError("cache hit should not decode click coords")):
            second = agent._click_candidate_context_map(
                frame,
                [click_a, click_b],
                frame_hash=frame_hash,
                preferred_click_coord=(22, 40),
                semantic_click_bonus_map=semantic_bonus_map,
                repeat_click_idx=click_b,
                blocked_click_idx=click_a,
                continuity_scale=1.0,
            )

        self.assertIs(first, second)

    def test_direction_candidate_context_map_batches_unique_directions(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        semantic_dirs = {0: 0.45, 2: 0.18, 4: 0.3}

        with mock.patch.object(
                agent,
                "_direction_matches_blocked_history",
                side_effect=[True, False]) as blocked_mock:
            with mock.patch.object(
                    agent,
                    "_bfs_priority_bonus",
                    side_effect=[0.1, 0.3, 0.6]) as bfs_mock:
                context = agent._direction_candidate_context_map(
                    frame,
                    [0, 2, 0, 4, 5, 2],
                    frame_hash=agent._fast_frame_hash(frame),
                    blocked_direction=0,
                    semantic_dirs=semantic_dirs,
                    repeat_direction_idx=2,
                    wait_recovery_bonus=0.3,
                )

        self.assertEqual(context, {
            0: {"blocked": True, "bfs_bonus": 0.1, "semantic_bonus": 0.45, "repeat_bonus": 0.0, "wait_bonus": 0.0},
            2: {"blocked": False, "bfs_bonus": 0.3, "semantic_bonus": 0.18, "repeat_bonus": 0.08, "wait_bonus": 0.0},
            4: {"blocked": False, "bfs_bonus": 0.6, "semantic_bonus": 0.3, "repeat_bonus": 0.0, "wait_bonus": 0.3},
        })
        self.assertEqual(blocked_mock.call_count, 2)
        self.assertEqual(bfs_mock.call_count, 3)

    def test_epsilon_exploration_clicks_use_sparse_semantic_candidates(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [{"center": (22.0, 40.0), "cell_count": 4}],
                }
            }

        semantic_idx = 5 + 22 * agent.G + 40
        decoy_idx = 5 + 10 * agent.G + 10

        agent._semantic_detector = fake_detector
        agent.cl = 0
        agent._wd = True
        agent._eps = 1.0
        agent.net = _FixedLogitNet(np.zeros(4101, dtype=np.float32), agent.device)
        agent._bfs = None

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 1)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
                result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})
        self.assertEqual(result.reasoning, "cnn:c(40,22)")

    def test_epsilon_exploration_clicks_can_use_raw_fallback_candidates(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        frame.frame[-1][18:20, 18:20] = 3
        frame.frame[-1][30:32, 40:42] = 5
        fallback_idx = 5 + 30 * agent.G + 40
        decoy_idx = 5 + 10 * agent.G + 10

        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.cl = 0
        agent._wd = True
        agent._eps = 1.0
        agent.net = _FixedLogitNet(np.zeros(4101, dtype=np.float32), agent.device)
        agent._bfs = None

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 2)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
                result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 30})
        self.assertEqual(result.reasoning, "cnn:c(40,30)")

    def test_epsilon_exploration_clicks_skip_dense_exploration_logits_on_sparse_shortlist(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        semantic_coord = (22, 40)

        agent._semantic_click_targets = lambda raw, limit=8, blocked_click_coord=None, frame_hash=None: [semantic_coord]
        agent.cl = 0
        agent._wd = True
        agent._eps = 1.0
        agent.net = _FixedLogitNet(np.zeros(4101, dtype=np.float32), agent.device)
        agent._bfs = None

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 1)
            self.assertGreater(float(probs[0].item()), 0.0)
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(agent, "_semantic_exploration_logits", side_effect=AssertionError("should stay sparse")):
            with mock.patch.object(self.mod.random, "random", return_value=0.0):
                with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
                    result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 22})
        self.assertEqual(result.reasoning, "cnn:c(40,22)")

    def test_epsilon_exploration_prefers_stronger_raw_fallback_click_logit(self):
        import torch

        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        frame.frame[-1][18:20, 18:20] = 3
        frame.frame[-1][30:32, 40:42] = 5
        first_idx = 5 + 18 * agent.G + 18
        second_idx = 5 + 30 * agent.G + 40
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[first_idx] = 7.0
        logits[second_idx] = 7.3

        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        agent.cl = 0
        agent._wd = True
        agent._eps = 1.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = None

        def fake_multinomial(probs, num_samples):
            self.assertEqual(int(probs.numel()), 2)
            self.assertGreater(float(probs[0].item()), float(probs[1].item()))
            return torch.tensor([0], device=agent.device)

        with mock.patch.object(self.mod.random, "random", return_value=0.0):
            with mock.patch.object(self.mod.torch, "multinomial", side_effect=fake_multinomial):
                result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 40, "y": 30})

    def test_cnn_rescoring_prefers_semantic_click_target(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION6], levels=0)
        frame.frame[-1][10:12, 20:22] = 3
        frame.frame[-1][30:34, 40:44] = 14
        logits = np.full(4101, -10.0, dtype=np.float32)
        logits[5 + 10 * agent.G + 20] = 8.25
        logits[5 + 32 * agent.G + 42] = 8.0

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = types.SimpleNamespace(
            _action_key=lambda act_id, data: (act_id, None if not data else (data.get("x"), data.get("y"))),
            _action_priority={},
        )
        agent._wm = np.ones((64, 64), dtype=np.float32)
        agent._bg = 0

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": 42, "y": 32})

    def test_heuristic_returns_action5_when_only_action5_available(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION5], step=9)

        self.assertEqual(action_idx, 4)
        self.assertIsNone(coords)

    def test_heuristic_prefers_continuing_legal_direction_over_wait(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        previous = np.zeros((64, 64), dtype=np.uint8)
        previous[0, 0] = 1
        agent._bg = 0
        agent.pai = 3
        agent.pr = previous

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION4, _GameAction.ACTION5], step=9)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)

    def test_heuristic_random_fallback_avoids_wait_when_direction_exists(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0

        with mock.patch.object(self.mod.random, "choice", return_value=2) as choice_mock:
            action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION5], step=9)

        self.assertEqual(action_idx, 1)
        self.assertIsNone(coords)
        self.assertEqual(choice_mock.call_args.args[0], [2])

    def test_heuristic_random_fallback_avoids_known_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        agent.pai = 1
        agent.pr = frame.copy()

        with mock.patch.object(self.mod.random, "choice", return_value=4) as choice_mock:
            action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION4], step=9)

        self.assertEqual(action_idx, 3)
        self.assertIsNone(coords)
        self.assertEqual(choice_mock.call_args.args[0], [4])

    def test_heuristic_prefers_wait_over_retrying_only_blocked_direction(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        agent.pai = 1
        agent.pr = frame.copy()

        action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION5], step=9)

        self.assertEqual(action_idx, 4)
        self.assertIsNone(coords)

    def test_heuristic_retries_blocked_direction_after_stale_wait(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        agent._unproductive = 7
        agent.pai = 4
        agent.pr = frame.copy()
        agent._remember_blocked_direction_index(1)

        with mock.patch.object(self.mod.random, "choice", return_value=2) as choice_mock:
            action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2, _GameAction.ACTION5], step=9)

        self.assertEqual(action_idx, 1)
        self.assertIsNone(coords)
        self.assertEqual(choice_mock.call_args.args[0], [2])

    def test_prime_warmup_action_exits_early_for_wait_recovery(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.la = 3
        agent._unproductive = 6
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        train_calls = []
        agent._maybe_train = lambda max_steps=0, force=False: train_calls.append((max_steps, force))

        result = agent._prime_warmup_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4, _GameAction.ACTION5], frame_hash=agent._fast_frame_hash(frame))

        self.assertIsNone(result)
        self.assertTrue(agent._wd)
        self.assertEqual(train_calls, [(0, True)])

    def test_prime_warmup_action_exits_early_for_exhausted_frontier(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.la = 3
        agent._unproductive = 6
        agent._remember_blocked_direction_index(1)
        agent._remember_blocked_direction_index(3)
        agent._semantic_detector = lambda grid: {"components_per_value": {}}
        train_calls = []
        agent._maybe_train = lambda max_steps=0, force=False: train_calls.append((max_steps, force))

        result = agent._prime_warmup_action(frame, [_GameAction.ACTION2, _GameAction.ACTION4], frame_hash=agent._fast_frame_hash(frame))

        self.assertIsNone(result)
        self.assertTrue(agent._wd)
        self.assertEqual(train_calls, [(0, True)])

    def test_heuristic_retries_blocked_direction_when_no_wait_exists(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0
        agent.pai = 1
        agent.pr = frame.copy()

        with mock.patch.object(self.mod.random, "choice", return_value=2) as choice_mock:
            action_idx, coords = agent._heuristic(frame, [_GameAction.ACTION2], step=9)

        self.assertEqual(action_idx, 1)
        self.assertIsNone(coords)
        self.assertEqual(choice_mock.call_args.args[0], [2])

    def test_heuristic_defaults_to_zero_when_no_known_actions_available(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent._bg = 0

        action_idx, coords = agent._heuristic(frame, [], step=9)

        self.assertEqual(action_idx, 0)
        self.assertIsNone(coords)

    def test_clear_replay_keep_fraction_retains_highest_reward_entries(self):
        agent = self.make_agent()
        agent.bsz = 2
        agent.buf = [np.full((64, 64), i, dtype=np.uint8) for i in range(5)]
        agent.buf_actions = self.mod.array('H', [0, 1, 2, 3, 4])
        agent.buf_rewards = self.mod.array('f', [0.1, 2.0, 1.5, 0.2, 3.0])
        agent.buf_next_frames = [None] * 5
        agent.buf_has_next = self.mod.array('b', [0, 0, 0, 0, 0])
        agent.buf_priorities = self.mod.array('f', [0.11, 2.01, 1.51, 0.21, 3.01])
        agent.buf_keys = [("k0", 0), ("k1", 1), None, ("k3", 3), ("k4", 4)]
        agent.buf_key_counts = {("stale", 9): 1}
        agent.buf_h = {("old", 1)}
        agent.buf_pos = 4

        agent._clear_replay(keep_frac=0.4)

        self.assertEqual(len(agent.buf), 2)
        self.assertEqual(sorted(float(x) for x in agent.buf_rewards), [2.0, 3.0])
        self.assertEqual(list(agent.buf_actions), [1, 4])
        self.assertEqual(agent.buf_keys, [("k1", 1), ("k4", 4)])
        self.assertEqual(agent.buf_key_counts, {("k1", 1): 1, ("k4", 4): 1})
        self.assertEqual(agent.buf_h, {("k1", 1), ("k4", 4)})
        self.assertEqual(agent.buf_pos, 0)

    def test_clear_replay_full_clear_resets_all_buffers(self):
        agent = self.make_agent()
        agent.buf = [np.ones((64, 64), dtype=np.uint8)]
        agent.buf_actions = self.mod.array('H', [3])
        agent.buf_rewards = self.mod.array('f', [1.25])
        agent.buf_next_frames = [np.zeros((64, 64), dtype=np.uint8)]
        agent.buf_has_next = self.mod.array('b', [1])
        agent.buf_priorities = self.mod.array('f', [1.26])
        agent.buf_keys = [("seen", 3)]
        agent.buf_hashes = self.mod.array('I', [123456789])
        agent.buf_key_counts = {("seen", 3): 1}
        agent.buf_h = {("seen", 3)}
        agent.buf_pos = 7

        agent._clear_replay(keep_frac=0.0)

        self.assertEqual(agent.buf, [])
        self.assertEqual(list(agent.buf_actions), [])
        self.assertEqual(list(agent.buf_rewards), [])
        self.assertEqual(agent.buf_next_frames, [])
        self.assertEqual(list(agent.buf_has_next), [])
        self.assertEqual(list(agent.buf_priorities), [])
        self.assertEqual(agent.buf_keys, [])
        self.assertEqual(list(agent.buf_hashes), [])
        self.assertEqual(agent.buf_key_counts, {})
        self.assertEqual(agent.buf_h, set())
        self.assertEqual(agent.buf_pos, 0)

    def test_clear_replay_keeps_small_buffer_intact(self):
        agent = self.make_agent()
        agent.bsz = 4
        frames = [np.full((64, 64), i, dtype=np.uint8) for i in range(3)]
        agent.buf = list(frames)
        agent.buf_actions = self.mod.array('H', [0, 1, 2])
        agent.buf_rewards = self.mod.array('f', [0.1, 0.2, 0.3])
        agent.buf_next_frames = [None, None, None]
        agent.buf_has_next = self.mod.array('b', [0, 0, 0])
        agent.buf_priorities = self.mod.array('f', [0.11, 0.21, 0.31])
        agent.buf_keys = [("k0", 0), None, ("k2", 2)]
        agent.buf_hashes = self.mod.array('I', [11, 12, 13])
        agent.buf_key_counts = {("keep", 1): 1}
        agent.buf_h = {("keep", 1)}
        agent.buf_pos = 2

        agent._clear_replay(keep_frac=0.5)

        self.assertEqual(len(agent.buf), 3)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(agent.buf, frames)))
        self.assertEqual(list(agent.buf_actions), [0, 1, 2])
        self.assertTrue(np.allclose(list(agent.buf_rewards), [0.1, 0.2, 0.3]))
        self.assertEqual(list(agent.buf_has_next), [0, 0, 0])
        self.assertEqual(list(agent.buf_hashes), [11, 12, 13])
        self.assertEqual(agent.buf_key_counts, {("keep", 1): 1})
        self.assertEqual(agent.buf_h, {("keep", 1)})
        self.assertEqual(agent.buf_keys, [("k0", 0), None, ("k2", 2)])
        self.assertEqual(agent.buf_pos, 2)

    def test_level_change_clears_dedup_state_even_when_small_replay_is_retained(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1], levels=0)
        agent.cl = -1
        agent._bfs_tried = True
        agent._bfs = None
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()
        agent.bsz = 4
        agent.buf = [np.zeros((64, 64), dtype=np.uint8) for _ in range(3)]
        agent.buf_actions = self.mod.array('H', [0, 1, 2])
        agent.buf_rewards = self.mod.array('f', [0.1, 0.2, 0.3])
        agent.buf_next_frames = [None, None, None]
        agent.buf_has_next = self.mod.array('b', [0, 0, 0])
        agent.buf_priorities = self.mod.array('f', [0.11, 0.21, 0.31])
        agent.buf_keys = [("k0", 0), None, ("k2", 2)]
        agent.buf_h = {("stale", 7)}

        result = agent.choose_action([], frame)

        self.assertEqual(result.value, 1)
        self.assertEqual(len(agent.buf), 3)
        self.assertEqual(agent.buf_h, set())

    def test_add_replay_boosts_recent_predecessor_rewards(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        for reward in [0.0, 0.0, 0.0]:
            agent._add_replay(frame, 1, reward)

        agent._add_replay(frame, 2, 2.0)

        rewards = list(agent.buf_rewards)
        self.assertAlmostEqual(rewards[-1], 2.0, places=5)
        self.assertGreater(rewards[-2], 0.0)
        self.assertGreater(rewards[-3], 0.0)
        self.assertGreater(rewards[-4], 0.0)
        self.assertAlmostEqual(agent.buf_priorities[-2], abs(rewards[-2]) + 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[-3], abs(rewards[-3]) + 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[-4], abs(rewards[-4]) + 0.01, places=5)

    def test_add_replay_overwrites_circular_buffer_and_clamps_action_index(self):
        agent = self.make_agent()
        agent.buf_max = 2
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)
        frame2 = np.full((64, 64), 2, dtype=np.uint8)

        agent._add_replay(frame0, 1, 0.1, dedup_key=("f0", 1))
        agent._add_replay(frame1, 2, 0.2, dedup_key=("f1", 2))
        agent._add_replay(frame2, 999999, 0.3, dedup_key=("f2", 65535))

        self.assertEqual(len(agent.buf), 2)
        self.assertEqual(agent.buf_pos, 1)
        self.assertEqual(list(agent.buf_actions), [65535, 2])
        self.assertTrue(np.array_equal(agent.buf[0], frame2))
        self.assertTrue(np.array_equal(agent.buf[1], frame1))
        self.assertEqual(list(agent.buf_has_next), [0, 0])
        self.assertEqual(agent.buf_keys, [("f2", 65535), ("f1", 2)])
        self.assertEqual(
            list(agent.buf_hashes),
            [agent._fast_frame_hash(frame2), agent._fast_frame_hash(frame1)],
        )
        self.assertEqual(agent.buf_key_counts, {("f1", 2): 1, ("f2", 65535): 1})
        self.assertEqual(agent.buf_h, {("f2", 65535), ("f1", 2)})

    def test_add_replay_overwrite_removes_evicted_dedup_key(self):
        agent = self.make_agent()
        agent.buf_max = 1
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)

        agent._add_replay(frame0, 1, 0.1, dedup_key=("f0", 1))
        agent._add_replay(frame1, 2, 0.2, dedup_key=("f1", 2))

        self.assertEqual(agent.buf_keys, [("f1", 2)])
        self.assertEqual(list(agent.buf_has_next), [0])
        self.assertEqual(agent.buf_key_counts, {("f1", 2): 1})
        self.assertEqual(agent.buf_h, {("f1", 2)})

    def test_add_replay_overwrite_preserves_shared_dedup_key_membership(self):
        agent = self.make_agent()
        agent.buf_max = 2
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)
        frame2 = np.full((64, 64), 2, dtype=np.uint8)
        shared = ("dup", 1)

        agent._add_replay(frame0, 1, 0.1, dedup_key=shared)
        agent._add_replay(frame1, 2, 0.2, dedup_key=shared)
        agent._add_replay(frame2, 3, 0.3, dedup_key=("new", 3))

        self.assertEqual(agent.buf_keys, [("new", 3), shared])
        self.assertEqual(list(agent.buf_has_next), [0, 0])
        self.assertEqual(agent.buf_key_counts, {shared: 1, ("new", 3): 1})
        self.assertEqual(agent.buf_h, {shared, ("new", 3)})

    def test_add_replay_normalizes_out_of_palette_snapshot_once(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[7, 9] = 42

        agent._add_replay(frame, 1, 0.5)

        self.assertEqual(int(agent.buf[0][7, 9]), 0)
        self.assertTrue(agent.buf[0].flags["C_CONTIGUOUS"])
        self.assertEqual(list(agent.buf_has_next), [0])
        self.assertEqual(list(agent.buf_hashes), [agent._fast_frame_hash(agent.buf[0])])

    def test_add_replay_tracks_next_frame_presence_in_packed_buffer(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        next_frame = np.ones((64, 64), dtype=np.uint8)

        agent._add_replay(frame, 1, 0.5, next_frame=next_frame)
        agent._add_replay(frame, 2, 0.1)

        self.assertEqual(list(agent.buf_has_next), [1, 0])

    def test_clear_replay_prunes_hashes_with_retained_entries(self):
        agent = self.make_agent()
        agent.bsz = 2
        frames = [np.full((64, 64), i, dtype=np.uint8) for i in range(5)]
        rewards = [1.0, 5.0, 3.0, 4.0, 2.0]
        agent.buf = list(frames)
        agent.buf_actions = self.mod.array('H', [0, 1, 2, 3, 4])
        agent.buf_rewards = self.mod.array('f', rewards)
        agent.buf_next_frames = [None] * 5
        agent.buf_has_next = self.mod.array('b', [0, 0, 0, 0, 0])
        agent.buf_priorities = self.mod.array('f', [agent._priority_from_reward(r) for r in rewards])
        agent.buf_keys = [(f"k{i}", i) for i in range(5)]
        agent.buf_hashes = self.mod.array('I', [100, 101, 102, 103, 104])
        agent.buf_key_counts = {("stale", 1): 7}

        agent._clear_replay(keep_frac=0.4)

        self.assertEqual(list(agent.buf_hashes), [103, 101])
        self.assertEqual(list(agent.buf_has_next), [0, 0])
        self.assertEqual(agent.buf_key_counts, {("k3", 3): 1, ("k1", 1): 1})

    def test_add_replay_boosts_recent_predecessors_after_buffer_wrap(self):
        agent = self.make_agent()
        agent.buf_max = 4
        frame = np.zeros((64, 64), dtype=np.uint8)
        for _ in range(4):
            agent._add_replay(frame, 1, 0.0)

        agent._add_replay(frame, 2, 2.0)

        rewards = list(agent.buf_rewards)
        self.assertAlmostEqual(rewards[0], 2.0, places=5)
        self.assertGreater(rewards[3], 0.0)
        self.assertGreater(rewards[2], 0.0)
        self.assertGreater(rewards[1], 0.0)
        self.assertAlmostEqual(agent.buf_priorities[3], abs(rewards[3]) + 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[2], abs(rewards[2]) + 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[1], abs(rewards[1]) + 0.01, places=5)

    def test_priority_from_reward_sanitizes_non_finite_values(self):
        agent = self.make_agent()

        self.assertAlmostEqual(agent._priority_from_reward(float("nan")), 0.01, places=5)
        self.assertAlmostEqual(agent._priority_from_reward(float("inf")), 0.01, places=5)
        self.assertAlmostEqual(agent._priority_from_reward(-2.5), 2.51, places=5)

    def test_sampling_probabilities_falls_back_to_uniform_for_invalid_priorities(self):
        agent = self.make_agent()
        agent.buf_priorities = self.mod.array('f', [float("nan"), -3.0, 0.0, float("inf")])

        probs = agent._sampling_probabilities(4)

        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(probs, np.full(4, 0.25, dtype=np.float64)))

    def test_sampling_probabilities_uses_positive_finite_entries(self):
        agent = self.make_agent()
        agent._per_alpha = 1.0
        agent.buf_priorities = self.mod.array('f', [1.0, 2.0, 4.0])

        probs = agent._sampling_probabilities(3)

        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(probs, np.array([1.0, 2.0, 4.0], dtype=np.float64) / 7.0))

    def test_update_sampled_priorities_preserves_last_duplicate_sample(self):
        agent = self.make_agent()
        agent.buf_priorities = self.mod.array('f', [1.0, 1.0, 1.0])

        agent._update_sampled_priorities(
            np.array([1, 2, 1], dtype=np.int64),
            np.array([0.2, 0.4, 0.9], dtype=np.float32),
        )

        self.assertAlmostEqual(agent.buf_priorities[0], 1.0, places=5)
        self.assertAlmostEqual(agent.buf_priorities[1], agent._priority_from_reward(0.9), places=5)
        self.assertAlmostEqual(agent.buf_priorities[2], agent._priority_from_reward(0.4), places=5)

    def test_update_sampled_priorities_sanitizes_non_finite_errors(self):
        agent = self.make_agent()
        agent.buf_priorities = self.mod.array('f', [1.0, 1.0, 1.0])

        agent._update_sampled_priorities(
            np.array([0, 1, 2], dtype=np.int64),
            np.array([np.nan, np.inf, -0.5], dtype=np.float32),
        )

        self.assertAlmostEqual(agent.buf_priorities[0], 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[1], 0.01, places=5)
        self.assertAlmostEqual(agent.buf_priorities[2], 0.51, places=5)

    def test_get_aem_tensors_uses_recent_tail_window(self):
        agent = self.make_agent()
        agent._aem_max_active = 3
        for i in range(5):
            agent._aem_diffs.append(np.full((64, 64), i, dtype=np.float32))
            agent._aem_actions.append(i)
            agent._aem_rewards.append(float(i) + 0.5)

        diffs, acts, rews = agent._get_aem_tensors()

        self.assertEqual(tuple(diffs.shape), (1, 3, 1, 64, 64))
        self.assertEqual(tuple(acts.shape), (1, 3))
        self.assertEqual(tuple(rews.shape), (1, 3))
        self.assertTrue(np.allclose(diffs.cpu().numpy()[0, :, 0, 0, 0], np.array([2.0, 3.0, 4.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(acts.cpu().numpy()[0], np.array([2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.allclose(rews.cpu().numpy()[0], np.array([2.5, 3.5, 4.5], dtype=np.float32)))

    def test_replay_numeric_views_cache_reuses_until_buffer_mutates(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        agent._add_replay(frame, 1, 0.5)
        first = agent._replay_numeric_views(1)
        second = agent._replay_numeric_views(1)

        self.assertIs(first, second)

        del first
        del second
        agent._add_replay(frame, 2, 0.1)
        third = agent._replay_numeric_views(2)

        self.assertEqual(len(third[0]), 2)

    def test_direction_and_error_paths_return_fresh_action_instances(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION2, _GameAction.ACTION4], levels=0)
        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        net = _ForwardOnlyNet(np.array([-5.0, 3.0, -2.0, -3.0, -4.0], dtype=np.float32), agent.device)
        agent.net = net
        agent._bfs = None
        agent._sample = lambda logits, avail=None, temp=1.0: (1, None)

        with mock.patch.object(self.mod.random, "random", return_value=1.0):
            first = agent.choose_action([], frame)
            second = agent.choose_action([], frame)
        self.assertIsNot(first, second)
        self.assertEqual(first.reasoning, "cnn:a2")
        self.assertEqual(second.reasoning, "cnn:a4")

        err_agent = self.make_agent()
        err_frame = _make_frame(0, actions=[_GameAction.ACTION6])
        err_agent._lvl = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        first_err = err_agent.choose_action([], err_frame)
        second_err = err_agent.choose_action([], err_frame)
        self.assertIsNot(first_err, second_err)
        self.assertEqual(first_err.data, {"x": 32, "y": 32})
        self.assertEqual(second_err.data, {"x": 35, "y": 32})

    def test_fresh_action_creates_independent_instances_with_payload(self):
        agent = self.make_agent()

        first = agent._fresh_action(6, {"x": 1, "y": 2})
        second = agent._fresh_action(6, {"x": 1, "y": 2})
        first.reasoning = "changed"
        first.set_data({"x": 9, "y": 9})

        self.assertIsNot(first, second)
        self.assertEqual(second.data, {"x": 1, "y": 2})
        self.assertEqual(second.reasoning, "")

    def test_raw_returns_last_frame_layer(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        last = np.full((64, 64), 7, dtype=np.uint8)
        fd = types.SimpleNamespace(frame=[frame, last])

        raw = agent._raw(fd)

        self.assertTrue(np.array_equal(raw, last))

    def test_bfs_action_key_includes_click_coordinates(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")

        self.assertEqual(solver._action_key(2, None), (2,))
        self.assertEqual(solver._action_key(6, {"x": 5, "y": 9}), (6, 5, 9))

    def test_bfs_priority_bonus_reuses_cached_click_lookup(self):
        agent = self.make_agent()
        calls = []

        class FakeBfs:
            def __init__(self):
                self._action_priority = {(6, 5, 9): 2.0}

            def _action_key(self, act_id, data):
                calls.append((act_id, data))
                return (act_id, None if not data else data.get("x"), None if not data else data.get("y"))

        agent._bfs = FakeBfs()
        payload = {"x": 5, "y": 9}

        first = agent._bfs_priority_bonus(6, payload)
        second = agent._bfs_priority_bonus(6, payload)

        self.assertAlmostEqual(first, 0.5, places=5)
        self.assertAlmostEqual(second, 0.5, places=5)
        self.assertEqual(len(calls), 1)

    def test_bfs_click_priority_bonus_reuses_cached_coord_lookup(self):
        agent = self.make_agent()
        calls = []

        class FakeBfs:
            def __init__(self):
                self._action_priority = {(6, 5, 9): 2.0}

            def _action_key(self, act_id, data):
                calls.append((act_id, data))
                return (act_id, None if not data else data.get("x"), None if not data else data.get("y"))

        agent._bfs = FakeBfs()

        first = agent._bfs_click_priority_bonus((9, 5))
        second = agent._bfs_click_priority_bonus((9, 5))

        self.assertAlmostEqual(first, 0.5, places=5)
        self.assertAlmostEqual(second, 0.5, places=5)
        self.assertEqual(len(calls), 1)

    def test_bfs_ordered_actions_prefers_higher_priority_then_action_id(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver._action_priority = {
            solver._action_key(6, {"x": 1, "y": 1}): 2.0,
            solver._action_key(2, None): 1.0,
        }
        actions = [
            (4, None),
            (6, {"x": 1, "y": 1}),
            (2, None),
            (1, None),
        ]

        ordered = solver._ordered_actions(actions)

        self.assertEqual(ordered[0], (6, {"x": 1, "y": 1}))
        self.assertEqual(ordered[1], (2, None))
        self.assertEqual(ordered[-1], (4, None))

    def test_bfs_state_hash_includes_requested_hidden_fields(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        frame = np.zeros((64, 64), dtype=np.uint8)
        game = _HashGame()

        visible_only = solver._state_hash(game, frame, hidden_fields=None)
        with_hidden = solver._state_hash(game, frame, hidden_fields=["energy", "_private", "missing"])

        self.assertNotEqual(visible_only, with_hidden)
        self.assertEqual(with_hidden[1], (("energy", 7), ("_private", 99)))

    def test_bfs_make_action_caches_plain_inputs_but_not_payload_inputs(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")

        plain_first = solver._make_action(2)
        plain_second = solver._make_action(2)
        payload_first = solver._make_action(6, {"x": 1, "y": 2})
        payload_second = solver._make_action(6, {"x": 1, "y": 2})

        self.assertIs(plain_first, plain_second)
        self.assertIsNot(payload_first, payload_second)
        self.assertEqual(payload_first.data, {"x": 1, "y": 2})
        self.assertEqual(payload_second.data, {"x": 1, "y": 2})

    def test_scan_actions_parallel_matches_serial_order_and_dedup(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver.game_cls = _ParallelScanGame
        solver._parallel_click_chunk = 2
        game = _ParallelScanGame()
        frame = game.frame.copy()
        expected = [
            (1, None),
            (2, None),
            (6, {"x": 1, "y": 1, "game_id": "bfs"}),
            (6, {"x": 4, "y": 4, "game_id": "bfs"}),
        ]

        solver._parallel_workers = 1
        serial_actions = solver._scan_actions(game, frame, 0)
        serial_priority = dict(solver._action_priority)

        solver._parallel_workers = 4
        parallel_actions = solver._scan_actions(game, frame, 0)
        parallel_priority = dict(solver._action_priority)

        self.assertEqual(serial_actions, expected)
        self.assertEqual(parallel_actions, expected)
        self.assertEqual(parallel_priority, serial_priority)
        self.assertEqual(sum(1 for act_id, _ in parallel_actions if act_id == 6), 2)

    def test_scan_actions_parallel_tolerates_worker_exceptions(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver._parallel_workers = 4
        solver._parallel_click_chunk = 2
        game = _ParallelScanGame()
        frame = game.frame.copy()

        with mock.patch.object(
                solver,
                "_click_candidates",
                return_value=[(1, 1), (3, 3), (4, 4)]):
            actions = solver._scan_actions(game, frame, 0)

        self.assertEqual(actions, [
            (1, None),
            (2, None),
            (6, {"x": 1, "y": 1, "game_id": "bfs"}),
            (6, {"x": 4, "y": 4, "game_id": "bfs"}),
        ])

    def test_solve_level_parallel_matches_serial_solution(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver.game_cls = _ParallelSolveGame
        solver._parallel_min_branching = 4
        solver._configure_clone_backend = lambda game, probe_actions: None
        solver._scan_actions = lambda game, f0, bg: [(1, None), (2, None), (3, None), (4, None)]
        solver._configure_snapshot_bfs = lambda game, actions: False

        solver._parallel_workers = 1
        serial_solution = solver._solve_level_impl(0, timeout=8, max_states=64)

        solver.solutions.clear()
        solver._parallel_workers = 4
        parallel_solution = solver._solve_level_impl(0, timeout=8, max_states=64)

        self.assertEqual(serial_solution, [(1, None), (1, None)])
        self.assertEqual(parallel_solution, serial_solution)

    def test_solve_level_parallel_preserves_first_winning_candidate_order(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver.game_cls = _FirstWinningParallelGame
        solver._parallel_workers = 4
        solver._parallel_min_branching = 4
        solver._configure_clone_backend = lambda game, probe_actions: None
        solver._scan_actions = lambda game, f0, bg: [(1, None), (2, None), (3, None), (4, None)]
        solver._configure_snapshot_bfs = lambda game, actions: False

        solution = solver._solve_level_impl(0, timeout=8, max_states=16)

        self.assertEqual(solution, [(1, None)])

    def test_solve_level_parallel_reuses_single_executor_for_one_run(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver.game_cls = _ParallelSolveGame
        solver._parallel_workers = 4
        solver._parallel_min_branching = 4
        solver._configure_clone_backend = lambda game, probe_actions: None
        solver._scan_actions = lambda game, f0, bg: [(1, None), (2, None), (3, None), (4, None)]
        solver._configure_snapshot_bfs = lambda game, actions: False

        created = []
        submitted = []

        class _CountingFuture:
            def __init__(self, fn, args):
                self._fn = fn
                self._args = args

            def result(self):
                return self._fn(*self._args)

        class _CountingExecutor:
            def __init__(self, max_workers):
                created.append(max_workers)

            def submit(self, fn, *args):
                submitted.append(args)
                return _CountingFuture(fn, args)

            def shutdown(self, wait=True):
                return None

        with mock.patch.object(self.mod, "ThreadPoolExecutor", _CountingExecutor):
            solution = solver.solve_level(0, timeout=8, max_states=64)

        self.assertEqual(solution, [(1, None), (1, None)])
        self.assertEqual(created, [4])
        self.assertGreaterEqual(len(submitted), 4)

    def test_hidden_retry_parallel_can_solve_with_hidden_state(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        solver.game_cls = _HiddenRetryParallelGame
        solver._parallel_workers = 4
        solver._parallel_min_branching = 4
        solver.hidden_retry_min_explored = 1
        solver.hidden_retry_unique_ratio = 1.0
        solver.hidden_retry_time_cap = 8.0
        solver._configure_clone_backend = lambda game, probe_actions: None
        solver._scan_actions = lambda game, f0, bg: [(1, None), (2, None), (3, None), (4, None)]
        solver._configure_snapshot_bfs = lambda game, actions: False

        solution = solver._solve_level_impl(0, timeout=8, max_states=32)

        self.assertEqual(solution, [(1, None), (3, None)])
        self.assertEqual(solver.solutions[0], solution)

    def test_implicit_search_graph_reconstructs_click_payload_path(self):
        graph = self.mod._ImplicitSearchGraph(root_state="root")
        child = graph.add_child(0, 6, {"x": 4, "y": 7}, state="child")
        grandchild = graph.add_child(child, 2, None, state="grandchild")

        path = graph.reconstruct(grandchild)

        self.assertEqual(path, [(6, {"x": 4, "y": 7, "game_id": "bfs"}), (2, None)])

    def test_frame_crc_changes_with_frame_contents(self):
        frame_a = np.zeros((64, 64), dtype=np.uint8)
        frame_b = np.zeros((64, 64), dtype=np.uint8)
        frame_b[0, 0] = 1

        crc_a = self.mod._frame_crc(frame_a)
        crc_b = self.mod._frame_crc(frame_b)

        self.assertNotEqual(crc_a, crc_b)

    def test_frame_signature_is_stable_for_identical_frames(self):
        frame = np.full((64, 64), 3, dtype=np.uint8)

        first = self.mod._frame_signature(frame)
        second = self.mod._frame_signature(frame.copy())

        self.assertEqual(first, second)

    def test_replay_batch_tensor_returns_expected_shape(self):
        agent = self.make_agent()
        agent.buf = [
            np.zeros((64, 64), dtype=np.uint8),
            np.ones((64, 64), dtype=np.uint8),
        ]
        agent.buf_hashes = self.mod.array(
            'I',
            [agent._fast_frame_hash(agent.buf[0]), agent._fast_frame_hash(agent.buf[1])],
        )

        states = agent._replay_batch_tensor([0, 1])

        self.assertEqual(tuple(states.shape), (2, 26, 64, 64))

    def test_encode_frame_tensor_normalizes_out_of_palette_values(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[4, 5] = 99

        tensor = agent._encode_frame_tensor(frame)

        self.assertEqual(tuple(tensor.shape), (26, 64, 64))
        self.assertTrue(np.isfinite(tensor.cpu().numpy()).all())
        self.assertEqual(float(tensor[:16, 4, 5].sum().item()), 1.0)
        self.assertEqual(float(tensor[0, 4, 5].item()), 1.0)

    def test_replay_batch_tensor_normalizes_out_of_palette_values(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10, 11] = 42
        agent.buf = [frame]
        agent.buf_hashes = self.mod.array('I', [agent._fast_frame_hash(frame)])

        states = agent._replay_batch_tensor([0])

        self.assertEqual(tuple(states.shape), (1, 26, 64, 64))
        self.assertTrue(np.isfinite(states.cpu().numpy()).all())
        self.assertEqual(float(states[0, :16, 10, 11].sum().item()), 1.0)
        self.assertEqual(float(states[0, 0, 10, 11].item()), 1.0)

    def test_replay_batch_tensor_uses_cached_hashes_when_available(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        encoded = agent._encode_frame_tensor(frame)
        frame_hash = agent._fast_frame_hash(frame)
        agent.buf = [frame]
        agent.buf_hashes = self.mod.array('I', [frame_hash])
        agent._frame_feature_cache[frame_hash] = (
            encoded[:16].unsqueeze(0),
            encoded[16:17].unsqueeze(0),
            encoded[17:18].unsqueeze(0),
            encoded[18:19].unsqueeze(0),
        )

        with mock.patch.object(agent, "_fast_frame_hash", side_effect=AssertionError("should use cached replay hash")):
            states = agent._replay_batch_tensor([0])

        self.assertEqual(tuple(states.shape), (1, 26, 64, 64))

    def test_replay_batch_tensor_reuses_positional_and_zero_tail_caches(self):
        agent = self.make_agent()
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)
        agent.buf = [frame0, frame1]
        agent.buf_hashes = self.mod.array('I', [agent._fast_frame_hash(frame0), agent._fast_frame_hash(frame1)])

        first = agent._replay_batch_tensor([0, 1])
        pos_key = (first.device.type, first.device.index, 2)
        zero_key = (first.device.type, first.device.index, 2, str(first.dtype))
        tail_key = (first.device.type, first.device.index, 2, str(first.dtype))
        first_pos = agent._replay_pos_cache[pos_key]
        first_zero = agent._replay_zero_tail_cache[zero_key]
        first_tail = agent._replay_tail_cache[tail_key]

        second = agent._replay_batch_tensor([1, 0])
        second_pos = agent._replay_pos_cache[pos_key]
        second_zero = agent._replay_zero_tail_cache[zero_key]
        second_tail = agent._replay_tail_cache[tail_key]

        self.assertEqual(tuple(second.shape), (2, 26, 64, 64))
        self.assertIs(first_pos, second_pos)
        self.assertIs(first_zero, second_zero)
        self.assertIs(first_tail, second_tail)

    def test_replay_batch_tensor_stores_packed_feature_cache_entries(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent.buf = [frame]
        agent.buf_hashes = self.mod.array('I', [frame_hash])

        states = agent._replay_batch_tensor([0])
        cached = agent._frame_feature_cache[frame_hash]

        self.assertEqual(tuple(states.shape), (1, 26, 64, 64))
        self.assertTrue(torch.is_tensor(cached))
        self.assertEqual(tuple(cached.shape), (1, 19, 64, 64))

    def test_replay_batch_tensor_reuses_packed_features_for_duplicate_hashes_in_batch(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame_hash = agent._fast_frame_hash(frame)
        agent.buf = [frame, frame.copy()]
        agent.buf_hashes = self.mod.array('I', [frame_hash, frame_hash])

        pack_calls = {"count": 0}
        original_pack = agent._pack_replay_feature_channels

        def counting_pack(*args, **kwargs):
            pack_calls["count"] += 1
            return original_pack(*args, **kwargs)

        agent._pack_replay_feature_channels = counting_pack
        try:
            states = agent._replay_batch_tensor([0, 1])
        finally:
            agent._pack_replay_feature_channels = original_pack

        self.assertEqual(tuple(states.shape), (2, 26, 64, 64))
        self.assertEqual(pack_calls["count"], 1)

    def test_replay_batch_tensor_upgrades_legacy_cache_entries_to_packed_tensors(self):
        import torch

        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        encoded = agent._encode_frame_tensor(frame)
        frame_hash = agent._fast_frame_hash(frame)
        agent.buf = [frame]
        agent.buf_hashes = self.mod.array('I', [frame_hash])
        agent._frame_feature_cache[frame_hash] = (
            encoded[:16].unsqueeze(0),
            encoded[16:17].unsqueeze(0),
            encoded[17:18].unsqueeze(0),
            encoded[18:19].unsqueeze(0),
        )

        states = agent._replay_batch_tensor([0])
        upgraded = agent._frame_feature_cache[frame_hash]

        self.assertEqual(tuple(states.shape), (1, 26, 64, 64))
        self.assertTrue(torch.is_tensor(upgraded))
        self.assertEqual(tuple(upgraded.shape), (1, 19, 64, 64))

    def test_sample_uses_supplied_avail_ids_without_recomputing(self):
        import torch

        agent = self.make_agent()
        logits = torch.tensor([0.1, 3.0, 0.2, 0.0, -1.0], dtype=torch.float32, device=agent.device)

        with mock.patch.object(agent, "_available_action_ids", side_effect=AssertionError("should use provided avail_ids")):
            action_idx, coords = agent._sample(
                logits,
                avail=[_GameAction.ACTION2],
                avail_ids=[2],
                temp=1.0,
            )

        self.assertEqual(action_idx, 1)
        self.assertIsNone(coords)

    def test_train_handles_next_frame_sampling_with_ndarray_indices(self):
        import torch

        agent = self.make_agent()
        agent.bsz = 2
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)
        next_frame = np.full((64, 64), 2, dtype=np.uint8)
        agent.buf = [frame0, frame1]
        agent.buf_actions = self.mod.array('H', [0, 1])
        agent.buf_rewards = self.mod.array('f', [1.0, 0.5])
        agent.buf_next_frames = [next_frame, None]
        agent.buf_has_next = self.mod.array('b', [1, 0])
        agent.buf_priorities = self.mod.array('f', [1.0, 1.0])
        agent.buf_hashes = self.mod.array('I', [agent._fast_frame_hash(frame0), agent._fast_frame_hash(frame1)])

        class TinyTrainNet(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0.2, 0.1, -0.1, -0.2, -0.3], dtype=torch.float32, device=device))

            def forward(self, x, *args, **kwargs):
                return self.logits.unsqueeze(0).expand(x.size(0), -1)

            def forward_actions(self, x, *args, **kwargs):
                return self.forward(x)

        agent.net = TinyTrainNet(agent.device)
        agent._target_net = TinyTrainNet(agent.device)
        agent._target_net.load_state_dict(agent.net.state_dict())
        agent.opt = torch.optim.SGD(agent.net.parameters(), lr=0.01)
        agent.scheduler = None
        agent._replay_batch_tensor = lambda indices: torch.zeros((len(indices), 26, 64, 64), dtype=torch.float32, device=agent.device)

        with mock.patch.object(self.mod.np.random, "choice", return_value=np.array([0, 1], dtype=np.int64)):
            trained = agent._train()

        self.assertTrue(trained)
        self.assertEqual(agent._model_revision, 1)
        self.assertTrue(all(np.isfinite(priority) and priority > 0.0 for priority in agent.buf_priorities))

    def test_frame_view_casts_dtype_without_changing_values(self):
        frame = np.arange(16, dtype=np.uint8).reshape(4, 4)

        viewed = self.mod._frame_view(frame, np.float32)

        self.assertEqual(viewed.dtype, np.float32)
        self.assertTrue(np.allclose(viewed, frame.astype(np.float32)))

    def test_bfs_last_frame_returns_none_when_result_has_no_frames(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")

        self.assertIsNone(solver._last_frame(None))
        self.assertIsNone(solver._last_frame(types.SimpleNamespace(frame=[])))

    def test_bfs_is_complete_checks_result_and_game_level_index(self):
        solver = self.mod.BFSSolver("dummy.py", "DummyGame")
        level_idx = 3

        incomplete = solver._is_complete(types.SimpleNamespace(_current_level_index=3), types.SimpleNamespace(levels_completed=3), level_idx)
        result_complete = solver._is_complete(types.SimpleNamespace(_current_level_index=3), types.SimpleNamespace(levels_completed=4), level_idx)
        game_complete = solver._is_complete(types.SimpleNamespace(_current_level_index=4), types.SimpleNamespace(levels_completed=3), level_idx)

        self.assertFalse(incomplete)
        self.assertTrue(result_complete)
        self.assertTrue(game_complete)

    def test_warm_fallthrough_bc_filters_out_click_actions(self):
        agent = self.make_agent()
        frame = _make_frame(0, actions=[_GameAction.ACTION1, _GameAction.ACTION6], levels=0)
        captured = {}

        class FakeBfs:
            def __init__(self):
                self.game_cls = _ReplayGame
                self.solutions = {}
                self._last_effective_actions = [
                    (1, None),
                    (6, {"x": 4, "y": 7}),
                    (2, None),
                    (3, None),
                    (4, None),
                ]
                self._action_priority = {}

            def _clone_game(self, game):
                cloned = _ReplayGame()
                cloned.step = game.step
                cloned._current_level_index = game._current_level_index
                return cloned

            def _action_key(self, act_id, data):
                if not data:
                    return (int(act_id),)
                return (int(act_id), int(data["x"]), int(data["y"]))

            def solve_level(self, level_idx, prev_solution=None, timeout=None, net=None, frame_tensor=None):
                return None

        agent._bfs = FakeBfs()
        agent._bfs_tried = True
        agent._try_bfs_solve = lambda level_idx, lf=None: None
        agent.net = _DummyNet()
        agent.opt = object()
        agent.scheduler = object()
        agent._target_net = None
        agent._train = lambda: False

        def fake_bc(self_ref, raw_frames, action_indices, batch_size, epochs):
            captured["actions"] = list(action_indices)
            return 0.0

        original_bc = self.mod.MyAgent._bc_train_on_solution
        self.mod.MyAgent._bc_train_on_solution = fake_bc
        try:
            result = agent.choose_action([], frame)
        finally:
            self.mod.MyAgent._bc_train_on_solution = original_bc

        self.assertTrue(captured["actions"])
        self.assertTrue(all(action_idx < 5 for action_idx in captured["actions"]))
        self.assertEqual(result.value, 1)

    def test_rescoring_path_can_choose_click_without_cloning_frame(self):
        import torch

        agent = self.make_agent()
        target_y, target_x = 7, 9
        click_index = 5 + target_y * agent.G + target_x
        logits = torch.full((4101,), -10.0, dtype=torch.float32)
        logits[0] = 1.0
        logits[click_index] = 0.9

        class FakeBfs:
            def __init__(self):
                self._action_priority = {(6, target_x, target_y): 1.0}

            def _action_key(self, act_id, data):
                if not data:
                    return (int(act_id),)
                return (int(act_id), int(data["x"]), int(data["y"]))

        agent.cl = 0
        agent._wd = True
        agent._eps = 0.0
        agent.net = _FixedLogitNet(logits, agent.device)
        agent._bfs = FakeBfs()
        agent._wm = torch.ones((64, 64), dtype=torch.float32)
        agent._wm[target_y, target_x] = 10.0
        frame = _make_frame(0, actions=[_GameAction.ACTION1, _GameAction.ACTION6], levels=0)

        with mock.patch.object(self.mod.random, "random", return_value=1.0):
            result = agent.choose_action([], frame)

        self.assertEqual(result.value, 6)
        self.assertEqual(result.data, {"x": target_x, "y": target_y})
        self.assertEqual(result.reasoning, f"cnn:c({target_x},{target_y})")

    def test_bc_train_on_solution_aborts_on_non_finite_loss(self):
        import torch

        agent = self.make_agent()
        frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(2)]
        agent.net = _ForwardOnlyNet(np.zeros(5, dtype=np.float32), agent.device)
        agent.net.train = lambda: agent.net
        agent.opt = mock.Mock()
        agent._grad_scaler = None
        before_revision = agent._model_revision

        with mock.patch.object(self.mod.F, "cross_entropy", return_value=torch.tensor(float("nan"), device=agent.device)):
            loss = agent._bc_train_on_solution(frames, [0, 1], batch_size=2, epochs=1)

        self.assertIsNone(loss)
        agent.opt.step.assert_not_called()
        self.assertEqual(agent._model_revision, before_revision)


if __name__ == "__main__":
    unittest.main()
