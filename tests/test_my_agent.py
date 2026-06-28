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
    RESET = _Action(8)

    @staticmethod
    def from_id(value: int):
        return _Action(value)


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

        self.assertEqual(result.value, 8)
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

        self.assertEqual(result.value, 8)
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
        self.assertEqual(reset_result.value, 8)
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

        self.assertEqual(result.value, 8)
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

    def test_semantic_target_choice_breaks_equal_scores_toward_continuity(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)

        def fake_detector(grid):
            return {
                "components_per_value": {
                    "4": [{"center": (20.0, 20.0), "cell_count": 4}],
                    "14": [
                        {"center": (25.0, 20.0), "cell_count": 4},
                        {"center": (20.0, 28.0), "cell_count": 4},
                    ],
                }
            }

        agent._semantic_detector = fake_detector
        agent._semantic_target_coord = (20, 29)

        choice = agent._semantic_target_choice(frame)

        self.assertEqual((round(choice["target_y"]), round(choice["target_x"])), (20, 28))
        self.assertGreater(choice["continuity_bonus"], 0.0)

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

    def test_blocked_click_coord_returns_last_click_when_state_is_unchanged(self):
        agent = self.make_agent()
        frame = np.zeros((64, 64), dtype=np.uint8)
        agent.pai = 5 + 18 * agent.G + 11
        agent.pr = frame.copy()

        blocked = agent._blocked_click_coord(frame)

        self.assertEqual(blocked, (18, 11))

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

        self.assertEqual(result.value, 4)
        self.assertEqual(result.reasoning, "cnn:a4")

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
        agent.buf_priorities = [0.11, 2.01, 1.51, 0.21, 3.01]
        agent.buf_keys = [("k0", 0), ("k1", 1), None, ("k3", 3), ("k4", 4)]
        agent.buf_h = {("old", 1)}
        agent.buf_pos = 4

        agent._clear_replay(keep_frac=0.4)

        self.assertEqual(len(agent.buf), 2)
        self.assertEqual(sorted(float(x) for x in agent.buf_rewards), [2.0, 3.0])
        self.assertEqual(list(agent.buf_actions), [1, 4])
        self.assertEqual(agent.buf_keys, [("k1", 1), ("k4", 4)])
        self.assertEqual(agent.buf_h, {("k1", 1), ("k4", 4)})
        self.assertEqual(agent.buf_pos, 0)

    def test_clear_replay_full_clear_resets_all_buffers(self):
        agent = self.make_agent()
        agent.buf = [np.ones((64, 64), dtype=np.uint8)]
        agent.buf_actions = self.mod.array('H', [3])
        agent.buf_rewards = self.mod.array('f', [1.25])
        agent.buf_next_frames = [np.zeros((64, 64), dtype=np.uint8)]
        agent.buf_priorities = [1.26]
        agent.buf_keys = [("seen", 3)]
        agent.buf_h = {("seen", 3)}
        agent.buf_pos = 7

        agent._clear_replay(keep_frac=0.0)

        self.assertEqual(agent.buf, [])
        self.assertEqual(list(agent.buf_actions), [])
        self.assertEqual(list(agent.buf_rewards), [])
        self.assertEqual(agent.buf_next_frames, [])
        self.assertEqual(agent.buf_priorities, [])
        self.assertEqual(agent.buf_keys, [])
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
        agent.buf_priorities = [0.11, 0.21, 0.31]
        agent.buf_keys = [("k0", 0), None, ("k2", 2)]
        agent.buf_h = {("keep", 1)}
        agent.buf_pos = 2

        agent._clear_replay(keep_frac=0.5)

        self.assertEqual(len(agent.buf), 3)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(agent.buf, frames)))
        self.assertEqual(list(agent.buf_actions), [0, 1, 2])
        self.assertTrue(np.allclose(list(agent.buf_rewards), [0.1, 0.2, 0.3]))
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
        agent.buf_priorities = [0.11, 0.21, 0.31]
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
        self.assertEqual(agent.buf_keys, [("f2", 65535), ("f1", 2)])
        self.assertEqual(agent.buf_h, {("f2", 65535), ("f1", 2)})

    def test_add_replay_overwrite_removes_evicted_dedup_key(self):
        agent = self.make_agent()
        agent.buf_max = 1
        frame0 = np.zeros((64, 64), dtype=np.uint8)
        frame1 = np.ones((64, 64), dtype=np.uint8)

        agent._add_replay(frame0, 1, 0.1, dedup_key=("f0", 1))
        agent._add_replay(frame1, 2, 0.2, dedup_key=("f1", 2))

        self.assertEqual(agent.buf_keys, [("f1", 2)])
        self.assertEqual(agent.buf_h, {("f1", 2)})

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

        states = agent._replay_batch_tensor([0, 1])

        self.assertEqual(tuple(states.shape), (2, 26, 64, 64))

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


if __name__ == "__main__":
    unittest.main()
