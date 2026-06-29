from __future__ import annotations

import importlib.util
import io
import logging
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAY_LOCAL_PATH = ROOT / "scripts" / "play_local.py"


def _load_play_local(module_name: str):
    arc_agi_mod = types.ModuleType("arc_agi")
    arc_agi_mod.OperationMode = types.SimpleNamespace(NORMAL="NORMAL")

    class _Arcade:
        def __init__(self, operation_mode=None):
            self.operation_mode = operation_mode

    arc_agi_mod.Arcade = _Arcade
    sys.modules["arc_agi"] = arc_agi_mod

    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, PLAY_LOCAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_agents():
    agents_pkg = types.ModuleType("agents")
    agent_mod = types.ModuleType("agents.agent")
    tracing_mod = types.ModuleType("agents.tracing")

    class FakeRecorder:
        def __init__(self):
            self.events = []

        def record(self, data):
            self.events.append(data)

    class FakeAgent:
        MAX_ACTIONS = 5

        def __init__(self):
            self.game_id = "sc25"
            self.frames = [types.SimpleNamespace(levels_completed=0, win_levels=6, state="INIT")]
            self.arc_env = types.SimpleNamespace(observation_space=None)
            self.action_counter = 0
            self.recorder = FakeRecorder()
            self.cleaned = False
            self.next_action = None

        @property
        def is_playback(self):
            return False

        def _convert_raw_frame_data(self, raw):
            return raw

        def choose_action(self, frames, latest_frame):
            return self.next_action

        def is_done(self, frames, latest_frame):
            return self.action_counter >= 1

        def do_action_request(self, action):
            return types.SimpleNamespace(levels_completed=0, win_levels=6, state="RUN")

        def take_action(self, action):
            return self.do_action_request(action)

        def append_frame(self, frame):
            self.frames.append(frame)
            self.recorder.record({"frame_levels_completed": frame.levels_completed})

        def cleanup(self):
            self.cleaned = True

        def main(self):
            raise AssertionError("install_agent_logging_patches should replace Agent.main")

    def trace_agent_session(func):
        return func

    agent_mod.Agent = FakeAgent
    tracing_mod.trace_agent_session = trace_agent_session
    agents_pkg.agent = agent_mod
    agents_pkg.tracing = tracing_mod
    sys.modules["agents"] = agents_pkg
    sys.modules["agents.agent"] = agent_mod
    sys.modules["agents.tracing"] = tracing_mod
    return FakeAgent


class PlayLocalTests(unittest.TestCase):
    def test_format_level_progress_uses_completed_and_total(self):
        mod = _load_play_local("test_play_local_format")

        self.assertEqual(mod.format_level_progress(0, 6), "level=1/6 completed=0/6")
        self.assertEqual(mod.format_level_progress(5, 6), "level=6/6 completed=5/6")
        self.assertEqual(mod.format_level_progress(6, 6), "level=6/6 completed=6/6")
        self.assertEqual(mod.format_level_progress(2, None), "completed=2")

    def test_configure_logging_preserves_existing_handlers(self):
        mod = _load_play_local("test_play_local_logging")
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        existing = logging.StreamHandler(io.StringIO())
        root.handlers = [existing]
        root.setLevel(logging.WARNING)
        try:
            mod.configure_logging()
            self.assertEqual(root.handlers, [existing])
            self.assertEqual(root.level, logging.INFO)
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)

    def test_runtime_patches_fix_action_log_and_record_action_input(self):
        mod = _load_play_local("test_play_local_patch")
        FakeAgent = _install_fake_agents()
        mod.install_agent_logging_patches()

        class _ActionData:
            def model_dump(self):
                return {"x": 11, "y": 13}

        action = types.SimpleNamespace(
            name="ACTION6",
            value=6,
            action_data=_ActionData(),
            reasoning="click target",
        )
        agent = FakeAgent()
        agent.next_action = action

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        try:
            agent.main()
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)

        log_output = stream.getvalue()
        self.assertIn("sc25 - ACTION6: count 1, level=1/6 completed=0/6", log_output)
        self.assertTrue(agent.cleaned)
        self.assertEqual(agent.recorder.events[0], {"frame_levels_completed": 0})
        self.assertEqual(
            agent.recorder.events[1],
            {
                "action_input": {
                    "id": 6,
                    "data": {"x": 11, "y": 13},
                    "game_id": "sc25",
                    "reasoning": {"text": "click target"},
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
