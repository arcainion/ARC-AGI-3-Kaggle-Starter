"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it â€” exactly what the Kaggle
gateway does, just in-process.

Usage:
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200
    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

_PATCHED_AGENT_MAIN = None
_PATCHED_DO_ACTION_REQUEST = None
_PATCHED_APPEND_FRAME = None


def format_level_progress(levels_completed: int, win_levels: int | None) -> str:
    """Render human-readable level progress from completed/total counts."""
    completed = max(int(levels_completed or 0), 0)
    try:
        total = int(win_levels) if win_levels is not None else 0
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        return f"completed={completed}"
    completed = min(completed, total)
    current = total if completed >= total else min(completed + 1, total)
    return f"level={current}/{total} completed={completed}/{total}"


def _action_reasoning_payload(action) -> dict | None:
    """Normalize action reasoning to the dict shape expected by the ARC wrapper."""
    reasoning = getattr(action, "reasoning", None)
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        return reasoning
    return {"text": str(reasoning)}


def _action_input_payload(agent, action) -> dict:
    """Build a playback-compatible action payload for recorder events."""
    action_id = action.value if hasattr(action, "value") else int(action)
    data = action.action_data.model_dump()
    payload = {
        "id": int(action_id),
        "data": dict(data),
        "game_id": getattr(agent, "game_id", None),
    }
    reasoning = _action_reasoning_payload(action)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return payload


def configure_logging() -> None:
    """Set a simple console logger without clobbering unrelated handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


def install_agent_logging_patches() -> None:
    """Patch vendor Agent methods at runtime without modifying agent.py on disk."""
    global _PATCHED_AGENT_MAIN, _PATCHED_DO_ACTION_REQUEST, _PATCHED_APPEND_FRAME
    from agents.agent import Agent
    from agents.tracing import trace_agent_session

    if getattr(Agent, "_play_local_logging_patched", False):
        return

    _PATCHED_AGENT_MAIN = Agent.main
    _PATCHED_DO_ACTION_REQUEST = Agent.do_action_request
    _PATCHED_APPEND_FRAME = Agent.append_frame

    @trace_agent_session
    def patched_main(self) -> None:
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(
                self.frames,
                self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                ),
            )
            if frame := self.take_action(action):
                self.append_frame(frame)
                display_count = int(self.action_counter) + 1
                progress = format_level_progress(
                    getattr(frame, "levels_completed", 0),
                    getattr(frame, "win_levels", None),
                )
                elapsed_seconds = max((time.time() - self.timer) * 100 // 1 / 100, 0.1)
                fps = round(display_count / elapsed_seconds, 2)
                logging.getLogger().info(
                    f"{self.game_id} - {action.name}: count {display_count}, {progress}, avg fps {fps})"
                )
            self.action_counter += 1
        self.cleanup()

    def patched_do_action_request(self, action):
        self._play_local_last_action_input = _action_input_payload(self, action)
        return _PATCHED_DO_ACTION_REQUEST(self, action)

    def patched_append_frame(self, frame) -> None:
        _PATCHED_APPEND_FRAME(self, frame)
        if hasattr(self, "recorder") and not self.is_playback:
            action_input = getattr(self, "_play_local_last_action_input", None)
            if action_input is not None:
                self.recorder.record({"action_input": action_input})
                self._play_local_last_action_input = None

    patched_main._play_local_patched = True
    Agent.main = patched_main
    Agent.do_action_request = patched_do_action_request
    Agent.append_frame = patched_append_frame
    Agent._play_local_logging_patched = True


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step.")
    args = p.parse_args()

    configure_logging()
    install_agent_logging_patches()

    # NORMAL = local execution; game source is downloaded on first call into
    # ./environment_files/ and cached for subsequent runs.
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Resolve which games to play. `arc.make()` accepts the short game id
    # (e.g. "ls20") even though the EnvironmentInfo.game_id includes the
    # version suffix ("ls20-9607627b"), so we normalize to short ids.
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; playing all {len(game_ids)} games "
              f"(this is what Kaggle does in competition rerun).\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    per_game = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
        env = arc.make(game_id, render_mode=args.render)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue

        agent = MyAgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev"],
        )
        agent.main()

        final = agent.frames[-1]
        progress = format_level_progress(final.levels_completed, getattr(final, "win_levels", None))
        per_game.append((game_id, final.state, progress, agent.action_counter))
        print(f"  -> state={final.state}, {progress}, actions={agent.action_counter}")

    sc = arc.get_scorecard()
    print("\n========= SUMMARY =========")
    for gid, state, progress, actions in per_game:
        print(f"  {gid:8} {progress:28}  actions={actions:5}  state={state}")
    score_val = sc.score if hasattr(sc, "score") else sc
    print(f"\nAggregate scorecard score: {score_val}")


if __name__ == "__main__":
    main()

