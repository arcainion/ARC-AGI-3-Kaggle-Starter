from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from arcengine import ActionInput, GameAction, GameState

from scripts.play_local import configure_logging, format_level_progress, load_my_agent_class


SIMPLE_GAMES = {
    "line4": ROOT / "simple_games" / "line4" / "line4.py",
    "clickopen": ROOT / "simple_games" / "clickopen" / "clickopen.py",
    "hiddenkey": ROOT / "simple_games" / "hiddenkey" / "hiddenkey.py",
}
SIMPLE_PLAYING_STATE = object()


def load_game_class(game_id: str):
    game_path = SIMPLE_GAMES[game_id]
    spec = importlib.util.spec_from_file_location(f"simple_game_{game_id}", game_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load simple game at {game_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls_name = game_id.capitalize() if len(game_id) != 4 or not game_id[0].isalpha() else (game_id[0].upper() + game_id[1:])
    return getattr(module, cls_name), game_path.parent


def frame_state(levels_completed: int, win_levels: int):
    if int(levels_completed) >= int(win_levels):
        return GameState.WIN
    return SIMPLE_PLAYING_STATE


def make_frame(game, result):
    available = getattr(game, "_available_actions", [])
    levels_completed = int(getattr(result, "levels_completed", getattr(game, "_current_level_index", 0)))
    return SimpleNamespace(
        frame=list(getattr(result, "frame", [])),
        levels_completed=levels_completed,
        score=levels_completed,
        available_actions=[GameAction.from_id(int(action_id)) for action_id in available],
        state=frame_state(levels_completed, getattr(game, "win_levels", 1)),
        win_levels=int(getattr(game, "win_levels", 1)),
    )


def action_payload(action):
    if hasattr(action, "action_data"):
        return dict(action.action_data.model_dump())
    return {}


def action_name(action_id: int):
    return GameAction.from_id(int(action_id)).name


def main():
    parser = argparse.ArgumentParser(description="Run MyAgent against tiny local toy games.")
    parser.add_argument("--game", default=None, help="Toy game id to run. Omit to run all.")
    parser.add_argument("--max-steps", type=int, default=60, help="Per-game action cap.")
    parser.add_argument("--list", action="store_true", help="List available toy games.")
    parser.add_argument("--fast", action="store_true", help="Skip BFS and training for cheap policy-only checks.")
    parser.add_argument("--disable-bfs", action="store_true", help="Disable BFS source loading and BFS solve attempts.")
    parser.add_argument("--disable-training", action="store_true", help="Disable replay/BC/DQN training while running toy games.")
    args = parser.parse_args()

    configure_logging()
    if args.list:
        for game_id in SIMPLE_GAMES:
            print(game_id)
        return

    game_ids = [args.game] if args.game else list(SIMPLE_GAMES)
    missing = [game_id for game_id in game_ids if game_id not in SIMPLE_GAMES]
    if missing:
        raise SystemExit(f"Unknown toy game id(s): {missing}. Use --list.")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        current_max = getattr(MyAgentCls, "MAX_ACTIONS")
        if current_max in (None, float("inf")):
            MyAgentCls.MAX_ACTIONS = int(args.max_steps)
        else:
            MyAgentCls.MAX_ACTIONS = min(int(current_max), int(args.max_steps))

    results = []
    for game_id in game_ids:
        GameCls, local_dir = load_game_class(game_id)
        game = GameCls()
        game.set_level(0)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        initial = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        fake_env = SimpleNamespace(environment_info=SimpleNamespace(local_dir=str(local_dir)))
        agent = MyAgentCls(
            card_id="simple-local",
            game_id=game_id,
            agent_name=f"MyAgent.simple.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=fake_env,
            tags=["simple-local"],
        )
        if args.fast or args.disable_bfs:
            agent._bfs_tried = True
            agent._bfs = None
            agent._try_bfs_solve = lambda *run_args, **run_kwargs: None
        if args.fast or args.disable_training:
            agent._train = lambda *run_args, **run_kwargs: False
            agent._maybe_train = lambda *run_args, **run_kwargs: 0

        frames = []
        lf = make_frame(game, initial)
        frames.append(lf)
        agent.start_time = time.time()
        agent.total_time_budget = 3600.0
        agent.estimated_total_levels = max(1, int(getattr(game, "win_levels", 1)))

        while agent.action_counter <= agent.MAX_ACTIONS and lf.state is not GameState.WIN:
            action = agent.choose_action(frames, lf)
            act_id = int(action.value) if hasattr(action, "value") else int(action)
            result = game.perform_action(
                ActionInput(id=GameAction.from_id(act_id), data=action_payload(action)),
                raw=True,
            )
            lf = make_frame(game, result)
            frames.append(lf)
            display_count = int(agent.action_counter)
            elapsed_seconds = max(time.time() - agent.start_time, 0.1)
            fps = round(display_count / elapsed_seconds, 2)
            progress = format_level_progress(lf.levels_completed, lf.win_levels)
            logging.getLogger().info(
                f"{game_id} - {action_name(act_id)}: count {display_count}, {progress}, avg fps {fps})"
            )

        print(
            f"{game_id}: state={lf.state}, {format_level_progress(lf.levels_completed, lf.win_levels)}, "
            f"actions={agent.action_counter}"
        )
        results.append((game_id, lf.state, lf.levels_completed, agent.action_counter))

    print("\n========= SIMPLE SUMMARY =========")
    for game_id, state, levels_completed, actions in results:
        print(
            f"  {game_id:10} state={state} completed={levels_completed} "
            f"actions={actions}"
        )


if __name__ == "__main__":
    main()
