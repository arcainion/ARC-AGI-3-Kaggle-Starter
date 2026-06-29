# AGENTS.md

## Scope

This repository is an ARC-AGI-3 local development starter. The main editable agent implementation is:

- [agent/my_agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\agent\my_agent.py)

Key local runner and packaging scripts:

- [scripts/play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\play_local.py)
- [scripts/build_notebook.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\build_notebook.py)
- [scripts/slim_framework.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\slim_framework.py)

Vendor framework checkout:

- [vendor/ARC-AGI-3-Agents](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\vendor\ARC-AGI-3-Agents)

## Hard Constraints

- Do not edit [vendor/ARC-AGI-3-Agents/agents/agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\vendor\ARC-AGI-3-Agents\agents\agent.py). This was explicitly called out as off-limits.
- If vendor action-log behavior needs to change for local development, implement it in [scripts/play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\play_local.py) via runtime patching, not by modifying vendor source.
- Preserve unrelated user changes. The worktree may be dirty.

## Current Local Action-Log Design

The vendor `Agent` base class still has two upstream issues:

- It logs per-action `count` before incrementing `action_counter`, so the displayed count is off by one.
- It records frames to the recorder, but base recordings do not include `action_input`, while `Playback` expects action events.

Because `agent.py` cannot be changed, [scripts/play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\play_local.py) patches the vendor `Agent` class at runtime:

- `install_agent_logging_patches()` patches `Agent.main`
  - emits corrected displayed action count
  - computes displayed FPS from the corrected count
  - formats level progress as `level=X/Y completed=A/Y` using `levels_completed` and `win_levels`
- patches `Agent.do_action_request`
  - captures the outgoing `action_input` payload
- patches `Agent.append_frame`
  - mirrors an `{"action_input": ...}` event into the recorder so local recordings become playback-compatible

Supporting helpers in `play_local.py`:

- `format_level_progress()`
- `_action_reasoning_payload()`
- `_action_input_payload()`
- `configure_logging()`

## Logging Notes

Local runner logging intentionally avoids `logging.basicConfig(..., force=True)`.

- `configure_logging()` only adds a basic stream handler if none exists.
- This avoids clobbering unrelated handlers from `arc_agi`, the vendor framework, or future diagnostics.

Current local action lines should look like:

```text
sc25 - ACTION6: count 13, level=1/6 completed=0/6, avg fps 0.12)
```

BFS logs from [agent/my_agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\agent\my_agent.py) are intentionally left unchanged, for example:

```text
BFS L0: UNLOCKED with ACTION1! 13 actions
BFS L0: 13 effective actions
```

## Recent Runtime Fixes In `my_agent.py`

These fixes are already implemented and should not be regressed:

- action counter double-counting fix during framework-driven `main()` loops
- ASCII-safe output in `play_local.py` and slimmed vendor package init
- replay feature-cache miss hardening in `_replay_batch_tensor()`
- reset/control-action id fix to use real engine reset semantics instead of stale literal `8`
- stricter unit-test `GameAction` stub contract in [tests/test_my_agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\tests\test_my_agent.py)
- prioritized replay probability sanitization to prevent `ValueError: probabilities contain NaN`
- palette normalization for out-of-range frame values to prevent CUDA device-side asserts during frame encoding

## Tests

Primary local agent suite:

- `.venv\Scripts\python.exe -m unittest tests.test_my_agent`

Local runner / action-log patch suite:

- `.venv\Scripts\python.exe -m unittest tests.test_play_local`

Current dedicated runner tests cover:

- progress formatter behavior
- non-destructive logging configuration
- runtime patch behavior for corrected action logging and mirrored `action_input` recorder events

## Known Remaining Limitations

- The vendor source still contains the original action-log issues. Only the local `play_local.py` entrypoint fixes them.
- A different entrypoint that imports the vendor `Agent` directly without calling `install_agent_logging_patches()` will still see upstream logging behavior.
- `play_local.py` live runs may require network access on first environment fetch or when cached environments are missing.

## Recommended Workflow

1. Edit [agent/my_agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\agent\my_agent.py) for agent behavior changes.
2. Edit [scripts/play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\play_local.py) for local runner, logging, or recorder/playback compatibility fixes.
3. Run:
   - `.venv\Scripts\python.exe -m unittest tests.test_my_agent`
   - `.venv\Scripts\python.exe -m unittest tests.test_play_local`
4. If needed, run a live local check:
   - `.venv\Scripts\python.exe scripts\play_local.py --game sc25`

## Files Most Relevant To Future Action-Log Work

- [scripts/play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\scripts\play_local.py)
- [tests/test_play_local.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\tests\test_play_local.py)
- [vendor/ARC-AGI-3-Agents/agents/agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\vendor\ARC-AGI-3-Agents\agents\agent.py)
- [vendor/ARC-AGI-3-Agents/agents/recorder.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\vendor\ARC-AGI-3-Agents\agents\recorder.py)
- [tests/test_my_agent.py](/abs/path/H:\ARC-AGI-3-Kaggle-Starter\tests\test_my_agent.py)
