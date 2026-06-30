from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _action_id(action_input) -> int:
    action_id = getattr(action_input, "id", action_input)
    return int(action_id.value) if hasattr(action_id, "value") else int(action_id)


class Hiddenkey:
    """Tiny hidden-state game: charge energy invisibly, then unlock the level."""

    def __init__(self):
        self.win_levels = 1
        self._available_actions = [1, 3, 5]
        self._current_level_index = 0
        self.set_level(0)

    def set_level(self, level_idx: int):
        self._current_level_index = int(level_idx)
        self.energy = 0
        self._player_y = 32
        self._player_x = 20
        self._goal_y = 32
        self._goal_x = 42

    def _frame(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[self._goal_y, self._goal_x] = 6
        frame[self._player_y, self._player_x] = 4
        return frame

    def get_pixels(self, x, y, w, h):
        return self._frame()[y:y + h, x:x + w]

    def perform_action(self, action_input, raw=True):
        action_id = _action_id(action_input)
        if action_id == 0:
            self.set_level(self._current_level_index)
        elif action_id == 1:
            self.energy += 1
        elif action_id == 3 and self.energy >= 2:
            self._current_level_index = 1
        return SimpleNamespace(frame=[self._frame()], levels_completed=self._current_level_index)
