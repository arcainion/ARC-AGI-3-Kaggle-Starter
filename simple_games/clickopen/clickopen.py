from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _action_id(action_input) -> int:
    action_id = getattr(action_input, "id", action_input)
    return int(action_id.value) if hasattr(action_id, "value") else int(action_id)


class Clickopen:
    """Tiny click-plus-move game: click the switch, then walk to the goal."""

    def __init__(self):
        self.win_levels = 1
        self._available_actions = [1, 2, 3, 4, 5, 6]
        self._current_level_index = 0
        self.set_level(0)

    def set_level(self, level_idx: int):
        self._current_level_index = int(level_idx)
        self._player_y = 32
        self._player_x = 8
        self._goal_y = 32
        self._goal_x = 26
        self._switch_y = 32
        self._switch_x = 14
        self._gate_open = False

    def _gate_cells(self):
        for y in range(20, 45):
            if y != self._switch_y:
                yield y, 18

    def _frame(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[self._goal_y, self._goal_x] = 6
        frame[self._switch_y, self._switch_x] = 14
        if not self._gate_open:
            for y, x in self._gate_cells():
                frame[y, x] = 9
        frame[self._player_y, self._player_x] = 4
        return frame

    def get_pixels(self, x, y, w, h):
        return self._frame()[y:y + h, x:x + w]

    def _blocked(self, new_y, new_x):
        return (not self._gate_open) and new_x == 18 and any(y == new_y for y, _ in self._gate_cells())

    def perform_action(self, action_input, raw=True):
        action_id = _action_id(action_input)
        if action_id == 0:
            self.set_level(self._current_level_index)
        elif action_id == 6:
            data = getattr(action_input, "data", None) or {}
            click_x = int(data.get("x", -999))
            click_y = int(data.get("y", -999))
            if abs(click_x - self._switch_x) + abs(click_y - self._switch_y) <= 1:
                self._gate_open = True
        else:
            new_y, new_x = self._player_y, self._player_x
            if action_id == 1:
                new_y = max(1, self._player_y - 1)
            elif action_id == 2:
                new_y = min(62, self._player_y + 1)
            elif action_id == 3:
                new_x = max(1, self._player_x - 1)
            elif action_id == 4:
                new_x = min(62, self._player_x + 1)
            if not self._blocked(new_y, new_x):
                self._player_y, self._player_x = new_y, new_x
        if self._player_x >= self._goal_x and self._player_y == self._goal_y:
            self._current_level_index = 1
        return SimpleNamespace(frame=[self._frame()], levels_completed=self._current_level_index)
