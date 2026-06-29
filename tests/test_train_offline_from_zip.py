from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_offline_from_zip.py"


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
        return _Action(int(value))


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
        self.frames = []
        self.action_counter = 0
        self.is_playback = False
        self.recorder = None


def _load_script_module():
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

    module_name = "test_train_offline_from_zip_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TinyNet(torch.nn.Module):
    def __init__(self, in_ch=26, g=64):
        super().__init__()
        self.linear = torch.nn.Linear(in_ch, 5 + 64 * 64)

    def forward(self, x, *args, **kwargs):
        pooled = x.mean(dim=(2, 3))
        return self.linear(pooled)

    def forward_actions(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)[:, :5]


class _TinyEncoderOwner:
    @staticmethod
    def _fast_frame_hash(self, frame):
        return hash(np.ascontiguousarray(frame).tobytes())

    @staticmethod
    def _normalized_palette_frame(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        invalid = frame > 15
        if invalid.any():
            frame = frame.copy()
            frame[invalid] = 0
        return frame

    @staticmethod
    def _encode_static_frame_cpu(self, frame, update_bg=False):
        frame = self._normalized_palette_frame(frame)
        counts = np.bincount(frame.ravel(), minlength=16).astype(np.float32, copy=False)
        bg = int(counts.argmax())
        if update_bg:
            self._bg = bg
        out = torch.zeros(21, 64, 64, dtype=torch.float32)
        idx = torch.from_numpy(frame).to(torch.long)
        out[:16].scatter_(0, idx.unsqueeze(0), 1.0)
        out[16] = torch.from_numpy((frame == bg).astype(np.float32, copy=False))
        rarity = (1.0 - counts / max(float(counts[bg]), 1.0)).astype(np.float32, copy=False)
        out[17] = torch.from_numpy(rarity[frame])
        edge = np.zeros((64, 64), dtype=bool)
        edge[1:, :] |= frame[1:, :] != frame[:-1, :]
        edge[:-1, :] |= frame[:-1, :] != frame[1:, :]
        edge[:, 1:] |= frame[:, 1:] != frame[:, :-1]
        edge[:, :-1] |= frame[:, :-1] != frame[:, 1:]
        out[18] = torch.from_numpy(edge.astype(np.float32, copy=False))
        out[19:21] = self._pos_aug
        return out

    @staticmethod
    def _tensor_zero_tail(self, like_tensor):
        return torch.zeros((5, 64, 64), dtype=like_tensor.dtype, device=like_tensor.device)

    @staticmethod
    def _encode_frame_tensor(self, frame):
        frame = self._normalized_palette_frame(frame)
        static = self._encode_static_frame_cpu(frame, update_bg=True)
        return torch.cat([static.to(self.device), self._tensor_zero_tail(static.to(self.device))], dim=0)


def _make_frame(fill: int):
    return np.full((64, 64), fill, dtype=np.uint8).tolist()


def _recording_line(frame_fill: int, action_id: int) -> str:
    return json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {
                "frame": [_make_frame(frame_fill)],
                "action_input": {"id": action_id, "data": {"game_id": "demo"}},
                "available_actions": [1, 2, 3, 4, 5],
                "game_id": "demo",
                "state": "NOT_FINISHED",
            },
        }
    )


class TrainOfflineFromZipTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_script_module()

    def test_iter_training_samples_normalizes_string_ids_and_includes_clicks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("__MACOSX/ignore.recording.jsonl", _recording_line(9, 1))
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, "RESET"),
                            _recording_line(2, "ACTION2"),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(3)],
                                        "action_input": {
                                            "id": "ACTION6",
                                            "data": {"game_id": "demo", "x": 7, "y": 9},
                                        },
                                        "available_actions": [1, 2, 3, 4, 5, 6],
                                        "game_id": "demo",
                                        "state": "NOT_FINISHED",
                                    },
                                }
                            ),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0][1], 1)
            self.assertEqual(samples[1][1], 5 + 9 * 64 + 7)
            self.assertTrue(np.all(samples[0][0] == 1))
            self.assertTrue(np.all(samples[1][0] == 2))

    def test_iter_training_samples_marks_unsupported_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION7"),
                            _recording_line(3, "ACTION2"),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "unsupported", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1][1], 1)

    def test_iter_training_events_filters_actions_missing_from_available_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {"id": "ACTION2", "data": {"game_id": "demo"}},
                                        "available_actions": [1, 3, 4],
                                    },
                                }
                            ),
                            _recording_line(3, "ACTION3"),
                        ]
                    ),
                )
            events = list(self.module.iter_training_events(zip_path))
            self.assertEqual(events[0], (None, "filtered_noisy", "public/demo.recording.jsonl"))
            self.assertEqual(events[1][1], 2)

    def test_iter_training_events_downweights_duplicate_state_action_streaks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(2, "ACTION2"),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )
            events = list(self.module.iter_training_events(zip_path))
            self.assertEqual([event[2] for event in events], [1.0, 1.0, 0.5])

    def test_iter_training_samples_marks_malformed_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, None),
                            _recording_line(3, "bogus"),
                            _recording_line(4, "ACTION2"),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[2][1], 1)

    def test_iter_training_samples_marks_non_dict_data_and_bad_frames_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "data": []}),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [[[1, 2], [3, 4]]],
                                        "action_input": {"id": "ACTION2", "data": {"game_id": "demo"}},
                                    },
                                }
                            ),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1], (None, "malformed", "public/demo.recording.jsonl"))

    def test_iter_training_samples_marks_invalid_json_and_non_dict_action_input_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            '{"timestamp": "2026-01-01T00:00:00+00:00", "data": ',
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(1)],
                                        "action_input": ["bad", "shape"],
                                    },
                                }
                            ),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1], (None, "malformed", "public/demo.recording.jsonl"))

    def test_iter_training_samples_marks_non_object_top_level_json_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            "[]",
                            '"text"',
                            "123",
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(
                samples,
                [
                    (None, "malformed", "public/demo.recording.jsonl"),
                    (None, "malformed", "public/demo.recording.jsonl"),
                    (None, "malformed", "public/demo.recording.jsonl"),
                ],
            )

    def test_iter_training_samples_marks_malformed_click_payloads_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {"id": 6, "data": {}},
                                    },
                                }
                            ),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:02+00:00",
                                    "data": {
                                        "frame": [_make_frame(3)],
                                        "action_input": {"id": "ACTION6", "data": {"x": "bad"}},
                                    },
                                }
                            ),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1], (None, "malformed", "public/demo.recording.jsonl"))

    def test_iter_training_samples_marks_invalid_utf8_json_line_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                bad_bytes = b"\xff\xfe\xfa\n"
                reset_line = json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:01+00:00",
                        "data": {
                            "frame": [_make_frame(1)],
                            "action_input": {"id": 0, "data": {"game_id": "demo"}},
                        },
                    }
                ).encode("utf-8") + b"\n"
                good_line = json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:02+00:00",
                        "data": {
                            "frame": [_make_frame(2)],
                            "action_input": {"id": "ACTION2", "data": {"game_id": "demo"}},
                        },
                    }
                ).encode("utf-8")
                zf.writestr("public/demo.recording.jsonl", bad_bytes + reset_line + good_line)
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[1][1], 1)

    def test_iter_training_samples_breaks_trajectory_after_malformed_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            json.dumps({"timestamp": "2026-01-01T00:00:01+00:00", "data": []}),
                            _recording_line(3, "ACTION3"),
                            _recording_line(4, "ACTION4"),
                        ]
                    ),
                )
            samples = list(self.module.iter_training_samples(zip_path))
            self.assertEqual(samples[0][1], 1)
            self.assertEqual(samples[1], (None, "malformed", "public/demo.recording.jsonl"))
            self.assertEqual(samples[2][1], 3)
            self.assertTrue(np.all(samples[0][0] == 1))
            self.assertTrue(np.all(samples[2][0] == 3))

    def test_normalize_action_id_handles_int_digit_and_named_actions(self):
        self.assertEqual(self.module._normalize_action_id(6), 6)
        self.assertEqual(self.module._normalize_action_id("6"), 6)
        self.assertEqual(self.module._normalize_action_id("ACTION6"), 6)
        self.assertEqual(self.module._normalize_action_id("reset"), 0)
        self.assertIsNone(self.module._normalize_action_id("unknown"))

    def test_shift_action_target_updates_click_coordinates(self):
        shifted = self.module._shift_action_target(5 + 9 * 64 + 7, 1, -1)
        self.assertEqual(shifted, 5 + 8 * 64 + 8)
        self.assertIsNone(self.module._shift_action_target(5 + 0 * 64 + 0, -1, 0))

    def test_sample_batch_shift_is_less_aggressive_for_click_batches(self):
        rng = mock.Mock()
        rng.randint.side_effect = [1, 0]
        rng.random.return_value = 0.3
        shift_dx, shift_dy, do_shift = self.module._sample_batch_shift([5 + 9 * 64 + 7], rng)
        self.assertEqual((shift_dx, shift_dy), (1, 0))
        self.assertFalse(do_shift)

    def test_sample_batch_shift_keeps_existing_probability_for_non_click_batches(self):
        rng = mock.Mock()
        rng.randint.side_effect = [1, 0]
        rng.random.return_value = 0.3
        shift_dx, shift_dy, do_shift = self.module._sample_batch_shift([0, 1, 2], rng)
        self.assertEqual((shift_dx, shift_dy), (1, 0))
        self.assertTrue(do_shift)

    def test_duplicate_sample_weight_downweights_repeated_streaks(self):
        self.assertEqual(self.module._duplicate_sample_weight(1), 1.0)
        self.assertEqual(self.module._duplicate_sample_weight(2), 0.5)
        self.assertEqual(self.module._duplicate_sample_weight(8), 0.25)

    def test_action_class_index_groups_all_clicks_together(self):
        self.assertEqual(self.module._action_class_index(0), 0)
        self.assertEqual(self.module._action_class_index(4), 4)
        self.assertEqual(self.module._action_class_index(5 + 0 * 64 + 0), 5)
        self.assertEqual(self.module._action_class_index(5 + 9 * 64 + 7), 5)

    def test_rebalance_sample_weights_upweights_rare_action_classes(self):
        weights = self.module._rebalance_sample_weights(
            [0, 0, 0, 5 + 9 * 64 + 7],
            [1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(weights[:3], [0.816496580927726] * 3)
        self.assertEqual(weights[3], 1.4142135623730951)

    def test_rebalance_sample_weights_respects_existing_sample_weights(self):
        weights = self.module._rebalance_sample_weights(
            [0, 0, 5 + 9 * 64 + 7],
            [0.5, 1.0, 0.25],
        )
        self.assertEqual(weights, [0.4330127018922193, 0.8660254037844386, 0.30618621784789724])

    def test_epoch_checkpoint_path_appends_epoch_suffix(self):
        output_path = Path("H:/tmp/pretrained_weights_offline.pt")
        checkpoint_path = self.module.epoch_checkpoint_path(output_path, 3)
        self.assertEqual(
            checkpoint_path,
            Path("H:/tmp/pretrained_weights_offline.epoch_3.pt"),
        )

    def test_best_validation_checkpoint_path_appends_best_val_suffix(self):
        output_path = Path("H:/tmp/pretrained_weights_offline.pt")
        checkpoint_path = self.module.best_validation_checkpoint_path(output_path)
        self.assertEqual(
            checkpoint_path,
            Path("H:/tmp/pretrained_weights_offline.best_val_loss.pt"),
        )

    def test_resolve_init_weights_path_prefers_explicit_init_weights(self):
        args = types.SimpleNamespace(
            init_weights=Path("H:/tmp/explicit.pt"),
            start_from_existing_weights=True,
            output=Path("H:/tmp/output.pt"),
        )
        self.assertEqual(
            self.module.resolve_init_weights_path(args),
            Path("H:/tmp/explicit.pt"),
        )

    def test_resolve_init_weights_path_uses_output_when_requested(self):
        args = types.SimpleNamespace(
            init_weights=None,
            start_from_existing_weights=True,
            output=Path("H:/tmp/output.pt"),
        )
        self.assertEqual(
            self.module.resolve_init_weights_path(args),
            Path("H:/tmp/output.pt"),
        )

    def test_parse_args_defaults_to_larger_batch_size(self):
        args = self.module.parse_args([])
        self.assertEqual(args.batch_size, 256)

    def test_parse_args_defaults_label_smoothing(self):
        args = self.module.parse_args([])
        self.assertEqual(args.label_smoothing, 0.05)

    def test_parse_args_defaults_to_lower_learning_rate(self):
        args = self.module.parse_args([])
        self.assertEqual(args.lr, 1e-4)

    def test_parse_args_defaults_validation_fraction(self):
        args = self.module.parse_args([])
        self.assertEqual(args.validation_fraction, 0.1)

    def test_split_recording_members_is_deterministic_and_non_overlapping(self):
        members = ["a.recording.jsonl", "b.recording.jsonl", "c.recording.jsonl", "d.recording.jsonl"]
        train_a, val_a = self.module.split_recording_members(members, validation_fraction=0.25, seed=7)
        train_b, val_b = self.module.split_recording_members(members, validation_fraction=0.25, seed=7)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(val_a), 1)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(sorted(train_a + val_a), sorted(members))

    def test_main_streams_zip_and_saves_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            out_path = Path(tmp_dir) / "weights.pt"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, 1),
                            _recording_line(3, 2),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(4)],
                                        "action_input": {
                                            "id": 6,
                                            "data": {"game_id": "demo", "x": 5, "y": 6},
                                        },
                                        "available_actions": [1, 2, 3, 4, 5, 6],
                                        "game_id": "demo",
                                        "state": "NOT_FINISHED",
                                    },
                                }
                            ),
                            _recording_line(5, "ACTION4"),
                        ]
                    ),
                )
            with mock.patch.object(
                self.module,
                "load_agent_components",
                return_value=(_TinyNet, _TinyEncoderOwner),
            ):
                rc = self.module.main(
                    [
                        "--zip",
                        str(zip_path),
                        "--output",
                        str(out_path),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            self.assertTrue((Path(tmp_dir) / "weights.epoch_1.pt").exists())

    def test_main_start_from_existing_weights_loads_output_first(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            out_path = Path(tmp_dir) / "weights.pt"
            out_path.write_bytes(b"existing")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            with mock.patch.object(
                self.module,
                "load_agent_components",
                return_value=(_TinyNet, _TinyEncoderOwner),
            ):
                with mock.patch.object(
                    self.module.OfflineBehaviorCloner,
                    "load_weights",
                    return_value=123,
                ) as load_weights_mock:
                    rc = self.module.main(
                        [
                            "--zip",
                            str(zip_path),
                            "--output",
                            str(out_path),
                            "--epochs",
                            "1",
                            "--batch-size",
                            "1",
                            "--device",
                            "cpu",
                            "--start-from-existing-weights",
                        ]
                    )
            self.assertEqual(rc, 0)
            load_weights_mock.assert_called_once_with(out_path)

    def test_main_start_from_existing_weights_requires_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            out_path = Path(tmp_dir) / "weights.pt"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            with mock.patch.object(
                self.module,
                "load_agent_components",
                return_value=(_TinyNet, _TinyEncoderOwner),
            ):
                with self.assertRaises(SystemExit) as exc:
                    self.module.main(
                        [
                            "--zip",
                            str(zip_path),
                            "--output",
                            str(out_path),
                            "--epochs",
                            "1",
                            "--batch-size",
                            "1",
                            "--device",
                            "cpu",
                            "--start-from-existing-weights",
                        ]
                    )
            self.assertIn("Initial weights not found", str(exc.exception))

    def test_train_from_zip_saves_checkpoint_after_each_epoch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            checkpoint_base = Path(tmp_dir) / "weights.pt"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(3, "ACTION3"),
                        ]
                    ),
                )

            saved_paths = []

            class _CheckpointingTrainer:
                def train_batch(self, frames, action_targets):
                    return (0.25, len(frames))

                def save_weights(self, output_path):
                    saved_paths.append(Path(output_path))
                    Path(output_path).write_bytes(b"checkpoint")

            stats = self.module.train_from_zip(
                zip_path,
                _CheckpointingTrainer(),
                epochs=3,
                batch_size=2,
                checkpoint_output_path=checkpoint_base,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 6)
            self.assertEqual(
                saved_paths,
                [
                    Path(tmp_dir) / "weights.epoch_1.pt",
                    Path(tmp_dir) / "weights.epoch_2.pt",
                    Path(tmp_dir) / "weights.epoch_3.pt",
                ],
            )
            for saved_path in saved_paths:
                self.assertTrue(saved_path.exists())

    def test_train_from_zip_saves_best_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            checkpoint_base = Path(tmp_dir) / "weights.pt"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for member_name, fill in (
                    ("public/a.recording.jsonl", 1),
                    ("public/b.recording.jsonl", 2),
                    ("public/c.recording.jsonl", 3),
                    ("public/d.recording.jsonl", 4),
                ):
                    zf.writestr(
                        member_name,
                        "\n".join(
                            [
                                _recording_line(fill, 0),
                                _recording_line(fill + 1, "ACTION2"),
                            ]
                        ),
                    )

            saved_paths = []
            validation_losses = iter([0.9, 0.7, 0.8])

            class _Trainer:
                def train_batch(self, frames, action_targets, sample_weights=None):
                    return (0.25, len(frames))

                def evaluate_batch(self, frames, action_targets, *, sample_weights=None):
                    return {
                        "loss": next(validation_losses),
                        "samples": len(frames),
                        "correct": len(frames),
                    }

                def save_weights(self, output_path):
                    saved_paths.append(Path(output_path))
                    Path(output_path).write_bytes(b"checkpoint")

            stats = self.module.train_from_zip(
                zip_path,
                _Trainer(),
                epochs=3,
                batch_size=2,
                checkpoint_output_path=checkpoint_base,
                shuffle_members=False,
                seed=3,
                validation_fraction=0.25,
            )
            self.assertEqual(stats["best_validation_epoch"], 2)
            self.assertEqual(stats["best_validation_loss"], 0.7)
            self.assertEqual(stats["best_validation_accuracy"], 1.0)
            self.assertIn(Path(tmp_dir) / "weights.best_val_loss.pt", saved_paths)

    def test_train_from_zip_logs_progress_in_later_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(3, "ACTION3"),
                            _recording_line(4, "ACTION4"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return (0.25, len(frames))

            with mock.patch.object(self.module.logger, "info") as info_mock:
                self.module.train_from_zip(
                    zip_path,
                    _CountingTrainer(),
                    epochs=3,
                    batch_size=2,
                    shuffle_members=False,
                    log_every=4,
                )

            progress_epochs = [
                call.args[1]
                for call in info_mock.call_args_list
                if call.args and call.args[0] == "epoch %s: streamed=%s trained=%s optimizer_steps=%s latest_loss=%.4f"
            ]
            self.assertEqual(progress_epochs, [2, 3])

    def test_evaluate_from_zip_reports_loss_accuracy_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(3, "ACTION3"),
                            _recording_line(4, "ACTION7"),
                        ]
                    ),
                )

            class _EvalTrainer:
                def evaluate_batch(self, frames, action_targets, *, sample_weights=None):
                    return {
                        "loss": 0.75,
                        "samples": len(frames),
                        "correct": len(frames) - 1,
                    }

            stats = self.module.evaluate_from_zip(
                zip_path,
                _EvalTrainer(),
                member_names=["public/demo.recording.jsonl"],
                batch_size=2,
            )
            self.assertEqual(stats["samples"], 2)
            self.assertEqual(stats["loss"], 0.75)
            self.assertEqual(stats["accuracy"], 0.5)
            self.assertEqual(stats["skipped_unsupported"], 1)

    def test_train_from_zip_reports_validation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for member_name, fill in (
                    ("public/a.recording.jsonl", 1),
                    ("public/b.recording.jsonl", 2),
                    ("public/c.recording.jsonl", 3),
                    ("public/d.recording.jsonl", 4),
                ):
                    zf.writestr(
                        member_name,
                        "\n".join(
                            [
                                _recording_line(fill, 0),
                                _recording_line(fill + 1, "ACTION2"),
                            ]
                        ),
                    )

            class _Trainer:
                def train_batch(self, frames, action_targets, sample_weights=None):
                    return (0.25, len(frames))

                def evaluate_batch(self, frames, action_targets, *, sample_weights=None):
                    return {
                        "loss": 0.5,
                        "samples": len(frames),
                        "correct": len(frames),
                    }

            with mock.patch.object(self.module.logger, "info") as info_mock:
                stats = self.module.train_from_zip(
                    zip_path,
                    _Trainer(),
                    epochs=1,
                    batch_size=2,
                    shuffle_members=False,
                    seed=3,
                    validation_fraction=0.25,
                )
            self.assertGreater(stats["validation_samples"], 0)
            self.assertEqual(stats["validation_loss"], 0.5)
            self.assertEqual(stats["validation_accuracy"], 1.0)
            validation_logs = [
                call.args[0]
                for call in info_mock.call_args_list
                if call.args and isinstance(call.args[0], str) and call.args[0].startswith("epoch %s validation:")
            ]
            self.assertEqual(len(validation_logs), 1)

    def test_max_samples_is_global_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, 1),
                            _recording_line(3, "ACTION2"),
                            _recording_line(4, "ACTION3"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def __init__(self):
                    self.total_seen = 0

                def train_batch(self, frames, action_targets):
                    self.total_seen += len(frames)
                    return 0.5

            trainer = _CountingTrainer()
            stats = self.module.train_from_zip(
                zip_path,
                trainer,
                epochs=3,
                batch_size=2,
                max_samples=2,
                shuffle_members=False,
            )
            self.assertEqual(trainer.total_seen, 2)
            self.assertEqual(stats["samples"], 2)
            self.assertGreaterEqual(stats["streamed_samples"], 2)

    def test_batch_size_controls_optimizer_batching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(3, "ACTION3"),
                            _recording_line(4, "ACTION4"),
                            _recording_line(5, "ACTION5"),
                        ]
                    ),
                )

            seen_batch_sizes = []

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    seen_batch_sizes.append(len(frames))
                    return 0.5

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=2,
                shuffle_members=False,
            )
            self.assertEqual(seen_batch_sizes, [2, 2])
            self.assertEqual(stats["samples"], 4)
            self.assertEqual(stats["steps"], 2)

    def test_train_batch_drops_shifted_clicks_that_leave_board(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
        )
        frame = np.zeros((64, 64), dtype=np.uint8)
        click_target = 5 + 0 * 64 + 0
        rng = self.module.random.Random(0)
        with mock.patch.object(rng, "randint", side_effect=[-1, 0]):
            with mock.patch.object(rng, "random", return_value=0.0):
                result = trainer.train_batch([frame], [click_target], rng=rng)
        self.assertIsNone(result)

    def test_train_from_zip_reports_unsupported_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION7"),
                            _recording_line(3, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["streamed_samples"], 1)
            self.assertEqual(stats["skipped_unsupported"], 1)
            self.assertEqual(stats["skipped_malformed"], 0)

    def test_train_from_zip_reports_malformed_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, None),
                            _recording_line(3, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_unsupported"], 0)
            self.assertEqual(stats["skipped_malformed"], 1)

    def test_train_from_zip_counts_non_dict_data_and_bad_frames_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "data": []}),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [[[1, 2], [3, 4]]],
                                        "action_input": {"id": "ACTION2", "data": {"game_id": "demo"}},
                                    },
                                }
                            ),
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_malformed"], 2)

    def test_train_from_zip_counts_filtered_noisy_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {"id": "ACTION2", "data": {"game_id": "demo"}},
                                        "available_actions": [1, 3, 4],
                                    },
                                }
                            ),
                            _recording_line(3, "ACTION3"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets, sample_weights=None):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["filtered_noisy"], 1)

    def test_train_from_zip_counts_invalid_json_and_non_dict_action_input_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            '{"timestamp": "2026-01-01T00:00:00+00:00", "data": ',
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(1)],
                                        "action_input": ["bad", "shape"],
                                    },
                                }
                            ),
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_malformed"], 2)

    def test_train_from_zip_counts_invalid_utf8_json_line_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                bad_bytes = b"\xff\xfe\xfa\n"
                good_reset = _recording_line(1, 0).encode("utf-8") + b"\n"
                good_action = _recording_line(2, "ACTION2").encode("utf-8")
                zf.writestr("public/demo.recording.jsonl", bad_bytes + good_reset + good_action)

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_malformed"], 1)

    def test_train_from_zip_counts_non_object_top_level_json_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            "[]",
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_malformed"], 1)

    def test_train_from_zip_classifies_malformed_click_payloads_as_malformed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {"id": 6, "data": {}},
                                    },
                                }
                            ),
                            _recording_line(3, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.25

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=4,
                shuffle_members=False,
            )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["skipped_unsupported"], 0)
            self.assertEqual(stats["skipped_malformed"], 1)

    def test_train_from_zip_does_not_count_dropped_batch_as_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {
                                            "id": 6,
                                            "data": {"game_id": "demo", "x": 0, "y": 0},
                                        },
                                        "available_actions": [1, 2, 3, 4, 5, 6],
                                        "game_id": "demo",
                                        "state": "NOT_FINISHED",
                                    },
                                }
                            ),
                        ]
                    ),
                )

            trainer = self.module.OfflineBehaviorCloner(
                _TinyNet,
                _TinyEncoderOwner,
                device=torch.device("cpu"),
                lr=3e-4,
                weight_decay=1e-5,
            )
            rng = self.module.random.Random(0)
            with mock.patch.object(rng, "randint", side_effect=[-1, 0]):
                with mock.patch.object(rng, "random", return_value=0.0):
                    stats = self.module.train_from_zip(
                        zip_path,
                        trainer,
                        epochs=1,
                        batch_size=1,
                        shuffle_members=False,
                        seed=123,
                    )
            self.assertEqual(stats["samples"], 0)
            self.assertEqual(stats["streamed_samples"], 1)
            self.assertEqual(stats["steps"], 0)
            self.assertEqual(stats["avg_loss"], 0.0)
            self.assertEqual(stats["dropped_after_augmentation"], 1)

    def test_train_from_zip_respects_zero_max_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def __init__(self):
                    self.calls = 0

                def train_batch(self, frames, action_targets):
                    self.calls += 1
                    return 0.5

            trainer = _CountingTrainer()
            stats = self.module.train_from_zip(
                zip_path,
                trainer,
                epochs=3,
                batch_size=1,
                max_samples=0,
                shuffle_members=False,
            )
            self.assertEqual(trainer.calls, 0)
            self.assertEqual(stats["samples"], 0)
            self.assertEqual(stats["streamed_samples"], 0)
            self.assertEqual(stats["steps"], 0)
            self.assertEqual(stats["skipped_malformed"], 0)

    def test_train_batch_uses_per_sample_background_for_padding(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
        )
        frame_a = np.full((64, 64), 5, dtype=np.uint8)
        frame_b = np.full((64, 64), 2, dtype=np.uint8)
        pad_values = []
        original_pad = self.module.np.pad

        def _capture_pad(array, pad_width, mode="constant", constant_values=0):
            pad_values.append(constant_values)
            return original_pad(
                array,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )

        with mock.patch.object(self.module.random, "randint", side_effect=[1, 0]):
            pass
        rng = self.module.random.Random(0)
        with mock.patch.object(rng, "randint", side_effect=[1, 0]):
            with mock.patch.object(rng, "random", return_value=0.0):
                with mock.patch.object(self.module.np, "pad", side_effect=_capture_pad):
                    result = trainer.train_batch([frame_a, frame_b], [0, 1], rng=rng)
        self.assertIsNotNone(result)
        loss, trained_count = result
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(trained_count, 2)
        self.assertEqual(pad_values, [5, 2])

    def test_train_batch_click_targets_shift_less_aggressively(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
        )
        frame = np.full((64, 64), 5, dtype=np.uint8)
        rng = self.module.random.Random(0)
        with mock.patch.object(rng, "randint", side_effect=[1, 0]):
            with mock.patch.object(rng, "random", return_value=0.3):
                with mock.patch.object(self.module.np, "pad") as pad_mock:
                    result = trainer.train_batch([frame], [5 + 9 * 64 + 7], rng=rng)
        self.assertIsNotNone(result)
        self.assertFalse(pad_mock.called)

    def test_train_batch_rebalances_action_classes_before_loss(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
        )
        frames = [np.full((64, 64), fill, dtype=np.uint8) for fill in (1, 2, 3, 4)]
        targets = [0, 0, 0, 5 + 9 * 64 + 7]
        captured = {}
        original_tensor = self.module.torch.tensor

        def _capture_tensor(data, *args, **kwargs):
            if kwargs.get("dtype") is torch.float32 and isinstance(data, list):
                captured["weights"] = list(data)
            return original_tensor(data, *args, **kwargs)

        rng = self.module.random.Random(0)
        with mock.patch.object(rng, "randint", side_effect=[0, 0]):
            with mock.patch.object(self.module.torch, "tensor", side_effect=_capture_tensor):
                result = trainer.train_batch(frames, targets, sample_weights=[1.0] * 4, rng=rng)
        self.assertIsNotNone(result)
        self.assertEqual(
            captured["weights"],
            self.module._rebalance_sample_weights(targets, [1.0] * 4),
        )

    def test_train_batch_passes_label_smoothing_to_cross_entropy(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
            label_smoothing=0.15,
        )
        frame = np.full((64, 64), 5, dtype=np.uint8)
        captured = {}
        original_cross_entropy = self.module.F.cross_entropy

        def _capture_cross_entropy(*args, **kwargs):
            captured["label_smoothing"] = kwargs.get("label_smoothing")
            return original_cross_entropy(*args, **kwargs)

        rng = self.module.random.Random(0)
        with mock.patch.object(rng, "randint", side_effect=[0, 0]):
            with mock.patch.object(self.module.F, "cross_entropy", side_effect=_capture_cross_entropy):
                result = trainer.train_batch([frame], [0], rng=rng)
        self.assertIsNotNone(result)
        self.assertEqual(captured["label_smoothing"], 0.15)

    def test_load_weights_falls_back_when_weights_only_is_unsupported(self):
        trainer = self.module.OfflineBehaviorCloner(
            _TinyNet,
            _TinyEncoderOwner,
            device=torch.device("cpu"),
            lr=3e-4,
            weight_decay=1e-5,
        )
        state = trainer.net.state_dict()
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "weights.pt"
            torch.save(state, weights_path)
            original_load = self.module.torch.load

            def _fake_load(*args, **kwargs):
                if kwargs.get("weights_only") is True:
                    raise TypeError("weights_only unsupported")
                return original_load(*args, **kwargs)

            with mock.patch.object(self.module.torch, "load", side_effect=_fake_load):
                loaded_keys = trainer.load_weights(weights_path)
        self.assertGreater(loaded_keys, 0)

    def test_train_from_zip_seed_controls_augmentation_rng(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, 1),
                            _recording_line(3, 2),
                            _recording_line(4, 3),
                        ]
                    ),
                )

            calls = []

            class _CapturingTrainer:
                def train_batch(self, frames, action_targets, *, rng=None):
                    calls.append((rng.randint(-1, 1), rng.randint(-1, 1), round(rng.random(), 6)))
                    return (0.5, len(frames))

            stats = self.module.train_from_zip(
                zip_path,
                _CapturingTrainer(),
                epochs=2,
                batch_size=2,
                shuffle_members=False,
                seed=99,
            )
            self.assertEqual(stats["samples"], 6)
            self.assertEqual(
                calls,
                [
                    (0, 0, 0.200075),
                    (-1, -1, 0.248431),
                    (-1, 0, 0.729031),
                    (1, 1, 0.700488),
                ],
            )

    def test_max_samples_targets_trained_samples_not_streamed_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "frame": [_make_frame(2)],
                                        "action_input": {
                                            "id": 6,
                                            "data": {"game_id": "demo", "x": 0, "y": 0},
                                        },
                                        "available_actions": [1, 2, 3, 4, 5, 6],
                                        "game_id": "demo",
                                        "state": "NOT_FINISHED",
                                    },
                                }
                            ),
                            _recording_line(3, "ACTION2"),
                        ]
                    ),
                )

            trainer = self.module.OfflineBehaviorCloner(
                _TinyNet,
                _TinyEncoderOwner,
                device=torch.device("cpu"),
                lr=3e-4,
                weight_decay=1e-5,
            )
            rng = self.module.random.Random(0)
            with mock.patch.object(rng, "randint", side_effect=[-1, 0, 0, 0]):
                with mock.patch.object(rng, "random", side_effect=[0.0, 1.0]):
                    with mock.patch.object(self.module.random, "Random", return_value=rng):
                        stats = self.module.train_from_zip(
                            zip_path,
                            trainer,
                            epochs=1,
                            batch_size=1,
                            max_samples=1,
                            shuffle_members=False,
                            seed=123,
                        )
            self.assertEqual(stats["samples"], 1)
            self.assertEqual(stats["streamed_samples"], 2)
            self.assertEqual(stats["dropped_after_augmentation"], 1)

    def test_max_samples_does_not_overshoot_with_large_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                            _recording_line(3, "ACTION3"),
                            _recording_line(4, "ACTION4"),
                        ]
                    ),
                )

            seen_batch_sizes = []

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    seen_batch_sizes.append(len(frames))
                    return 0.5

            stats = self.module.train_from_zip(
                zip_path,
                _CountingTrainer(),
                epochs=1,
                batch_size=128,
                max_samples=1,
                shuffle_members=False,
            )
            self.assertEqual(seen_batch_sizes, [1])
            self.assertEqual(stats["samples"], 1)

    def test_train_from_zip_rejects_non_positive_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "public/demo.recording.jsonl",
                    "\n".join(
                        [
                            _recording_line(1, 0),
                            _recording_line(2, "ACTION2"),
                        ]
                    ),
                )

            class _CountingTrainer:
                def train_batch(self, frames, action_targets):
                    return 0.5

            with self.assertRaises(SystemExit):
                self.module.train_from_zip(
                    zip_path,
                    _CountingTrainer(),
                    epochs=1,
                    batch_size=0,
                    shuffle_members=False,
                )


if __name__ == "__main__":
    unittest.main()
