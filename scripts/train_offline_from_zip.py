r"""Train `agent/my_agent.py` offline from zipped ARC-AGI-3 recordings.

The dataset is read directly from `*.recording.jsonl` members inside the zip;
the archive is never unpacked to disk.

Example:
    .venv\Scripts\python.exe scripts\train_offline_from_zip.py ^
        --zip arc_agi_3_public_demo_human_testing.zip ^
        --output pretrained_weights.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


def recording_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            name for name in zf.namelist()
            if name.endswith(".recording.jsonl") and "__MACOSX" not in name
        ]


def epoch_checkpoint_path(output_path: Path, epoch: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.epoch_{int(epoch)}{output_path.suffix}")


def resolve_init_weights_path(args) -> Path | None:
    if args.init_weights is not None:
        return args.init_weights
    if getattr(args, "start_from_existing_weights", False):
        return args.output
    return None


def best_validation_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.best_val_loss{output_path.suffix}")


def _extract_frame_array(frame_payload) -> np.ndarray | None:
    if frame_payload is None:
        return None
    if isinstance(frame_payload, list) and frame_payload:
        candidate = frame_payload[-1]
    else:
        candidate = frame_payload
    frame = np.ascontiguousarray(candidate, dtype=np.uint8)
    if frame.shape != (64, 64):
        return None
    invalid = frame > 15
    if invalid.any():
        frame = frame.copy()
        frame[invalid] = 0
    return frame


def _normalize_action_id(action_id) -> int | None:
    if action_id is None:
        return None
    if isinstance(action_id, str):
        normalized = action_id.strip().upper()
        if normalized == "RESET":
            return 0
        if normalized.startswith("ACTION"):
            suffix = normalized[6:]
            if suffix.isdigit():
                return int(suffix)
        if normalized.isdigit():
            return int(normalized)
        return None
    try:
        return int(action_id)
    except (TypeError, ValueError):
        return None


def _action_target_index(action_input: dict) -> tuple[int | None, str | None]:
    action_id = _normalize_action_id(action_input.get("id"))
    if action_id is None:
        return None, "malformed"
    if 1 <= action_id <= 5:
        return action_id - 1, None
    if action_id != 6:
        return None, "unsupported"
    data = action_input.get("data") or {}
    if not isinstance(data, dict):
        return None, "malformed"
    try:
        x = int(data["x"])
        y = int(data["y"])
    except (KeyError, TypeError, ValueError):
        return None, "malformed"
    if not (0 <= x < 64 and 0 <= y < 64):
        return None, "malformed"
    return 5 + y * 64 + x, None


def _decode_action_target(action_target: int) -> tuple[int, tuple[int, int] | None]:
    action_target = int(action_target)
    if action_target < 5:
        return action_target, None
    click_index = action_target - 5
    y, x = divmod(click_index, 64)
    return 5, (y, x)


def _shift_action_target(action_target: int, shift_dx: int, shift_dy: int) -> int | None:
    action_idx, click_coord = _decode_action_target(action_target)
    if click_coord is None:
        return int(action_target)
    y, x = click_coord
    shifted_x = x + int(shift_dx)
    shifted_y = y + int(shift_dy)
    if not (0 <= shifted_x < 64 and 0 <= shifted_y < 64):
        return None
    return 5 + shifted_y * 64 + shifted_x


def _sample_batch_shift(
    action_targets: list[int],
    rng: random.Random,
) -> tuple[int, int, bool]:
    shift_dx = rng.randint(-1, 1)
    shift_dy = rng.randint(-1, 1)
    if shift_dx == 0 and shift_dy == 0:
        return shift_dx, shift_dy, False
    has_click = any(int(action_target) >= 5 for action_target in action_targets)
    shift_probability = 0.2 if has_click else 0.5
    return shift_dx, shift_dy, rng.random() < shift_probability


def _normalize_available_actions(available_actions) -> set[int] | None:
    if available_actions is None:
        return None
    if not isinstance(available_actions, list):
        return None
    normalized = set()
    for action_id in available_actions:
        normalized_action_id = _normalize_action_id(action_id)
        if normalized_action_id is not None:
            normalized.add(normalized_action_id)
    return normalized


def _duplicate_sample_weight(duplicate_streak: int) -> float:
    duplicate_streak = max(1, int(duplicate_streak))
    return max(0.25, 1.0 / float(duplicate_streak))


def _action_class_index(action_target: int) -> int:
    action_target = int(action_target)
    if action_target < 5:
        return action_target
    return 5


def _rebalance_sample_weights(
    action_targets: list[int],
    sample_weights: list[float],
) -> list[float]:
    if not action_targets:
        return []
    class_counts: dict[int, int] = {}
    for action_target in action_targets:
        class_index = _action_class_index(action_target)
        class_counts[class_index] = class_counts.get(class_index, 0) + 1
    active_class_count = max(1, len(class_counts))
    rebalanced = []
    for action_target, sample_weight in zip(action_targets, sample_weights):
        class_index = _action_class_index(action_target)
        class_weight = math.sqrt(len(action_targets) / float(active_class_count * class_counts[class_index]))
        rebalanced.append(float(sample_weight) * class_weight)
    return rebalanced


def split_recording_members(
    members: list[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not members:
        return [], []
    validation_fraction = min(max(float(validation_fraction), 0.0), 1.0)
    if validation_fraction <= 0.0 or len(members) < 2:
        return list(members), []
    shuffled = list(members)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_fraction))
    validation_count = max(1, min(len(shuffled) - 1, validation_count))
    validation_members = shuffled[:validation_count]
    training_members = shuffled[validation_count:]
    return training_members, validation_members


def _normalize_train_batch_result(result, input_count: int) -> tuple[float, int] | None:
    if result is None:
        return None
    if isinstance(result, tuple):
        loss, trained_count = result
        return float(loss), int(trained_count)
    return float(result), int(input_count)


def _call_train_batch(trainer, frames, action_targets, sample_weights, rng) -> tuple[float, int] | None:
    try:
        result = trainer.train_batch(
            frames,
            action_targets,
            sample_weights=sample_weights,
            rng=rng,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'sample_weights'" in str(exc):
            try:
                result = trainer.train_batch(frames, action_targets, rng=rng)
            except TypeError as inner_exc:
                if "unexpected keyword argument 'rng'" not in str(inner_exc):
                    raise
                result = trainer.train_batch(frames, action_targets)
        elif "unexpected keyword argument 'rng'" in str(exc):
            result = trainer.train_batch(frames, action_targets, sample_weights=sample_weights)
        else:
            raise
    return _normalize_train_batch_result(result, len(frames))


def iter_training_events(
    zip_path: Path,
    *,
    member_names: list[str] | None = None,
    max_recordings: int | None = None,
    max_samples: int | None = None,
):
    with zipfile.ZipFile(zip_path) as zf:
        members = member_names if member_names is not None else [
            name for name in zf.namelist()
            if name.endswith(".recording.jsonl") and "__MACOSX" not in name
        ]
        sample_count = 0
        for member_index, member_name in enumerate(members):
            if max_recordings is not None and member_index >= max_recordings:
                break
            previous_frame = None
            previous_frame_hash = None
            previous_sample_signature = None
            duplicate_streak = 0
            with zf.open(member_name) as handle:
                for raw_line in handle:
                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        previous_frame = None
                        previous_frame_hash = None
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                        continue
                    if not isinstance(row, dict):
                        previous_frame = None
                        previous_frame_hash = None
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                        continue
                    data = row.get("data")
                    if not isinstance(data, dict):
                        previous_frame = None
                        previous_frame_hash = None
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                        continue
                    current_frame = _extract_frame_array(data.get("frame"))
                    if current_frame is None:
                        previous_frame = None
                        previous_frame_hash = None
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                        continue
                    action_input = data.get("action_input") or {}
                    if not isinstance(action_input, dict):
                        previous_frame = None
                        previous_frame_hash = None
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                        continue
                    action_id = _normalize_action_id(action_input.get("id"))
                    if action_id == 0 or previous_frame is None:
                        previous_frame = current_frame
                        previous_frame_hash = hash(current_frame.tobytes())
                        previous_sample_signature = None
                        duplicate_streak = 0
                        continue
                    action_target, classification = _action_target_index(action_input)
                    if action_target is not None:
                        available_actions = _normalize_available_actions(data.get("available_actions"))
                        if available_actions is not None and action_id not in available_actions:
                            previous_frame = current_frame
                            previous_frame_hash = hash(current_frame.tobytes())
                            previous_sample_signature = None
                            duplicate_streak = 0
                            yield None, "filtered_noisy", member_name
                            continue
                        sample_signature = (previous_frame_hash, int(action_target))
                        if sample_signature == previous_sample_signature:
                            duplicate_streak += 1
                        else:
                            previous_sample_signature = sample_signature
                            duplicate_streak = 1
                        yield previous_frame, action_target, _duplicate_sample_weight(duplicate_streak), member_name
                        sample_count += 1
                        if max_samples is not None and sample_count >= max_samples:
                            return
                    elif classification == "unsupported":
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "unsupported", member_name
                    else:
                        previous_sample_signature = None
                        duplicate_streak = 0
                        yield None, "malformed", member_name
                    previous_frame = current_frame
                    previous_frame_hash = hash(current_frame.tobytes())


def iter_training_samples(
    zip_path: Path,
    *,
    member_names: list[str] | None = None,
    max_recordings: int | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, int, str] | tuple[None, str, str]:
    """Yield `(frame_before_action, action_target_index, member_name)` samples."""
    for event in iter_training_events(
        zip_path,
        member_names=member_names,
        max_recordings=max_recordings,
        max_samples=max_samples,
    ):
        if event[0] is None:
            _, classification, member_name = event
            if classification == "filtered_noisy":
                continue
            yield None, classification, member_name
            continue
        frame, action_target, _sample_weight, member_name = event
        yield frame, action_target, member_name


def _frame_symbolic_summary(frame: np.ndarray) -> dict[str, object]:
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    counts = np.bincount(frame.ravel(), minlength=16)
    background = int(counts.argmax()) if counts.size else 0
    palette = [int(value) for value in np.unique(frame).tolist() if int(value) != background][:8]
    edge = np.zeros((64, 64), dtype=bool)
    edge[1:, :] |= frame[1:, :] != frame[:-1, :]
    edge[:-1, :] |= frame[:-1, :] != frame[1:, :]
    edge[:, 1:] |= frame[:, 1:] != frame[:, :-1]
    edge[:, :-1] |= frame[:, :-1] != frame[:, 1:]
    return {
        "background": background,
        "palette": palette,
        "non_background_pixels": int(np.count_nonzero(frame != background)),
        "edge_pixels": int(np.count_nonzero(edge)),
        "frame_hash": int(hash(frame.tobytes())),
    }


def _hyperon_bootstrap_record(
    frame: np.ndarray,
    *,
    action_target: int,
    sample_weight: float,
    member_name: str,
) -> dict[str, object]:
    symbolic_state = _frame_symbolic_summary(frame)
    frame_hash = int(symbolic_state["frame_hash"])
    state_atom = f"(state offline {frame_hash})"
    action_atom = f"(observed-action {frame_hash} {int(action_target)})"
    return {
        "member_name": member_name,
        "action_target": int(action_target),
        "sample_weight": float(sample_weight),
        "symbolic_state": symbolic_state,
        "hyperon_bootstrap": {
            "atoms": [
                state_atom,
                f"(background {frame_hash} {int(symbolic_state['background'])})",
                action_atom,
                f"(sample-weight {frame_hash} {float(sample_weight):.6f})",
            ] + [
                f"(palette {frame_hash} {int(color)})" for color in symbolic_state["palette"]
            ],
        },
    }


class HyperonCorpusBuilder:
    """Build a lightweight Hyperon bootstrap corpus for symbolic agents."""

    def build_from_zip(
        self,
        zip_path: Path,
        output_path: Path,
        *,
        max_recordings: int | None = None,
        max_samples: int | None = None,
    ) -> dict[str, int]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples = 0
        skipped_unsupported = 0
        skipped_malformed = 0
        filtered_noisy = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for event in iter_training_events(
                zip_path,
                max_recordings=max_recordings,
                max_samples=max_samples,
            ):
                if event[0] is None:
                    _, classification, _member_name = event
                    if classification == "unsupported":
                        skipped_unsupported += 1
                    elif classification == "filtered_noisy":
                        filtered_noisy += 1
                    else:
                        skipped_malformed += 1
                    continue
                frame, action_target, sample_weight, member_name = event
                payload = _hyperon_bootstrap_record(
                    frame,
                    action_target=int(action_target),
                    sample_weight=float(sample_weight),
                    member_name=member_name,
                )
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                samples += 1
        return {
            "samples": samples,
            "skipped_unsupported": skipped_unsupported,
            "skipped_malformed": skipped_malformed,
            "filtered_noisy": filtered_noisy,
        }


def load_agent_components():
    from agent.my_agent import ForgeNet, MyAgent

    return ForgeNet, MyAgent


class OfflineBehaviorCloner:
    """Minimal trainer that reuses my_agent's frame encoder contract."""

    def __init__(
        self,
        forge_net_cls,
        encoder_owner_cls,
        *,
        device: torch.device,
        lr: float,
        weight_decay: float,
        label_smoothing: float = 0.05,
    ) -> None:
        self.device = device
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.label_smoothing = max(0.0, float(label_smoothing))
        self._amp_enabled = (device.type == "cuda")
        self._grad_scaler = None
        if self._amp_enabled:
            try:
                self._grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.G = 64
        self.IN = 26
        row_pos = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        col_pos = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        self._pos_aug = torch.from_numpy(np.stack([row_pos, col_pos]))
        self._bg = 0
        self._tensor_last_frame_hash = None
        self._tensor_cached_static = None
        self._tensor_cached_full = None
        self._tensor_zero_tail_cache = {}
        self._aem_encoded_cache_sig = None
        self._aem_encoded_cache = None
        self._model_revision = 0

        self._fast_frame_hash = encoder_owner_cls._fast_frame_hash.__get__(self, type(self))
        self._normalized_palette_frame = encoder_owner_cls._normalized_palette_frame.__get__(self, type(self))
        self._encode_static_frame_cpu = encoder_owner_cls._encode_static_frame_cpu.__get__(self, type(self))
        self._tensor_zero_tail = encoder_owner_cls._tensor_zero_tail.__get__(self, type(self))
        self._encode_frame_tensor = encoder_owner_cls._encode_frame_tensor.__get__(self, type(self))

        self.net = forge_net_cls(self.IN, self.G).to(self.device)
        if self.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)
        self.opt = self._make_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=10000,
            eta_min=min(3e-5, self.lr),
        )

    def _make_optimizer(self):
        if self.device.type == "cuda":
            try:
                return optim.AdamW(
                    self.net.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    fused=True,
                )
            except (TypeError, RuntimeError):
                try:
                    return optim.AdamW(
                        self.net.parameters(),
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                        foreach=True,
                    )
                except TypeError:
                    pass
        return optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _amp_context(self):
        if self._amp_enabled:
            try:
                return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            except AttributeError:
                return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
        return torch.autocast(device_type="cpu", enabled=False)

    def load_weights(self, weights_path: Path) -> int:
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=self.device)
        model_state = self.net.state_dict()
        loaded_keys = 0
        for key, value in state.items():
            if key in model_state and value.shape == model_state[key].shape:
                model_state[key] = value
                loaded_keys += 1
        self.net.load_state_dict(model_state)
        return loaded_keys

    def save_weights(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), output_path)

    def train_batch(
        self,
        frames: list[np.ndarray],
        action_targets: list[int],
        *,
        sample_weights: list[float] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[float, int] | None:
        if not frames:
            return None
        rng = rng or random
        shift_dx, shift_dy, do_shift = _sample_batch_shift(action_targets, rng)

        tensors = []
        effective_targets = []
        effective_weights = []
        if sample_weights is None:
            input_weights = [1.0] * len(frames)
        else:
            input_weights = list(sample_weights)
        input_weights = _rebalance_sample_weights(action_targets, input_weights)
        for frame, action_target, sample_weight in zip(frames, action_targets, input_weights):
            frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
            shifted_target = int(action_target)
            if do_shift:
                bg_color = int(np.bincount(frame_c.ravel(), minlength=16).argmax())
                shifted_target = _shift_action_target(action_target, shift_dx, shift_dy)
                if shifted_target is None:
                    continue
                pad = np.pad(
                    frame_c,
                    ((1, 1), (1, 1)),
                    mode="constant",
                    constant_values=bg_color,
                )
                frame_c = pad[1 + shift_dy:65 + shift_dy, 1 + shift_dx:65 + shift_dx]
            tensors.append(self._encode_frame_tensor(frame_c).to(self.device, non_blocking=True))
            effective_targets.append(shifted_target)
            effective_weights.append(float(sample_weight))

        if not tensors:
            return None

        states = torch.stack(tensors)
        if self.device.type == "cuda":
            states = states.contiguous(memory_format=torch.channels_last)
        targets = torch.tensor(effective_targets, device=self.device, dtype=torch.long)
        weights = torch.tensor(effective_weights, device=self.device, dtype=torch.float32)

        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        with self._amp_context():
            logits = self.net(states)
            losses = F.cross_entropy(
                logits,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self._grad_scaler.step(self.opt)
            self._grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.opt.step()
        self.scheduler.step()
        self.net.eval()
        self._model_revision += 1
        self._aem_encoded_cache_sig = None
        self._aem_encoded_cache = None
        return float(loss.detach().item()), len(effective_targets)

    def evaluate_batch(
        self,
        frames: list[np.ndarray],
        action_targets: list[int],
        *,
        sample_weights: list[float] | None = None,
    ) -> dict[str, float | int] | None:
        if not frames:
            return None
        tensors = []
        effective_targets = []
        if sample_weights is None:
            input_weights = [1.0] * len(frames)
        else:
            input_weights = list(sample_weights)
        input_weights = _rebalance_sample_weights(action_targets, input_weights)
        effective_weights = []
        for frame, action_target, sample_weight in zip(frames, action_targets, input_weights):
            frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
            tensors.append(self._encode_frame_tensor(frame_c).to(self.device, non_blocking=True))
            effective_targets.append(int(action_target))
            effective_weights.append(float(sample_weight))
        if not tensors:
            return None
        states = torch.stack(tensors)
        if self.device.type == "cuda":
            states = states.contiguous(memory_format=torch.channels_last)
        targets = torch.tensor(effective_targets, device=self.device, dtype=torch.long)
        weights = torch.tensor(effective_weights, device=self.device, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(states)
            losses = F.cross_entropy(
                logits,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            weighted_loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            predictions = logits.argmax(dim=1)
            correct = int((predictions == targets).sum().item())
        return {
            "loss": float(weighted_loss.detach().item()),
            "samples": len(effective_targets),
            "correct": correct,
        }


def evaluate_from_zip(
    zip_path: Path,
    trainer: OfflineBehaviorCloner,
    *,
    member_names: list[str],
    batch_size: int,
    max_recordings: int | None = None,
) -> dict[str, float | int]:
    if not member_names:
        return {
            "samples": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "skipped_unsupported": 0,
            "skipped_malformed": 0,
            "filtered_noisy": 0,
        }
    batch_frames: list[np.ndarray] = []
    batch_actions: list[int] = []
    batch_weights: list[float] = []
    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    total_steps = 0
    skipped_unsupported = 0
    skipped_malformed = 0
    filtered_noisy = 0
    for event in iter_training_events(
        zip_path,
        member_names=member_names,
        max_recordings=max_recordings,
    ):
        if event[0] is None:
            _frame, action_index, _member_name = event
            if action_index == "unsupported":
                skipped_unsupported += 1
            elif action_index == "filtered_noisy":
                filtered_noisy += 1
            else:
                skipped_malformed += 1
            continue
        frame, action_index, sample_weight, _member_name = event
        batch_frames.append(frame)
        batch_actions.append(action_index)
        batch_weights.append(sample_weight)
        if len(batch_frames) >= batch_size:
            result = trainer.evaluate_batch(batch_frames, batch_actions, sample_weights=batch_weights)
            if result is not None:
                total_loss += float(result["loss"])
                total_samples += int(result["samples"])
                total_correct += int(result["correct"])
                total_steps += 1
            batch_frames = []
            batch_actions = []
            batch_weights = []
    if batch_frames:
        result = trainer.evaluate_batch(batch_frames, batch_actions, sample_weights=batch_weights)
        if result is not None:
            total_loss += float(result["loss"])
            total_samples += int(result["samples"])
            total_correct += int(result["correct"])
            total_steps += 1
    return {
        "samples": total_samples,
        "loss": (total_loss / total_steps) if total_steps else 0.0,
        "accuracy": (total_correct / total_samples) if total_samples else 0.0,
        "skipped_unsupported": skipped_unsupported,
        "skipped_malformed": skipped_malformed,
        "filtered_noisy": filtered_noisy,
    }


def train_from_zip(
    zip_path: Path,
    trainer: OfflineBehaviorCloner,
    *,
    epochs: int,
    batch_size: int,
    checkpoint_output_path: Path | None = None,
    validation_fraction: float = 0.0,
    max_recordings: int | None = None,
    max_samples: int | None = None,
    shuffle_members: bool = True,
    seed: int = 0,
    log_every: int = 512,
) -> dict[str, float | int]:
    if batch_size <= 0:
        raise SystemExit(f"batch_size must be positive, got {batch_size}")
    members = recording_members(zip_path)
    if not members:
        raise SystemExit(f"No .recording.jsonl members found in {zip_path}")
    train_members, validation_members = split_recording_members(
        members,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if max_samples is not None and max_samples <= 0:
        return {
            "samples": 0,
            "streamed_samples": 0,
            "steps": 0,
            "skipped_unsupported": 0,
            "skipped_malformed": 0,
            "filtered_noisy": 0,
            "dropped_after_augmentation": 0,
            "avg_loss": 0.0,
            "validation_samples": 0,
            "validation_loss": 0.0,
            "validation_accuracy": 0.0,
            "best_validation_epoch": 0,
            "best_validation_loss": 0.0,
            "best_validation_accuracy": 0.0,
        }

    rng = random.Random(seed)
    total_loss = 0.0
    total_steps = 0
    total_streamed_samples = 0
    total_trained_samples = 0
    total_skipped_unsupported = 0
    total_skipped_malformed = 0
    total_filtered_noisy = 0
    total_dropped_augmented = 0
    global_sample_cap_reached = False
    next_progress_log_at = log_every if log_every > 0 else None
    latest_validation_samples = 0
    latest_validation_loss = 0.0
    latest_validation_accuracy = 0.0
    best_validation_epoch = 0
    best_validation_loss = math.inf
    best_validation_accuracy = 0.0

    for epoch in range(epochs):
        if global_sample_cap_reached:
            break
        epoch_members = list(train_members)
        if shuffle_members:
            rng.shuffle(epoch_members)
        batch_frames: list[np.ndarray] = []
        batch_actions: list[int] = []
        batch_weights: list[float] = []
        epoch_streamed_samples = 0
        epoch_trained_samples = 0
        epoch_skipped_unsupported = 0
        epoch_skipped_malformed = 0
        epoch_filtered_noisy = 0
        epoch_dropped_augmented = 0
        for event in iter_training_events(
            zip_path,
            member_names=epoch_members,
            max_recordings=max_recordings,
        ):
            if event[0] is None:
                _frame, action_index, _member_name = event
                if action_index == "unsupported":
                    epoch_skipped_unsupported += 1
                    total_skipped_unsupported += 1
                elif action_index == "filtered_noisy":
                    epoch_filtered_noisy += 1
                    total_filtered_noisy += 1
                else:
                    epoch_skipped_malformed += 1
                    total_skipped_malformed += 1
                continue
            frame, action_index, sample_weight, _member_name = event
            batch_frames.append(frame)
            batch_actions.append(action_index)
            batch_weights.append(sample_weight)
            epoch_streamed_samples += 1
            total_streamed_samples += 1
            remaining_target_budget = None
            if max_samples is not None:
                remaining_target_budget = max(0, max_samples - total_trained_samples)
                if remaining_target_budget == 0:
                    global_sample_cap_reached = True
                    break
            effective_batch_size = batch_size
            if remaining_target_budget is not None:
                effective_batch_size = min(effective_batch_size, remaining_target_budget)
            if effective_batch_size > 0 and len(batch_frames) >= effective_batch_size:
                train_frames = batch_frames[:effective_batch_size]
                train_actions = batch_actions[:effective_batch_size]
                train_weights = batch_weights[:effective_batch_size]
                result = _call_train_batch(trainer, train_frames, train_actions, train_weights, rng)
                if result is not None:
                    loss, trained_count = result
                    total_loss += loss
                    total_steps += 1
                    total_trained_samples += trained_count
                    epoch_trained_samples += trained_count
                    dropped_count = len(train_frames) - trained_count
                    total_dropped_augmented += dropped_count
                    epoch_dropped_augmented += dropped_count
                else:
                    dropped_count = len(train_frames)
                    total_dropped_augmented += dropped_count
                    epoch_dropped_augmented += dropped_count
                if max_samples is not None and total_trained_samples >= max_samples:
                    global_sample_cap_reached = True
                if (
                    result is not None
                    and next_progress_log_at is not None
                    and total_streamed_samples >= next_progress_log_at
                ):
                    logger.info(
                        "epoch %s: streamed=%s trained=%s optimizer_steps=%s latest_loss=%.4f",
                        epoch + 1,
                        total_streamed_samples,
                        total_trained_samples,
                        total_steps,
                        loss,
                    )
                    while next_progress_log_at is not None and total_streamed_samples >= next_progress_log_at:
                        next_progress_log_at += log_every
                batch_frames = batch_frames[effective_batch_size:]
                batch_actions = batch_actions[effective_batch_size:]
                batch_weights = batch_weights[effective_batch_size:]
            if global_sample_cap_reached:
                break
        if batch_frames:
            if max_samples is not None:
                remaining_target_budget = max(0, max_samples - total_trained_samples)
                if remaining_target_budget <= 0:
                    batch_frames = []
                    batch_actions = []
                    batch_weights = []
                else:
                    batch_frames = batch_frames[:remaining_target_budget]
                    batch_actions = batch_actions[:remaining_target_budget]
                    batch_weights = batch_weights[:remaining_target_budget]
            if not batch_frames:
                logger.info(
                    "epoch %s complete: streamed=%s trained=%s skipped_unsupported=%s skipped_malformed=%s filtered_noisy=%s dropped_after_augmentation=%s cumulative_steps=%s",
                    epoch + 1,
                    epoch_streamed_samples,
                    epoch_trained_samples,
                    epoch_skipped_unsupported,
                    epoch_skipped_malformed,
                    epoch_filtered_noisy,
                    epoch_dropped_augmented,
                    total_steps,
                )
                continue
            result = _call_train_batch(trainer, batch_frames, batch_actions, batch_weights, rng)
            if result is not None:
                loss, trained_count = result
                total_loss += loss
                total_steps += 1
                total_trained_samples += trained_count
                epoch_trained_samples += trained_count
                dropped_count = len(batch_frames) - trained_count
                total_dropped_augmented += dropped_count
                epoch_dropped_augmented += dropped_count
            else:
                dropped_count = len(batch_frames)
                total_dropped_augmented += dropped_count
                epoch_dropped_augmented += dropped_count
            if max_samples is not None and total_trained_samples >= max_samples:
                global_sample_cap_reached = True
        logger.info(
            "epoch %s complete: streamed=%s trained=%s skipped_unsupported=%s skipped_malformed=%s filtered_noisy=%s dropped_after_augmentation=%s cumulative_steps=%s",
            epoch + 1,
            epoch_streamed_samples,
            epoch_trained_samples,
            epoch_skipped_unsupported,
            epoch_skipped_malformed,
            epoch_filtered_noisy,
            epoch_dropped_augmented,
            total_steps,
        )
        if validation_members:
            validation_stats = evaluate_from_zip(
                zip_path,
                trainer,
                member_names=validation_members,
                batch_size=batch_size,
                max_recordings=max_recordings,
            )
            latest_validation_samples = int(validation_stats["samples"])
            latest_validation_loss = float(validation_stats["loss"])
            latest_validation_accuracy = float(validation_stats["accuracy"])
            logger.info(
                "epoch %s validation: samples=%s loss=%.4f accuracy=%.4f skipped_unsupported=%s skipped_malformed=%s filtered_noisy=%s",
                epoch + 1,
                validation_stats["samples"],
                validation_stats["loss"],
                validation_stats["accuracy"],
                validation_stats["skipped_unsupported"],
                validation_stats["skipped_malformed"],
                validation_stats["filtered_noisy"],
            )
            if float(validation_stats["loss"]) < best_validation_loss:
                best_validation_epoch = epoch + 1
                best_validation_loss = float(validation_stats["loss"])
                best_validation_accuracy = float(validation_stats["accuracy"])
                if checkpoint_output_path is not None:
                    best_path = best_validation_checkpoint_path(checkpoint_output_path)
                    trainer.save_weights(best_path)
                    logger.info(
                        "saved best validation checkpoint at epoch %s to %s (loss=%.4f accuracy=%.4f)",
                        epoch + 1,
                        best_path,
                        best_validation_loss,
                        best_validation_accuracy,
                    )
        if checkpoint_output_path is not None:
            checkpoint_path = epoch_checkpoint_path(checkpoint_output_path, epoch + 1)
            trainer.save_weights(checkpoint_path)
            logger.info("saved epoch %s checkpoint to %s", epoch + 1, checkpoint_path)

    return {
        "samples": total_trained_samples,
        "streamed_samples": total_streamed_samples,
        "steps": total_steps,
        "skipped_unsupported": total_skipped_unsupported,
        "skipped_malformed": total_skipped_malformed,
        "filtered_noisy": total_filtered_noisy,
        "dropped_after_augmentation": total_dropped_augmented,
        "avg_loss": (total_loss / total_steps) if total_steps else 0.0,
        "validation_samples": latest_validation_samples,
        "validation_loss": latest_validation_loss,
        "validation_accuracy": latest_validation_accuracy,
        "best_validation_epoch": best_validation_epoch,
        "best_validation_loss": (best_validation_loss if best_validation_epoch else 0.0),
        "best_validation_accuracy": best_validation_accuracy,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=ROOT / "arc_agi_3_public_demo_human_testing.zip",
        help="Path to the recordings zip archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "hyperon_bootstrap.jsonl",
        help="Where to save the Hyperon bootstrap corpus or trained weights.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "symbolic", "neural"),
        default="auto",
        help="`symbolic` builds a Hyperon bootstrap corpus; `neural` runs the legacy behavior-cloning trainer; `auto` picks symbolic mode for non-weight outputs.",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional existing weights file to warm-start from.",
    )
    parser.add_argument(
        "--start-from-existing-weights",
        action="store_true",
        help="Warm-start from the current --output weights file.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    if not args.zip_path.exists():
        raise SystemExit(f"Zip archive not found: {args.zip_path}")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed % (2**32 - 1))

    selected_mode = args.mode
    if selected_mode == "auto":
        selected_mode = "neural" if args.output.suffix.lower() in {".pt", ".pth"} else "symbolic"

    if selected_mode == "symbolic":
        builder = HyperonCorpusBuilder()
        stats = builder.build_from_zip(
            args.zip_path,
            args.output,
            max_recordings=args.max_recordings,
            max_samples=args.max_samples,
        )
        logger.info(
            "saved Hyperon bootstrap corpus %s with %s samples, %s skipped unsupported actions, %s skipped malformed actions, %s filtered noisy actions",
            args.output,
            stats["samples"],
            stats["skipped_unsupported"],
            stats["skipped_malformed"],
            stats["filtered_noisy"],
        )
        return 0

    device = resolve_device(args.device)
    logger.info("device=%s", device)

    forge_net_cls, encoder_owner_cls = load_agent_components()
    trainer = OfflineBehaviorCloner(
        forge_net_cls,
        encoder_owner_cls,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )
    init_weights_path = resolve_init_weights_path(args)
    if init_weights_path is not None:
        if not init_weights_path.exists():
            raise SystemExit(f"Initial weights not found: {init_weights_path}")
        loaded_keys = trainer.load_weights(init_weights_path)
        logger.info("warm-started from %s (%s keys)", init_weights_path, loaded_keys)

    stats = train_from_zip(
        args.zip_path,
        trainer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_output_path=args.output,
        validation_fraction=args.validation_fraction,
        max_recordings=args.max_recordings,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    trainer.save_weights(args.output)
    logger.info(
        "saved %s after %s trained samples (%s streamed), %s skipped unsupported actions, %s skipped malformed actions, %s filtered noisy actions, %s dropped after augmentation, %s steps, avg_loss=%.4f, validation_samples=%s, validation_loss=%.4f, validation_accuracy=%.4f, best_validation_epoch=%s, best_validation_loss=%.4f, best_validation_accuracy=%.4f",
        args.output,
        stats["samples"],
        stats["streamed_samples"],
        stats["skipped_unsupported"],
        stats["skipped_malformed"],
        stats["filtered_noisy"],
        stats["dropped_after_augmentation"],
        stats["steps"],
        stats["avg_loss"],
        stats["validation_samples"],
        stats["validation_loss"],
        stats["validation_accuracy"],
        stats["best_validation_epoch"],
        stats["best_validation_loss"],
        stats["best_validation_accuracy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
