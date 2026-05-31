# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for :mod:`cosmos_framework.scripts.eval_utils` aggregation and matching helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from cosmos_framework.scripts.eval_utils import (
    aggregate_metrics,
    compute_video_metrics,
    derive_match_key_and_group,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


# ---------------------------------------------------------------------------
# aggregate_metrics — mean / count for every scalar metric
# ---------------------------------------------------------------------------


def _write_metrics(tmp_path: Path, name: str, mode: str, values: dict) -> None:
    d = tmp_path / mode / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"mode": mode, "name": name, **values}))


def test_aggregate_metrics_empty_dir_returns_empty(tmp_path):
    assert aggregate_metrics(tmp_path) == {}


def test_aggregate_metrics_single_sample(tmp_path):
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    out = aggregate_metrics(tmp_path)
    assert out == {"vision": {"psnr": {"mean": 20.0, "count": 1}}}


def test_aggregate_metrics_mean_correct(tmp_path):
    for i, v in enumerate([10.0, 20.0, 30.0]):
        _write_metrics(tmp_path, f"s{i}", "vision", {"psnr": v})
    out = aggregate_metrics(tmp_path)["vision"]["psnr"]
    assert out["count"] == 3
    assert math.isclose(out["mean"], 20.0)


def test_aggregate_metrics_skips_files_without_mode(tmp_path):
    d = tmp_path / "orphan"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"name": "x", "psnr": 99.0}))
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    out = aggregate_metrics(tmp_path)
    assert set(out.keys()) == {"vision"}
    assert out["vision"]["psnr"]["count"] == 1


def test_aggregate_metrics_separates_modes(tmp_path):
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    _write_metrics(tmp_path, "s1", "forward_dynamics", {"psnr": 24.0})
    out = aggregate_metrics(tmp_path)
    assert set(out.keys()) == {"vision", "forward_dynamics"}
    assert out["vision"]["psnr"]["mean"] == 20.0
    assert out["forward_dynamics"]["psnr"]["mean"] == 24.0


def test_aggregate_metrics_flattens_nested_dicts(tmp_path):
    # nested dicts (e.g. grouped action_mse) flatten to "k/sub_k"
    _write_metrics(tmp_path, "s0", "policy", {"group_mse": {"arm": 0.1, "gripper": 0.2}})
    _write_metrics(tmp_path, "s1", "policy", {"group_mse": {"arm": 0.3, "gripper": 0.4}})
    out = aggregate_metrics(tmp_path)["policy"]
    assert "group_mse/arm" in out and "group_mse/gripper" in out
    assert math.isclose(out["group_mse/arm"]["mean"], 0.2)
    assert math.isclose(out["group_mse/gripper"]["mean"], 0.3)


# ---------------------------------------------------------------------------
# compute_video_metrics — vision lenient T-trim vs strict (action) on mismatch
# ---------------------------------------------------------------------------


def _write_synthetic_mp4(path: Path, frames_cthw_uint8: torch.Tensor, fps: int = 5) -> None:
    """Write a (C, T, H, W) uint8 tensor as an mp4 via torchvision.

    Lossy encoding will shift pixel values slightly; tests assert structural
    properties (shape, presence of metrics) rather than exact PSNR.
    """
    import torchvision.io as tvio

    # write_video expects (T, H, W, C) uint8
    thwc = frames_cthw_uint8.permute(1, 2, 3, 0).contiguous()
    tvio.write_video(str(path), thwc, fps=fps)


def test_compute_video_metrics_vision_lenient_trims_to_min_t(tmp_path, caplog):
    """VFM mode: pred has fewer frames than GT → trim both to min(T), warn, return metrics."""
    g = torch.Generator().manual_seed(0)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with caplog.at_level("WARNING"):
        metrics = compute_video_metrics(gt, pred_path, mode="vision")

    assert "psnr" in metrics


def test_compute_video_metrics_vision_no_warning_on_matching_shapes(tmp_path, caplog):
    """VFM mode: matching shapes → no warning, full metrics."""
    g = torch.Generator().manual_seed(1)
    gt = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with caplog.at_level("WARNING"):
        metrics = compute_video_metrics(gt, pred_path, mode="vision")

    assert "psnr" in metrics
    assert "trimmed to" not in caplog.text.lower()


def test_compute_video_metrics_action_strict_on_t_mismatch(tmp_path):
    """forward_dynamics: T mismatch still raises ValueError (the action chunk size is fixed)."""
    g = torch.Generator().manual_seed(2)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_video_metrics(gt, pred_path, mode="forward_dynamics")


def test_compute_video_metrics_vision_spatial_mismatch_still_errors(tmp_path):
    """Even in vision mode, an H mismatch that survives the top-left crop is a hard error."""
    g = torch.Generator().manual_seed(3)
    gt = torch.randint(0, 256, (3, 5, 16, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 8, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="spatial mismatch"):
        compute_video_metrics(gt, pred_path, mode="vision")


# ---------------------------------------------------------------------------
# derive_match_key_and_group — generic path-structure-based pairing rule
# ---------------------------------------------------------------------------


def test_derive_match_key_and_group_user_tree_cosmos_nano(tmp_path):
    """Tree 1: <root>/cosmos_nano_t2w/episode_*/vision.mp4 → key=episode_*, group=cosmos_nano_t2w."""
    p = tmp_path / "cosmos_nano_t2w" / "episode_002345_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_002345_clip000"
    assert group == "cosmos_nano_t2w"


def test_derive_match_key_and_group_user_tree_mixed_modality(tmp_path):
    """Tree 2: <root>/mixed_modality_*/t2v/episode_*/vision.mp4 → group=mixed_modality_*/t2v."""
    p = tmp_path / "mixed_modality_sft_8b_0507e" / "t2v" / "episode_002345_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_002345_clip000"
    assert group == "mixed_modality_sft_8b_0507e/t2v"


def test_derive_match_key_and_group_flat_layout(tmp_path):
    """Flat: <root>/<key>/vision.mp4 → key=<key>, group empty string."""
    p = tmp_path / "clip0" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "clip0"
    assert group == ""


def test_derive_match_key_and_group_inference_py_output(tmp_path):
    """Canonical inference.py output: <output_dir>/<sample.name>/vision.mp4."""
    p = tmp_path / "t2v" / "episode_049683_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_049683_clip000"
    assert group == "t2v"


def test_derive_match_key_and_group_non_vision_filename_uses_stem(tmp_path):
    """If basename isn't vision.*, the filename stem becomes the key (no parent-dir drop)."""
    p = tmp_path / "sub" / "foo.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "foo"
    assert group == "sub"


def test_derive_match_key_and_group_rejects_path_outside_predictions_dir(tmp_path):
    other = tmp_path.parent / "elsewhere" / "vision.mp4"
    with pytest.raises(ValueError, match="not under predictions_dir"):
        derive_match_key_and_group(other, tmp_path)
