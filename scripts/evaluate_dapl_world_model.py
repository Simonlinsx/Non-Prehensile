"""Evaluate a DAPL world-model checkpoint on its complete held-out split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from dapl.models import (
    DAPLFeatureNormalizer,
    DAPLSemanticPatchTokenizerConfig,
    DAPLWorldModel,
    DAPLWorldModelConfig,
    DAPLWorldModelLoss,
    DAPLWorldModelLossConfig,
)
from dapl.representation import DAPLSceneTensorConfig


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument(
    "--max-batches",
    type=int,
    help="Optional smoke-test limit; omit it for the strict complete-split report.",
)
args = parser.parse_args()


COMPONENT_SLICES = {
    "target": slice(0, 512),
    "obstacle": slice(512, 1024),
    "end_effector": slice(1024, 1280),
}
LOSS_KEYS = ("total", "position", "velocity", "variance")


def _model_config(payload: dict) -> DAPLWorldModelConfig:
    values = dict(payload)
    tokenizer_values = dict(values.pop("tokenizer"))
    scene = DAPLSceneTensorConfig(**tokenizer_values.pop("scene"))
    tokenizer = DAPLSemanticPatchTokenizerConfig(scene=scene, **tokenizer_values)
    return DAPLWorldModelConfig(tokenizer=tokenizer, **values)


def _load_transitions(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported transition schema in {path}")
    if abs(float(payload.get("control_dt_s", -1.0)) - 0.1) > 1.0e-9:
        raise ValueError(f"transition shard {path} does not use 0.1 s control steps")
    transitions = payload.get("transitions")
    required = {"scene_t", "scene_tp1", "end_effector_flow"}
    if not isinstance(transitions, dict) or not required.issubset(transitions):
        raise ValueError(f"transition shard {path} is missing required tensors")
    count = transitions["scene_t"].shape[0]
    if transitions["scene_t"].shape != (count, 1280, 7):
        raise ValueError(f"invalid scene_t shape in {path}")
    if transitions["scene_tp1"].shape != (count, 1280, 7):
        raise ValueError(f"invalid scene_tp1 shape in {path}")
    if transitions["end_effector_flow"].shape != (count, 3):
        raise ValueError(f"invalid end_effector_flow shape in {path}")
    return transitions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_loss(total: dict[str, float], losses) -> None:
    for key in LOSS_KEYS:
        total[key] += float(getattr(losses, key).item())


def _add_physical_errors(
    totals: dict[str, dict[str, float]],
    name: str,
    position: torch.Tensor,
    velocity: torch.Tensor,
    future: torch.Tensor,
    position_scale: torch.Tensor,
    velocity_scale: torch.Tensor,
) -> None:
    for component, point_slice in COMPONENT_SLICES.items():
        position_error = (
            position[:, point_slice] - future[:, point_slice, :3]
        ) * position_scale
        velocity_error = (
            velocity[:, point_slice] - future[:, point_slice, 4:7]
        ) * velocity_scale
        entry = totals[name][component]
        entry["position_squared_error_m2"] += float(position_error.square().sum().item())
        entry["position_values"] += position_error.numel()
        entry["velocity_squared_error_m2_s2"] += float(velocity_error.square().sum().item())
        entry["velocity_values"] += velocity_error.numel()


def _finalize_physical(totals: dict[str, dict[str, float]]) -> dict:
    result = {}
    for method, components in totals.items():
        result[method] = {}
        for component, values in components.items():
            result[method][component] = {
                "position_rmse_m": (
                    values["position_squared_error_m2"] / values["position_values"]
                )
                ** 0.5,
                "velocity_rmse_m_s": (
                    values["velocity_squared_error_m2_s2"] / values["velocity_values"]
                )
                ** 0.5,
            }
    return result


def main() -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"unsupported world-model checkpoint: {checkpoint_path}")

    model_config = _model_config(checkpoint["model_config"])
    loss_config = DAPLWorldModelLossConfig(**checkpoint["loss_config"])
    if asdict(model_config) != checkpoint["model_config"]:
        raise ValueError("checkpoint model configuration did not round-trip")
    device = torch.device(args.device)
    model = DAPLWorldModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    normalizer = DAPLFeatureNormalizer().to(device)
    normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
    objective = DAPLWorldModelLoss(loss_config)

    validation_paths = [Path(value) for value in checkpoint["validation_shards"]]
    missing = [str(path) for path in validation_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validation shards no longer exist: {missing}")

    losses = {
        method: {key: 0.0 for key in LOSS_KEYS}
        for method in ("model", "zero_action", "persistence")
    }
    physical_template = {
        component: {
            "position_squared_error_m2": 0.0,
            "position_values": 0,
            "velocity_squared_error_m2_s2": 0.0,
            "velocity_values": 0,
        }
        for component in COMPONENT_SLICES
    }
    physical = {
        method: {
            component: dict(values) for component, values in physical_template.items()
        }
        for method in losses
    }
    action_delta_position_squared_error_m2 = 0.0
    action_delta_velocity_squared_error_m2_s2 = 0.0
    action_delta_values = 0
    position_scale = torch.sqrt(normalizer.variance[:3] + normalizer.epsilon)
    velocity_scale = torch.sqrt(normalizer.variance[4:7] + normalizer.epsilon)
    batches = 0
    transitions_seen = 0

    with torch.inference_mode():
        for path in validation_paths:
            transitions = _load_transitions(path)
            count = transitions["scene_t"].shape[0]
            for start in range(0, count, args.batch_size):
                indices = slice(start, min(start + args.batch_size, count))
                scene_t = transitions["scene_t"][indices].to(device)
                scene_tp1 = transitions["scene_tp1"][indices].to(device)
                flow = transitions["end_effector_flow"][indices].to(device)
                scene_t = normalizer.normalize(scene_t)
                scene_tp1 = normalizer.normalize(scene_tp1)
                normalized_flow = flow / position_scale

                prediction = model(scene_t, normalized_flow)
                zero_prediction = model(scene_t, torch.zeros_like(normalized_flow))
                persistence_prediction = type(prediction)(
                    position=scene_t[..., :3],
                    velocity=scene_t[..., 4:7],
                    dynamics_tokens=prediction.dynamics_tokens,
                    point_features=prediction.point_features,
                    tokenization=prediction.tokenization,
                )
                _add_loss(losses["model"], objective(prediction, scene_tp1))
                _add_loss(losses["zero_action"], objective(zero_prediction, scene_tp1))
                _add_loss(
                    losses["persistence"], objective(persistence_prediction, scene_tp1)
                )
                for name, current in (
                    ("model", prediction),
                    ("zero_action", zero_prediction),
                    ("persistence", persistence_prediction),
                ):
                    _add_physical_errors(
                        physical,
                        name,
                        current.position,
                        current.velocity,
                        scene_tp1,
                        position_scale,
                        velocity_scale,
                    )

                position_delta = (
                    prediction.position - zero_prediction.position
                ) * position_scale
                velocity_delta = (
                    prediction.velocity - zero_prediction.velocity
                ) * velocity_scale
                action_delta_position_squared_error_m2 += float(
                    position_delta.square().sum().item()
                )
                action_delta_velocity_squared_error_m2_s2 += float(
                    velocity_delta.square().sum().item()
                )
                action_delta_values += position_delta.numel()
                batches += 1
                transitions_seen += scene_t.shape[0]
                if args.max_batches is not None and batches >= args.max_batches:
                    break
            if args.max_batches is not None and batches >= args.max_batches:
                break

    if batches == 0:
        raise RuntimeError("validation split contains no transitions")
    averaged_losses = {
        method: {key: value / batches for key, value in values.items()}
        for method, values in losses.items()
    }
    model_total = averaged_losses["model"]["total"]
    persistence_total = averaged_losses["persistence"]["total"]
    zero_total = averaged_losses["zero_action"]["total"]
    report = {
        "schema_version": 1,
        "strict_complete_split": args.max_batches is None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "device": str(device),
        "batch_size": args.batch_size,
        "validation_shards": len(validation_paths),
        "validation_batches": batches,
        "validation_transitions": transitions_seen,
        "normalized_losses": averaged_losses,
        "physical_rmse": _finalize_physical(physical),
        "relative_total_improvement_percent": {
            "model_vs_persistence": 100.0 * (persistence_total - model_total) / persistence_total,
            "model_vs_zero_action": 100.0 * (zero_total - model_total) / zero_total,
        },
        "action_conditioning_prediction_delta": {
            "position_rmse_m": (
                action_delta_position_squared_error_m2 / action_delta_values
            )
            ** 0.5,
            "velocity_rmse_m_s": (
                action_delta_velocity_squared_error_m2_s2 / action_delta_values
            )
            ** 0.5,
        },
        "model_config": checkpoint["model_config"],
        "loss_config": checkpoint["loss_config"],
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_evaluation.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output_path)
    print(
        "DAPL_WORLD_MODEL_EVALUATION_OK",
        f"step={checkpoint['step']}",
        f"transitions={transitions_seen}",
        f"model_total={model_total:.6f}",
        f"persistence_total={persistence_total:.6f}",
        f"zero_action_total={zero_total:.6f}",
        f"report={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
