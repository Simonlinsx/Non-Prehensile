"""Train the DAPL physical world model from transition shards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from dapl.models import (
    DAPLFeatureNormalizer,
    DAPLSemanticPatchTokenizerConfig,
    DAPLWorldModel,
    DAPLWorldModelConfig,
    DAPLWorldModelLoss,
    DAPLWorldModelLossConfig,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-steps", type=int, default=500_000)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--weight-decay", type=float, default=1.0e-4)
parser.add_argument("--gradient-clip", type=float, default=1.0)
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--validation-interval", type=int, default=1_000)
parser.add_argument("--validation-batches", type=int, default=20)
parser.add_argument("--checkpoint-interval", type=int, default=10_000)
parser.add_argument("--log-interval", type=int, default=100)
parser.add_argument("--encoder-depth", type=int, default=12)
parser.add_argument("--token-dim", type=int, default=128)
parser.add_argument("--attention-heads", type=int, default=8)
parser.add_argument(
    "--resume",
    type=Path,
    help="Continue from a world-model checkpoint; --max-steps remains the total target.",
)
args = parser.parse_args()


def _validate_args() -> None:
    for name in (
        "batch_size",
        "max_steps",
        "validation_interval",
        "validation_batches",
        "checkpoint_interval",
        "log_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.learning_rate <= 0.0
        or args.weight_decay < 0.0
        or args.gradient_clip <= 0.0
    ):
        raise ValueError("optimizer settings are invalid")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.encoder_depth <= 0 or args.token_dim <= 0 or args.attention_heads <= 0:
        raise ValueError("model dimensions must be positive")
    if args.token_dim % args.attention_heads != 0:
        raise ValueError("--token-dim must be divisible by --attention-heads")
    if args.resume is not None and not args.resume.expanduser().is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")


def _load_transitions(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported transition schema in {path}")
    if abs(float(payload.get("control_dt_s", -1.0)) - 0.1) > 1.0e-9:
        raise ValueError(f"transition shard {path} does not use 0.1 s control steps")
    transitions = payload.get("transitions")
    required = {"scene_t", "scene_tp1", "end_effector_flow"}
    if not isinstance(transitions, dict) or not required.issubset(transitions):
        missing = required.difference(transitions or {})
        raise ValueError(f"transition shard {path} is missing {sorted(missing)}")
    count = transitions["scene_t"].shape[0]
    if transitions["scene_t"].shape != (count, 1280, 7):
        raise ValueError(f"invalid scene_t shape in {path}")
    if transitions["scene_tp1"].shape != (count, 1280, 7):
        raise ValueError(f"invalid scene_tp1 shape in {path}")
    if transitions["end_effector_flow"].shape != (count, 3):
        raise ValueError(f"invalid end_effector_flow shape in {path}")
    return transitions


def _split_shards(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(args.seed).shuffle(shuffled)
    if len(shuffled) == 1 or args.validation_fraction == 0.0:
        return shuffled, shuffled
    validation_count = max(1, round(len(shuffled) * args.validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def _fit_normalizer(paths: list[Path]) -> DAPLFeatureNormalizer:
    normalizer = DAPLFeatureNormalizer()
    for path in paths:
        transitions = _load_transitions(path)
        normalizer.update(transitions["scene_t"])
        normalizer.update(transitions["scene_tp1"])
    return normalizer


def _normalized_batch(
    transitions: dict[str, torch.Tensor],
    indices: torch.Tensor,
    normalizer: DAPLFeatureNormalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene_t = transitions["scene_t"][indices].to(device, non_blocking=True)
    scene_tp1 = transitions["scene_tp1"][indices].to(device, non_blocking=True)
    flow = transitions["end_effector_flow"][indices].to(device, non_blocking=True)
    scene_t = normalizer.normalize(scene_t)
    scene_tp1 = normalizer.normalize(scene_tp1)
    # Flow is a displacement, so apply the coordinate scale without subtracting
    # the absolute position mean.
    position_scale = torch.sqrt(normalizer.variance[:3] + normalizer.epsilon)
    return scene_t, scene_tp1, flow / position_scale


@torch.no_grad()
def _evaluate(
    model: DAPLWorldModel,
    objective: DAPLWorldModelLoss,
    normalizer: DAPLFeatureNormalizer,
    paths: list[Path],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "position": 0.0, "velocity": 0.0, "variance": 0.0}
    batches = 0
    for path in paths:
        transitions = _load_transitions(path)
        for start in range(0, transitions["scene_t"].shape[0], args.batch_size):
            indices = torch.arange(
                start,
                min(start + args.batch_size, transitions["scene_t"].shape[0]),
            )
            scene_t, scene_tp1, flow = _normalized_batch(
                transitions, indices, normalizer, device
            )
            losses = objective(model(scene_t, flow), scene_tp1)
            for key in totals:
                totals[key] += float(getattr(losses, key).item())
            batches += 1
            if batches >= args.validation_batches:
                break
        if batches >= args.validation_batches:
            break
    model.train()
    if batches == 0:
        raise RuntimeError("validation split contains no transitions")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def _evaluate_persistence(
    normalizer: DAPLFeatureNormalizer,
    loss_config: DAPLWorldModelLossConfig,
    paths: list[Path],
    device: torch.device,
) -> dict[str, float]:
    """Score the no-change prediction on the exact validation batches."""

    totals = {"total": 0.0, "position": 0.0, "velocity": 0.0, "variance": 0.0}
    batches = 0
    for path in paths:
        transitions = _load_transitions(path)
        for start in range(0, transitions["scene_t"].shape[0], args.batch_size):
            indices = torch.arange(
                start,
                min(start + args.batch_size, transitions["scene_t"].shape[0]),
            )
            scene_t, scene_tp1, _ = _normalized_batch(
                transitions, indices, normalizer, device
            )
            position = torch.mean((scene_t[..., :3] - scene_tp1[..., :3]).square())
            velocity = torch.mean((scene_t[..., 4:7] - scene_tp1[..., 4:7]).square())
            current_variance = scene_t[..., 4:7].reshape(-1, 3).var(
                dim=0, unbiased=False
            )
            future_variance = scene_tp1[..., 4:7].reshape(-1, 3).var(
                dim=0, unbiased=False
            )
            variance = torch.mean((current_variance - future_variance).square())
            total = (
                loss_config.position_weight * position
                + loss_config.velocity_weight * velocity
                + loss_config.variance_weight * variance
            )
            for key, value in (
                ("total", total),
                ("position", position),
                ("velocity", velocity),
                ("variance", variance),
            ):
                totals[key] += float(value.item())
            batches += 1
            if batches >= args.validation_batches:
                break
        if batches >= args.validation_batches:
            break
    if batches == 0:
        raise RuntimeError("validation split contains no transitions")
    return {key: value / batches for key, value in totals.items()}


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: DAPLWorldModel,
    normalizer: DAPLFeatureNormalizer,
    optimizer: torch.optim.Optimizer,
    model_config: DAPLWorldModelConfig,
    loss_config: DAPLWorldModelLossConfig,
    train_shards: list[Path],
    validation_shards: list[Path],
    filename: str | None = None,
) -> Path:
    path = output_dir / (
        f"world_model_step_{step:07d}.pt" if filename is None else filename
    )
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_shards": [str(item) for item in train_shards],
            "validation_shards": [str(item) for item in validation_shards],
        },
        temporary,
    )
    temporary.replace(path)
    return path


def main() -> None:
    _validate_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    paths = sorted(data_dir.glob("transitions_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no transition shards found in {data_dir}")
    train_shards, validation_shards = _split_shards(paths)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("world_model_step_*.pt")) or (
        output_dir / "world_model_best.pt"
    ).exists():
        raise FileExistsError(
            f"refusing to overwrite an existing world-model run in {output_dir}"
        )

    normalizer = _fit_normalizer(train_shards).to(device)
    model_config = DAPLWorldModelConfig(
        tokenizer=DAPLSemanticPatchTokenizerConfig(token_dim=args.token_dim),
        encoder_depth=args.encoder_depth,
        attention_heads=args.attention_heads,
    )
    loss_config = DAPLWorldModelLossConfig()
    model = DAPLWorldModel(model_config).to(device)
    objective = DAPLWorldModelLoss(loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    step = 0
    resumed_from: str | None = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        checkpoint_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if checkpoint_payload.get("schema_version") != 1:
            raise ValueError(f"unsupported world-model checkpoint: {resume_path}")
        if checkpoint_payload.get("model_config") != asdict(model_config):
            raise ValueError("resume checkpoint model configuration does not match CLI")
        if checkpoint_payload.get("loss_config") != asdict(loss_config):
            raise ValueError("resume checkpoint loss configuration does not match CLI")
        expected_train = [str(item) for item in train_shards]
        expected_validation = [str(item) for item in validation_shards]
        if checkpoint_payload.get("train_shards") != expected_train or checkpoint_payload.get(
            "validation_shards"
        ) != expected_validation:
            raise ValueError("resume checkpoint uses a different train/validation split")
        model.load_state_dict(checkpoint_payload["model"], strict=True)
        normalizer.load_state_dict(checkpoint_payload["normalizer"], strict=True)
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        step = int(checkpoint_payload["step"])
        if step >= args.max_steps:
            raise ValueError(
                f"resume step {step} must be lower than --max-steps {args.max_steps}"
            )
        resumed_from = str(resume_path)

    writer = SummaryWriter(output_dir)
    print(
        "DAPL_WORLD_MODEL_TRAIN_START",
        f"device={device}",
        f"train_shards={len(train_shards)}",
        f"validation_shards={len(validation_shards)}",
        f"start_step={step}",
        f"max_steps={args.max_steps}",
        f"resumed_from={resumed_from}",
        flush=True,
    )

    epoch = 0
    started = time.time()
    last_validation = _evaluate(
        model, objective, normalizer, validation_shards, device
    )
    persistence_baseline = _evaluate_persistence(
        normalizer, loss_config, validation_shards, device
    )
    for key, value in last_validation.items():
        writer.add_scalar(f"validation/{key}", value, step)
    for key, value in persistence_baseline.items():
        writer.add_scalar(f"baseline/persistence_{key}", value, step)
    print(
        "DAPL_WORLD_MODEL_VALIDATION",
        f"step={step}",
        *(f"{key}={value:.6f}" for key, value in last_validation.items()),
        flush=True,
    )
    print(
        "DAPL_WORLD_MODEL_PERSISTENCE_BASELINE",
        *(f"{key}={value:.6f}" for key, value in persistence_baseline.items()),
        flush=True,
    )
    best_validation = dict(last_validation)
    best_validation_step = step
    best_checkpoint = _save_checkpoint(
        output_dir,
        step,
        model,
        normalizer,
        optimizer,
        model_config,
        loss_config,
        train_shards,
        validation_shards,
        filename="world_model_best.pt",
    )
    print(
        "DAPL_WORLD_MODEL_BEST_CHECKPOINT",
        f"step={step}",
        f"total={best_validation['total']:.6f}",
        f"path={best_checkpoint}",
        flush=True,
    )
    model.train()
    while step < args.max_steps:
        shard_order = list(train_shards)
        random.Random(args.seed + epoch).shuffle(shard_order)
        for path in shard_order:
            transitions = _load_transitions(path)
            permutation = torch.randperm(
                transitions["scene_t"].shape[0],
                generator=torch.Generator().manual_seed(args.seed + epoch + step),
            )
            for start in range(0, len(permutation), args.batch_size):
                indices = permutation[start : start + args.batch_size]
                scene_t, scene_tp1, flow = _normalized_batch(
                    transitions, indices, normalizer, device
                )
                prediction = model(scene_t, flow)
                losses = objective(prediction, scene_tp1)
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                optimizer.step()
                step += 1

                writer.add_scalar("train/total", losses.total.item(), step)
                writer.add_scalar("train/position", losses.position.item(), step)
                writer.add_scalar("train/velocity", losses.velocity.item(), step)
                writer.add_scalar("train/variance", losses.variance.item(), step)
                writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
                if step == 1 or step % args.log_interval == 0:
                    print(
                        "DAPL_WORLD_MODEL_STEP",
                        f"step={step}",
                        f"total={losses.total.item():.6f}",
                        f"position={losses.position.item():.6f}",
                        f"velocity={losses.velocity.item():.6f}",
                        f"variance={losses.variance.item():.6f}",
                        flush=True,
                    )
                if step % args.validation_interval == 0 or step == args.max_steps:
                    last_validation = _evaluate(
                        model, objective, normalizer, validation_shards, device
                    )
                    for key, value in last_validation.items():
                        writer.add_scalar(f"validation/{key}", value, step)
                    print(
                        "DAPL_WORLD_MODEL_VALIDATION",
                        f"step={step}",
                        *(f"{key}={value:.6f}" for key, value in last_validation.items()),
                        flush=True,
                    )
                    if last_validation["total"] < best_validation["total"]:
                        best_validation = dict(last_validation)
                        best_validation_step = step
                        best_checkpoint = _save_checkpoint(
                            output_dir,
                            step,
                            model,
                            normalizer,
                            optimizer,
                            model_config,
                            loss_config,
                            train_shards,
                            validation_shards,
                            filename="world_model_best.pt",
                        )
                        print(
                            "DAPL_WORLD_MODEL_BEST_CHECKPOINT",
                            f"step={step}",
                            f"total={best_validation['total']:.6f}",
                            f"path={best_checkpoint}",
                            flush=True,
                        )
                if step % args.checkpoint_interval == 0 or step == args.max_steps:
                    checkpoint = _save_checkpoint(
                        output_dir,
                        step,
                        model,
                        normalizer,
                        optimizer,
                        model_config,
                        loss_config,
                        train_shards,
                        validation_shards,
                    )
                    print("DAPL_WORLD_MODEL_CHECKPOINT", f"path={checkpoint}", flush=True)
                if step >= args.max_steps:
                    break
            if step >= args.max_steps:
                break
        epoch += 1

    elapsed = time.time() - started
    summary = {
        "schema_version": 1,
        "steps": step,
        "elapsed_seconds": elapsed,
        "train_shards": len(train_shards),
        "validation_shards": len(validation_shards),
        "validation": last_validation,
        "persistence_baseline": persistence_baseline,
        "best_validation": best_validation,
        "best_validation_step": best_validation_step,
        "best_checkpoint": str(best_checkpoint),
        "resumed_from": resumed_from,
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    writer.close()
    print(
        "DAPL_WORLD_MODEL_TRAIN_OK",
        f"steps={step}",
        f"elapsed_seconds={elapsed:.3f}",
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
