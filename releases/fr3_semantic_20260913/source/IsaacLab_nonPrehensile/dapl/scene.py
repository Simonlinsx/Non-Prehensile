"""Versioned JSONL schema for generated Clutter6D scenes and tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCENE_SCHEMA_VERSION = 1


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _pose(value: Sequence[float], name: str) -> tuple[float, ...]:
    result = _finite_tuple(value, 7, name)
    quat_norm = math.sqrt(sum(component * component for component in result[3:]))
    if not math.isclose(quat_norm, 1.0, abs_tol=1.0e-3):
        raise ValueError(f"{name} quaternion must be normalized, got norm {quat_norm:.6f}")
    return result


class ClutterTrack(str, Enum):
    """Clutter6D density tracks and their paper-defined object counts."""

    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"

    @property
    def object_count(self) -> int:
        return {
            ClutterTrack.SPARSE: 4,
            ClutterTrack.MODERATE: 8,
            ClutterTrack.DENSE: 12,
        }[self]

    @property
    def large_obstacle_count(self) -> int:
        """Number of paper-defined large non-target objects."""

        return {
            ClutterTrack.SPARSE: 1,
            ClutterTrack.MODERATE: 3,
            ClutterTrack.DENSE: 5,
        }[self]

    @property
    def small_obstacle_count(self) -> int:
        """Number of paper-defined small non-target objects."""

        return {
            ClutterTrack.SPARSE: 2,
            ClutterTrack.MODERATE: 4,
            ClutterTrack.DENSE: 6,
        }[self]


@dataclass(frozen=True)
class SceneObject:
    """One rigid object in a generated clutter arrangement.

    Poses use Isaac Lab's ``[x, y, z, qw, qx, qy, qz]`` convention in the
    environment frame.  The pose is the scene's base pose; a task may replace
    the target object's pose with its own initial pose.
    """

    instance_id: str
    asset_id: str
    pose: tuple[float, ...]
    scale: tuple[float, ...]
    mass_kg: float
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        object.__setattr__(self, "pose", _pose(self.pose, "pose"))
        scale = _finite_tuple(self.scale, 3, "scale")
        if any(component <= 0.0 for component in scale):
            raise ValueError("scale components must be positive")
        object.__setattr__(self, "scale", scale)
        for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.mass_kg == 0.0:
            raise ValueError("mass_kg must be greater than zero")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic_friction must not exceed static_friction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "asset_id": self.asset_id,
            "pose": list(self.pose),
            "scale": list(self.scale),
            "mass_kg": self.mass_kg,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneObject":
        return cls(**value)


@dataclass(frozen=True)
class ManipulationTask:
    """One target initial/goal pose pair within a clutter scene."""

    task_id: str
    target_instance_id: str
    initial_pose: tuple[float, ...]
    goal_pose: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.target_instance_id:
            raise ValueError("task_id and target_instance_id must be non-empty")
        object.__setattr__(self, "initial_pose", _pose(self.initial_pose, "initial_pose"))
        object.__setattr__(self, "goal_pose", _pose(self.goal_pose, "goal_pose"))

    @property
    def planar_displacement(self) -> float:
        dx = self.goal_pose[0] - self.initial_pose[0]
        dy = self.goal_pose[1] - self.initial_pose[1]
        return math.hypot(dx, dy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_instance_id": self.target_instance_id,
            "initial_pose": list(self.initial_pose),
            "goal_pose": list(self.goal_pose),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManipulationTask":
        return cls(**value)


@dataclass(frozen=True)
class ClutterScene:
    """A reproducible multi-object arrangement and its manipulation tasks."""

    scene_id: str
    split: str
    track: ClutterTrack
    objects: tuple[SceneObject, ...]
    tasks: tuple[ManipulationTask, ...]
    schema_version: int = SCENE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        if self.split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        object.__setattr__(self, "track", ClutterTrack(self.track))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if self.schema_version != SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scene schema {self.schema_version}; expected {SCENE_SCHEMA_VERSION}"
            )
        if len(self.objects) != self.track.object_count:
            raise ValueError(
                f"{self.track.value} scenes require {self.track.object_count} objects, "
                f"got {len(self.objects)}"
            )
        instance_ids = [item.instance_id for item in self.objects]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("scene object instance_id values must be unique")
        task_ids = [item.task_id for item in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique within a scene")
        unknown_targets = {
            task.target_instance_id for task in self.tasks if task.target_instance_id not in instance_ids
        }
        if unknown_targets:
            raise ValueError(f"tasks reference unknown target instances: {sorted(unknown_targets)}")
        target_ids = {task.target_instance_id for task in self.tasks}
        if len(target_ids) > 1:
            raise ValueError("all tasks in one scene must manipulate the same target instance")

    @property
    def target_instance_id(self) -> str:
        """Return the single target instance used by all tasks."""

        if not self.tasks:
            raise ValueError(f"scene {self.scene_id!r} has no manipulation tasks")
        return self.tasks[0].target_instance_id

    @property
    def target_object(self) -> SceneObject:
        """Return the object selected as the scene target."""

        target_id = self.target_instance_id
        return next(item for item in self.objects if item.instance_id == target_id)

    @property
    def obstacle_objects(self) -> tuple[SceneObject, ...]:
        """Return non-target objects in manifest order."""

        target_id = self.target_instance_id
        return tuple(item for item in self.objects if item.instance_id != target_id)

    def validate_paper_contract(
        self,
        *,
        tasks_per_scene: int = 16,
        minimum_planar_displacement: float = 0.15,
    ) -> None:
        """Validate the task-generation constraints reported in the paper."""

        if len(self.tasks) != tasks_per_scene:
            raise ValueError(f"expected {tasks_per_scene} tasks, got {len(self.tasks)}")
        if len({task.target_instance_id for task in self.tasks}) != 1:
            raise ValueError("paper scenes require one shared target for all tasks")
        too_short = [
            task.task_id
            for task in self.tasks
            if task.planar_displacement + 1.0e-9 < minimum_planar_displacement
        ]
        if too_short:
            raise ValueError(
                "tasks below minimum planar displacement "
                f"{minimum_planar_displacement}: {too_short}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "split": self.split,
            "track": self.track.value,
            "objects": [item.to_dict() for item in self.objects],
            "tasks": [item.to_dict() for item in self.tasks],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClutterScene":
        data = dict(value)
        data["objects"] = tuple(SceneObject.from_dict(item) for item in data.get("objects", ()))
        data["tasks"] = tuple(ManipulationTask.from_dict(item) for item in data.get("tasks", ()))
        return cls(**data)


def load_scene_manifest(path: str | Path) -> Iterator[ClutterScene]:
    """Yield scenes from a versioned JSON Lines manifest."""

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                yield ClutterScene.from_dict(value)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid scene manifest {path}:{line_number}: {exc}") from exc


def write_scene_manifest(path: str | Path, scenes: Iterable[ClutterScene]) -> None:
    """Write deterministic JSON Lines suitable for train/eval split hashing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for scene in scenes:
            stream.write(json.dumps(scene.to_dict(), sort_keys=True, separators=(",", ":")))
            stream.write("\n")
