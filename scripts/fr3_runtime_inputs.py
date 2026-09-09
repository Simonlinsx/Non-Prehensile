"""Enumerate native parameter/model dependencies and compare fixed controller inputs."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

try:
    from .prepare_fr3_osc_scenes import PARAMETERS, SCENE_KEYS
except ImportError:
    from prepare_fr3_osc_scenes import PARAMETERS, SCENE_KEYS


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def inventory(runtime):
    runtime = Path(os.path.abspath(runtime))
    pending = list((runtime / PARAMETERS).glob("*.yaml"))
    pending += [runtime / "examples/sampling_c3/urdf" / name for name in (
        "ground.urdf", "platform.urdf", "end_effector_simple_model.urdf")]
    pending.append(runtime / "shared_contact_model.json")
    files, semantic_inputs = {}, {}
    while pending:
        path = Path(os.path.abspath(pending.pop()))
        if not path.is_relative_to(runtime):
            raise ValueError(f"Native runtime reference escapes its root: {path}")
        name = str(path.relative_to(runtime))
        if name in files:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        files[name] = digest
        semantic_inputs[name] = digest
        if path.suffix == ".yaml":
            parsed = yaml.safe_load(raw)
            for value in strings(parsed):
                candidate = runtime / value
                if candidate.is_file():
                    pending.append(candidate)
                elif Path(value).suffix in (".yaml", ".sdf", ".urdf", ".obj"):
                    raise ValueError(f"Missing runtime YAML dependency: {value}")
            fixed = dict(parsed) if isinstance(parsed, dict) else parsed
            if path.parent == runtime / PARAMETERS and path.name in SCENE_KEYS:
                fixed = {k: v for k, v in fixed.items() if k not in SCENE_KEYS[path.name]}
            semantic_inputs[name] = fixed
        elif path.suffix in (".urdf", ".sdf"):
            # Upstream accepts unbound drake: extension tags. We only inspect
            # mesh references; normalize these tags in memory, hash raw bytes.
            xml = ET.fromstring(raw.replace(b"drake:", b"drake_"))
            references = [node.text for node in xml.findall(".//uri")]
            references += [node.get("filename") for node in xml.findall(".//mesh") if node.get("filename")]
            for value in references:
                if not value:
                    raise ValueError(f"Empty model mesh URI in {path}")
                if "://" in value:
                    raise ValueError(f"Unresolved model package URI: {value}")
                pending.append(path.parent / value)
        elif path.name == "shared_contact_model.json":
            model = json.loads(raw)
            for relative, expected in model["files"].items():
                geometry = runtime / relative
                if hashlib.sha256(geometry.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Shared contact geometry checksum changed: {relative}")
                pending.append(geometry)
    signature = hashlib.sha256(json.dumps(semantic_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(schema="nonprehensile.fr3_native_runtime_inputs.v1", files=files,
                controller_signature_sha256=signature,
                scope="Native runtime parameters and reachable geometry. Robot model, binary, source and Python environment require separate freeze.")
