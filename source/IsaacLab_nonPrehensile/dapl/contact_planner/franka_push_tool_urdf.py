"""Generate a Franka URDF with Push Anything's physical end effector.

Push Anything plans contact for a 19.5 mm spherical tip mounted 126.5 mm
from the Franka hand frame.  Isaac Lab's stock Franka instead exposes the
parallel gripper.  Tracking the planned sphere center with the stock gripper
therefore creates a different contact point and an unmodelled object moment.

The generated file keeps the stock Franka kinematic tree, adds the original
Push Anything peg and sphere directly to ``panda_hand``, and removes collision
geometry from the unused fingers.  Visual finger geometry is retained so the
robot remains recognizable in rendered diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


PUSH_TOOL_PEG_RADIUS_M = 0.0127
PUSH_TOOL_PEG_LENGTH_M = 0.1016
PUSH_TOOL_PEG_CENTER_FROM_HAND_M = 0.0604
PUSH_TOOL_TIP_RADIUS_M = 0.0195
PUSH_TOOL_TIP_FROM_HAND_M = 0.1265



def _geometry_element(
    parent: ET.Element,
    *,
    shape: str,
    center_z_m: float,
    visual: bool,
) -> None:
    element = ET.SubElement(parent, "visual" if visual else "collision")
    ET.SubElement(
        element,
        "origin",
        xyz=f"0 0 {center_z_m:.4f}",
        rpy="0 0 0",
    )
    geometry = ET.SubElement(element, "geometry")
    if shape == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            radius=f"{PUSH_TOOL_PEG_RADIUS_M:.4f}",
            length=f"{PUSH_TOOL_PEG_LENGTH_M:.4f}",
        )
    elif shape == "sphere":
        ET.SubElement(
            geometry,
            "sphere",
            radius=f"{PUSH_TOOL_TIP_RADIUS_M:.4f}",
        )
    else:  # pragma: no cover - private helper has fixed call sites.
        raise ValueError(f"unsupported push-tool shape: {shape}")
    if visual:
        material = ET.SubElement(element, "material", name="push_anything_tool")
        ET.SubElement(material, "color", rgba="0.08 0.08 0.08 1.0")


def build_push_anything_franka_urdf(
    source_urdf: str | Path,
    output_urdf: str | Path,
    *,
    disable_finger_collisions: bool = True,
    sphere_only_collision: bool = False,
) -> Path:
    """Write a stock-compatible Franka URDF with the C3 spherical push tool.

    ``package://franka_description`` mesh paths are made absolute because the
    generated URDF normally lives in ``/tmp`` rather than inside the original
    ROS package tree.
    """

    source = Path(source_urdf).expanduser().resolve()
    output = Path(output_urdf).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Franka source URDF does not exist: {source}")

    tree = ET.parse(source)
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError(f"expected a URDF robot root, got {robot.tag!r}")

    links = {link.get("name"): link for link in robot.findall("link")}
    hand = links.get("panda_hand")
    if hand is None:
        raise ValueError("Franka URDF does not contain panda_hand")

    if sphere_only_collision:
        for collision in list(hand.findall("collision")):
            hand.remove(collision)

    package_prefix = "package://franka_description/"
    package_root = source.parents[1]
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith(package_prefix):
            mesh.set(
                "filename",
                str((package_root / filename[len(package_prefix) :]).resolve()),
            )

    if disable_finger_collisions:
        for link_name in ("panda_leftfinger", "panda_rightfinger"):
            link = links.get(link_name)
            if link is None:
                raise ValueError(f"Franka URDF does not contain {link_name}")
            for collision in list(link.findall("collision")):
                link.remove(collision)

    _geometry_element(
        hand,
        shape="cylinder",
        center_z_m=PUSH_TOOL_PEG_CENTER_FROM_HAND_M,
        visual=True,
    )
    if not sphere_only_collision:
        _geometry_element(
            hand,
            shape="cylinder",
            center_z_m=PUSH_TOOL_PEG_CENTER_FROM_HAND_M,
            visual=False,
        )
    for visual in (True, False):
        _geometry_element(
            hand,
            shape="sphere",
            center_z_m=PUSH_TOOL_TIP_FROM_HAND_M,
            visual=visual,
        )

    ET.indent(tree, space="  ")
    payload = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file() or output.read_bytes() != payload:
        output.write_bytes(payload)
    return output


__all__ = [
    "PUSH_TOOL_PEG_CENTER_FROM_HAND_M",
    "PUSH_TOOL_PEG_LENGTH_M",
    "PUSH_TOOL_PEG_RADIUS_M",
    "PUSH_TOOL_TIP_FROM_HAND_M",
    "PUSH_TOOL_TIP_RADIUS_M",
    "build_push_anything_franka_urdf",
]
