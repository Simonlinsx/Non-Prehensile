from pathlib import Path
import xml.etree.ElementTree as ET

from dapl.contact_planner.franka_push_tool_urdf import (
    PUSH_TOOL_TIP_FROM_HAND_M,
    build_push_anything_franka_urdf,
)


def test_build_push_anything_franka_urdf(tmp_path: Path) -> None:
    package = tmp_path / "franka_description"
    robots = package / "robots"
    mesh = package / "meshes" / "collision" / "hand.stl"
    robots.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    source = robots / "panda.urdf"
    source.write_text(
        """<?xml version="1.0"?>
<robot name="panda">
  <link name="panda_hand">
    <visual><geometry><mesh filename="package://franka_description/meshes/collision/hand.stl"/></geometry></visual>
  </link>
  <link name="panda_leftfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="panda_rightfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
</robot>
"""
    )
    output = tmp_path / "generated" / "panda_push.urdf"

    assert build_push_anything_franka_urdf(source, output) == output
    root = ET.parse(output).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    hand = links["panda_hand"]
    assert len(hand.findall("collision")) == 2
    assert len(hand.findall("visual")) == 3
    sphere_collision = next(
        collision
        for collision in hand.findall("collision")
        if collision.find("geometry/sphere") is not None
    )
    assert float(sphere_collision.find("origin").get("xyz").split()[2]) == (
        PUSH_TOOL_TIP_FROM_HAND_M
    )
    assert not links["panda_leftfinger"].findall("collision")
    assert not links["panda_rightfinger"].findall("collision")
    assert hand.find("visual/geometry/mesh").get("filename") == str(mesh.resolve())


def test_build_can_keep_native_finger_collisions(tmp_path: Path) -> None:
    source = tmp_path / "franka_description" / "robots" / "panda.urdf"
    source.parent.mkdir(parents=True)
    source.write_text(
        """<robot name="panda">
<link name="panda_hand"/>
<link name="panda_leftfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
<link name="panda_rightfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
</robot>"""
    )
    output = tmp_path / "generated.urdf"
    build_push_anything_franka_urdf(
        source, output, disable_finger_collisions=False
    )
    links = {
        link.get("name"): link for link in ET.parse(output).getroot().findall("link")
    }
    assert len(links["panda_leftfinger"].findall("collision")) == 1
    assert len(links["panda_rightfinger"].findall("collision")) == 1


def test_build_sphere_only_collision_model(tmp_path: Path) -> None:
    source = tmp_path / "franka_description" / "robots" / "panda.urdf"
    source.parent.mkdir(parents=True)
    source.write_text(
        """<robot name="panda">
<link name="panda_hand"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
<link name="panda_leftfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
<link name="panda_rightfinger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
</robot>"""
    )
    output = tmp_path / "generated.urdf"
    build_push_anything_franka_urdf(
        source, output, sphere_only_collision=True
    )
    links = {
        link.get("name"): link for link in ET.parse(output).getroot().findall("link")
    }
    hand_collisions = links["panda_hand"].findall("collision")
    assert len(hand_collisions) == 1
    assert hand_collisions[0].find("geometry/sphere") is not None
    assert len(links["panda_hand"].findall("visual")) == 2
    assert not links["panda_leftfinger"].findall("collision")
    assert not links["panda_rightfinger"].findall("collision")
