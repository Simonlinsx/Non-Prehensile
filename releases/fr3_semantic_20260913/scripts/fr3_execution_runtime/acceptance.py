"""Evaluate the transferred pose with the registered offline precision/definition."""
from audit_strict_pose_dwell import pose_errors


def acceptance_errors(packet, goal):
    # Both online and offline see the exact serialized position/quaternion values.
    # Do not round, add an epsilon, or loosen the registered strict thresholds.
    pose = packet['pose']
    return pose_errors(pose[:3], pose[3:], goal)
