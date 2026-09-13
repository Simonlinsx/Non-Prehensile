"""Admit a replacement approach only if every original safety test still passes."""
import math


def route_admissible(values, segments, *, position_error, rotation_error,
                     position_tolerance, rotation_tolerance,
                     approach_clearance, forbidden_clearance, support_clearance,
                     contact_distance):
    if len(segments)!=2:
        return False
    numeric=[position_error,rotation_error,*values.values(),
             *(v for s in segments for v in s.values())]
    # +inf is permitted for an empty obstacle set, but NaN must never pass.
    if any(math.isnan(float(x)) for x in numeric):
        return False
    if not (position_error<=position_tolerance and rotation_error<=rotation_tolerance):
        return False
    if not all(s['target_clearance_m']>approach_clearance and
               s['support_clearance_m']>support_clearance for s in segments):
        return False
    return (all(values[f'joint_{phase}_forbidden_clearance_m']>forbidden_clearance
                for phase in ('contact','push','retreat'))
            and all(values[f'joint_{phase}_support_clearance_m']>support_clearance
                    for phase in ('contact','push','retreat'))
            and values['joint_contact_safe_distance_m']<=contact_distance)
