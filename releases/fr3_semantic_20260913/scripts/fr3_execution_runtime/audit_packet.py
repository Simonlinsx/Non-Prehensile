"""Pack the original acceptance expressions into one device-to-host transfer."""
import torch


def read_packet(contact, force_matrix, position, quaternion, origin, goal, initial, fingers):
    # Keep the scalar reference operation order, including quaternion normalization.
    p = position - origin
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    gq = torch.nn.functional.normalize(goal[3:7], dim=-1)
    values = torch.cat((
        torch.stack((
            torch.linalg.vector_norm(force_matrix, dim=-1).max(),
            contact["legal_physical_safe_hand_contact"][0].to(p.dtype),
            contact["forbidden_robot_contact"][0].to(p.dtype),
            contact["legal_safe_robot_contact"][0].to(p.dtype),
            torch.linalg.vector_norm(p[:2] - goal[:2]),
            abs(p[2] - goal[2]),
            2 * torch.acos(torch.clamp(torch.abs(torch.dot(q, gq)), max=1.)),
            fingers.abs().max(),
            torch.linalg.vector_norm(position[:2] - initial[:2]),
        )), p, q, fingers,
    )).cpu().tolist()
    return dict(force=values[0], safe=bool(values[1]), c1=bool(values[2]),
                legal_low=bool(values[3]), xy=values[4], height=values[5],
                rotation=values[6], max_finger=values[7], displacement=values[8],
                pose=values[9:16], fingers=values[16:])
