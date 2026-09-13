"""C2/C3 observation independent of which constraints stop the controller."""
ROBOT_SENSORS = tuple([f'robot_obstacle_link{i}_contacts' for i in range(8)] +
    ['robot_obstacle_hand_contacts', 'robot_obstacle_leftfinger_contacts', 'robot_obstacle_rightfinger_contacts'])
CLUTTER_FORCE_N = .02


def read_typed_state(base):
    from fr3_semantic_cost_map import read_contacts
    return read_contacts(base)
    import torch
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp
    if int(base._clutter_active_obstacle_count) == 0:
        zero = torch.zeros(base.num_envs, device=base.device)
        return dict(c2=zero.bool(), c3=zero.bool(), target_obstacle_force_n=zero,
                    robot_obstacle_force_n=zero, protected_clearance_m=zero + float('inf'))

    def force(name):
        if name not in base.scene.sensors:
            raise RuntimeError(f'Typed audit missing sensor: {name}')
        matrix = base.scene.sensors[name].data.force_matrix_w
        if matrix is None or not matrix.numel() or matrix.shape[0] != base.num_envs:
            raise RuntimeError(f'Typed audit missing filtered forces: {name}')
        values = torch.linalg.vector_norm(matrix, dim=-1).reshape(base.num_envs, -1).amax(dim=1)
        if not torch.isfinite(values).all():
            raise RuntimeError(f'Typed audit nonfinite force: {name}')
        return values

    target_force = force('target_obstacle_contacts')
    robot_force = torch.stack([force(name) for name in ROBOT_SENSORS]).amax(dim=0)
    state = mdp.domino_affordance_contact_state(base, contact_distance_m=.01,
        protected_clearance_m=.005, evaluate_protected=True, evaluate_robot_obstacle=True,
        require_physical_protected_contact=True, physical_contact_force_threshold_n=CLUTTER_FORCE_N,
        robot_target_sensor_name='target_robot_contacts', hand_target_sensor_name='target_hand_contacts',
        target_obstacle_sensor_name='target_obstacle_contacts', robot_obstacle_sensor_name=ROBOT_SENSORS)
    return dict(c2=state['protected_obstacle_collision'], c3=state['robot_obstacle_collision'],
        target_obstacle_force_n=target_force, robot_obstacle_force_n=robot_force,
        protected_clearance_m=state['protected_clearance'])


def should_stop(c2, c3, scope):
    if scope not in ('observe', 'c1-c2', 'c1-c3', 'combined'):
        raise ValueError(f'Unknown typed stop scope: {scope}')
    return bool((c2 and scope in ('c1-c2', 'combined')) or
                (c3 and scope in ('c1-c3', 'combined')))


def audit_typed_rows(rows, goal, initial, dt=1/240):
    import math
    from audit_m1_c3_comparison import audit_rows
    values = list(rows)
    c2 = c3 = False
    for row in values:
        for key in ('c2', 'c3'):
            if type(row[key]) is not bool:
                raise ValueError(f'Invalid typed predicate: {key}')
        for key in ('target_obstacle_force_n', 'robot_obstacle_force_n'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Invalid typed force: {key}')
        clearance = row['protected_clearance_m']
        if clearance is not None and (not math.isfinite(clearance) or clearance < 0):
            raise ValueError('Invalid protected clearance')
        expected_c2 = (row['target_obstacle_force_n'] > CLUTTER_FORCE_N and
                       clearance is not None and clearance <= .005)
        expected_c3 = row['robot_obstacle_force_n'] > CLUTTER_FORCE_N
        if row['c2'] != expected_c2 or row['c3'] != expected_c3:
            raise ValueError('Typed predicates disagree with recorded force/geometry')
        c2 |= row['c2']; c3 |= row['c3']
    original = audit_rows(iter(values), goal, initial, dt=dt)
    return {**original, 'c1_only_success': original['strict_success'], 'c2_violation': c2,
            'c3_violation': c3, 'strict_success': original['strict_success'] and not c2 and not c3}
