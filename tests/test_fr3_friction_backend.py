import pytest
import torch
from dapl.contact_planner.fr3_model_runtime import synchronize_arm_friction


class BufferedBackend:
    def __init__(self, ignore_writes=False):
        self.legacy = torch.full((1, 9), .2)
        self.properties = torch.full((1, 9, 3), .3)
        self.ignore_writes = ignore_writes

    def get_dof_friction_coefficients(self):
        return self.legacy.clone()

    def get_dof_friction_properties(self):
        return self.properties.clone()

    def set_dof_friction_coefficients(self, data, indices):
        if not self.ignore_writes:
            self.legacy[indices.long()] = data

    def set_dof_friction_properties(self, data, indices):
        if not self.ignore_writes:
            self.properties[indices.long()] = data


def test_friction_is_committed_to_backend_and_finger_values_are_preserved():
    view = BufferedBackend()
    report = synchronize_arm_friction(view, list(range(7)))
    assert torch.all(view.legacy[:, :7] == 0)
    assert torch.all(view.properties[:, :7] == 0)
    assert torch.all(view.legacy[:, 7:] == .2)
    assert torch.all(view.properties[:, 7:] == .3)
    assert report['properties_before'] != report['properties_after']


def test_backend_ignoring_writes_fails_closed():
    with pytest.raises(RuntimeError, match='did not apply'):
        synchronize_arm_friction(BufferedBackend(ignore_writes=True), list(range(7)))


class LimitRobot:
    device = 'cpu'
    joint_names = ['panda_joint1', 'panda_finger_joint2', 'panda_finger_joint1']

    def __init__(self, ignore_writes=False):
        self.root_physx_view = self
        self.ignore_writes = ignore_writes
        self.position = torch.tensor([[[-2., 2.], [-.008, .048], [0., .04]]])
        self.velocity = torch.tensor([[2.62, 3.402823466e38, .2]])
        self.effort = torch.tensor([[87., 200., 200.]])

    def get_dof_limits(self): return self.position.clone()
    def get_dof_max_velocities(self): return self.velocity.clone()
    def get_dof_max_forces(self): return self.effort.clone()

    def write_joint_position_limit_to_sim(self, data, joint_ids):
        if not self.ignore_writes: self.position[:, joint_ids] = data

    def write_joint_velocity_limit_to_sim(self, data, joint_ids):
        if not self.ignore_writes: self.velocity[:, joint_ids] = data

    def write_joint_effort_limit_to_sim(self, data, joint_ids):
        if not self.ignore_writes: self.effort[:, joint_ids] = data


def test_finger_limits_use_joint_names_and_preserve_arm():
    from dapl.contact_planner.fr3_model_runtime import synchronize_finger_limits
    robot = LimitRobot()
    contract = {'joint_limits': {name: dict(lower=0., upper=.04, velocity=.2, effort=100.)
                                for name in robot.joint_names[1:]}}
    report = synchronize_finger_limits(robot, contract)
    assert torch.equal(robot.position[:, 0], torch.tensor([[-2., 2.]]))
    assert torch.all(robot.velocity[:, 1:] == .2)
    assert torch.all(robot.effort[:, 1:] == 100.)
    assert torch.all(robot.position[:, 1:, 0] == 0.)
    assert report['before']['velocity'] != report['after']['velocity']
    with pytest.raises(RuntimeError, match='did not apply'):
        synchronize_finger_limits(LimitRobot(ignore_writes=True), contract)
