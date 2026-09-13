"""Separate policy, PhysX, and virtual hardware servo clocks using integer ticks."""
from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Timebase:
    physics_hz: int = 1000
    action_hz: int = 20
    servo_hz: int = 1000

    def __post_init__(self):
        if self.physics_hz not in (120, 240, 1000) or self.action_hz != 20 or self.servo_hz != 1000:
            raise ValueError("Experiment supports PhysX 120/240/1000 Hz, action 20 Hz, servo 1000 Hz")
        if self.physics_hz % self.action_hz:
            raise ValueError("Action must span an integer number of physics steps")

    @property
    def decimation(self):
        return self.physics_hz // self.action_hz

    @property
    def physics_dt(self):
        return 1 / self.physics_hz

    def steps_for(self, seconds):
        return math.ceil(Fraction(str(seconds)) * self.physics_hz)

    def servo_ticks(self, physics_step):
        if physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        return ((physics_step + 1) * self.servo_hz // self.physics_hz
                - physics_step * self.servo_hz // self.physics_hz)

    def contract(self):
        return dict(physics_hz=self.physics_hz, physics_dt_s=self.physics_dt,
                    action_rate_hz=self.action_hz, action_dt_s=1/self.action_hz,
                    decimation=self.decimation, dwell_steps=self.steps_for(.5),
                    virtual_servo_hz=self.servo_hz,
                    servo_ticks_per_action=[self.servo_ticks(i) for i in range(self.decimation)])
