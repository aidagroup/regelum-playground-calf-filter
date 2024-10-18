from regelum.simulator import Simulator
import gymnasium as gym
import numpy as np
from typing import Optional
import torch
from regelum.system import InvertedPendulum
from regelum.utils import rg
from regelum.callback import detach


@detach
class QuanserInvertedPendulum(detach(InvertedPendulum)):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumEnv(gym.Env):
    def __init__(self, simulator: Simulator) -> None:
        self.render_mode = None
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.simulator = simulator
        self.max_torque = 2
        self.action_space = gym.spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -1, -np.inf]),
            high=np.array([2, 1, np.inf]),
            dtype=np.float32,
        )

    def step(self, u):

        self.state = np.copy(self.simulator.state[0])
        th, thdot = self.state[0], self.state[1]  # th := theta
        u = np.clip(u, -self.max_torque, self.max_torque)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        self.simulator.receive_action(u)
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state)[0]
        th, thdot = self.state[0], self.state[1]
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, sim_step is not None, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.simulator.reset()
        self.state = np.copy(self.simulator.state[0])
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)


class PendulumObserver:
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        cos_angles = observation[:, 0, None]
        sin_angles = observation[:, 1, None]
        angles = torch.arctan2(sin_angles, cos_angles)
        velocities = observation[:, 2, None]
        return torch.cat([angles, velocities], dim=1)


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


class PendulumStabilizingPolicy:
    
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        switch_vel_loc: float,
        pd_coeffs: np.ndarray,
        system: InvertedPendulum,
    ):
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.switch_vel_loc = switch_vel_loc
        self.system = system

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )
        cos_angle = observation[0, 0]
        sin_angle = observation[0, 1]

        angle = torch.arctan2(sin_angle, cos_angle)
        angle_vel = observation[0, 2]

        energy_total = (
            mass * grav_const * length * (torch.cos(angle) - 1) / 2
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * torch.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * torch.sin(angle)
            - self.pd_coeffs[1] * angle_vel,
            condition=torch.cos(angle) <= self.switch_loc
            or torch.abs(angle_vel) > self.switch_vel_loc,
        )

        return torch.tensor(
            [
                [
                    torch.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ],
            device=observation.device,
        )


pendulum_nominal_policy = PendulumStabilizingPolicy(
    gain=0.6,
    action_min=-2,
    action_max=2,
    switch_loc=np.cos(np.pi / 10),
    switch_vel_loc=0.2,
    pd_coeffs=[12, 4],
    system=QuanserInvertedPendulum(),
)

# print(
#     pendulum_nominal_policy.get_action(
#         torch.FloatTensor([[np.cos(np.pi / 10), np.sin(np.pi / 10), 1]])
#     )
# )
