from numpy.core.multiarray import array as array
from regelum.policy import Policy
import numpy as np
from scipy.special import expit
from src.system import (
    InvertedPendulum,
    InvertedPendulumWithFriction,
    InvertedPendulumWithMotor,
)
from typing import Union
from regelum.utils import rg
from regelum import CasadiOptimizerConfig

from .utilities import uptria2vec
from .utilities import to_row_vec
from .utilities import to_scalar

def soft_switch(signal1, signal2, gate, loc=np.cos(np.pi / 4), scale=10):

    # Soft switch coefficient
    switch_coeff = expit((gate - loc) * scale)

    return (1 - switch_coeff) * signal1 + switch_coeff * signal2


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


def pd_based_on_sin(observation, pd_coeffs=[20, 10]):
    return -pd_coeffs[0] * np.sin(observation[0, 0]) - pd_coeffs[1] * observation[0, 1]


class InvPendulumPolicyPD(Policy):
    def __init__(self, pd_coeffs: np.ndarray, action_min: float, action_max: float):
        super().__init__()

        self.pid_coeffs = np.array(pd_coeffs).reshape(1, -1)
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray):
        action = np.clip(
            (self.pid_coeffs * observation).sum(),
            self.action_min,
            self.action_max,
        )
        return np.array([[action]])


class InvertedPendulumEnergyBased(Policy):
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coeffs: np.ndarray,
        system: Union[InvertedPendulum, InvertedPendulumWithFriction],
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionCompensation(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coeffs: np.ndarray,
        system: InvertedPendulumWithFriction,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length, friction_coeff = (
            params["mass"],
            params["grav_const"],
            params["length"],
            params["friction_coeff"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + friction_coeff * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionAdaptive(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        sampling_time: float,
        gain_adaptive: float,
        switch_loc: float,
        pd_coeffs: list,
        system: InvertedPendulumWithFriction,
        friction_coeff_est_init: float = 0,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.friction_coeff_est = friction_coeff_est_init
        self.sampling_time = sampling_time
        self.gain_adaptive = gain_adaptive
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + self.friction_coeff_est * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        # Parameter adaptation using Euler scheme
        self.friction_coeff_est += (
            -self.gain_adaptive
            * energy_total
            * mass
            * length**2
            * np.abs(angle_vel) ** 3
            * self.sampling_time
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )
        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvertedPendulumBackstepping(Policy):

    def __init__(
        self,
        energy_gain: float,
        backstepping_gain: float,
        switch_loc: float,
        pd_coeffs: list[float],
        action_min: float,
        action_max: float,
        system: InvertedPendulumWithMotor,
    ):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.energy_gain = energy_gain
        self.backstepping_gain = backstepping_gain
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = self.system._parameters

        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        torque = observation[0, 2]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.energy_gain * np.sign(angle_vel * energy_total)
        backstepping_action = torque - self.backstepping_gain * (
            torque - energy_control_action
        )
        action_pd = -self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel

        action = hard_switch(
            signal1=backstepping_action,
            signal2=action_pd,
            condition=(np.cos(angle) - 1) ** 2 + angle_vel**2 >= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvertedPendulumWithMotorPD(Policy):

    def __init__(self, pd_coeffs: list, action_min: float, action_max: float):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.pd_coeffs = pd_coeffs

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        action = -self.pd_coeffs[0] * angle - self.pd_coeffs[1] * angle_vel
        return np.array([[np.clip(action, self.action_min, self.action_max)]])


class ThreeWheeledRobotKinematicMinGradCLF(Policy):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        eps=0.01,
    ):
        super().__init__(optimizer_config=optimizer_config)
        self.action_bounds = action_bounds
        # An epsilon for numerical stability
        self.eps = eps
        self.instantiate_optimization_procedure()

    def derivative_of_three_wheeled_robot_kin_lyapunov_function(
        self, x_coord, y_coord, angle, vel, angle_vel
    ):
        x_derivative = vel * rg.cos(angle)
        y_derivative = vel * rg.sin(angle)

        return (
            x_coord * x_derivative
            + y_coord * y_derivative
            + (angle - np.arctan(y_coord / (rg.sign(x_coord) * self.eps + x_coord)))
            * (
                angle_vel
                - (y_derivative * x_coord - x_derivative * y_coord)
                / (x_coord**2 + y_coord**2)
            )
        )

    def instantiate_optimization_procedure(self):
        self.x_coord_var = self.create_variable(1, name="x_coord", is_constant=True)
        self.y_coord_var = self.create_variable(1, name="y_coord", is_constant=True)
        self.angle_var = self.create_variable(1, name="angle", is_constant=True)
        self.vel_var = self.create_variable(
            1, name="vel", is_constant=False, like=np.array([[0]])
        )
        self.angle_vel_var = self.create_variable(
            1, name="angle_vel", is_constant=False, like=np.array([[0]])
        )
        self.register_bounds(self.vel_var, np.array(self.action_bounds[None, 0]))
        self.register_bounds(self.angle_vel_var, np.array(self.action_bounds[None, 1]))

        self.register_objective(
            self.derivative_of_three_wheeled_robot_kin_lyapunov_function,
            variables=[
                self.x_coord_var,
                self.y_coord_var,
                self.angle_var,
                self.vel_var,
                self.angle_vel_var,
            ],
        )

    def get_action(self, observation: np.ndarray):
        x_coord = observation[0, 0]
        y_coord = observation[0, 1]
        angle = observation[0, 2]

        optimized_vel_and_angle_vel = self.optimize(
            x_coord=x_coord, y_coord=y_coord, angle=angle
        )

        # The result of optimization is a dict of casadi tensors, so we convert them to float
        angle_vel = float(optimized_vel_and_angle_vel["angle_vel"][0, 0])
        vel = float(optimized_vel_and_angle_vel["vel"][0, 0])

        return np.array([[vel, angle_vel]])


class ThreeWheeledRobotDynamicMinGradCLF(ThreeWheeledRobotKinematicMinGradCLF):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        gain: float,
        eps: float = 0.01,
    ):
        super().__init__(
            optimizer_config=optimizer_config, eps=eps, action_bounds=action_bounds
        )
        self.gain = gain

    def get_action(self, observation: np.ndarray):
        three_wheeled_robot_kin_action = super().get_action(observation)
        force_and_moment = np.array([[observation[0, 3], observation[0, 4]]])
        action = -self.gain * (force_and_moment - three_wheeled_robot_kin_action)

        return action

class InvertedPendulumRcognitaCALFQ(Policy):
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coeffs: np.ndarray,
        system: Union[InvertedPendulum, InvertedPendulumWithFriction],
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

        ######################################################################### rc
        #### Initialization of CALFQ

        self.dim_state = 2
        self.dim_input = 1
        self.dim_observation = self.dim_state

        self.state_init = np.array([[np.pi, 1]])    # Taken from initial_conditions config
        self.action_init = self.get_safe_action(self.state_init)

        self.action_sampling_time = 0.01    # Taken from common/inv_pendulum config

        self.run_obj_pars = np.diag([1.0, 1.0, 0.0])

        self.gamma = 1

        self.buffer_size = 20
        self.action_buffer = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer = np.zeros( [self.buffer_size, self.dim_observation] ) 

        self.score = 0

        ## Critic
        self.critic_struct = 'quad-nomix'

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_observation + self.dim_input ) + 1 ) *
            ( self.dim_observation + self.dim_input )/2 + 
            (self.dim_observation + self.dim_input) ) 
            self.critic_weight_min = -1e3*np.ones(self.dim_critic) 
            self.critic_weight_max = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_observation + self.dim_input ) + 1 ) *
            ( self.dim_observation +
            self.dim_input )/2 )
            self.critic_weight_min = np.zeros(self.dim_critic) 
            self.critic_weight_max = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_observation + self.dim_input
            self.critic_weight_min = np.zeros(self.dim_critic) 
            self.critic_weight_max = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_observation + self.dim_observation * self.dim_input + self.dim_input )
            self.critic_weight_min = -1e3*np.ones(self.dim_critic)  
            self.critic_weight_max = 1e3*np.ones(self.dim_critic)

        self.critic_weight_tensor_init = to_row_vec( np.random.uniform(10, 1000, size = self.dim_critic) )

        ## CALFQ

        self.critic_weight_tensor_safe = self.critic_weight_tensor_init
        self.observation_safe = self.state_init
        self.action_safe = self.action_init

        self.critic_weight_tensor_init_safe = self.critic_weight_tensor_init

        self.critic_weight_tensor_buffer_safe = []        

        self.critic_low_kappa_coeff = 1e-2
        self.critic_up_kappa_coeff = 1e4
        self.critic_desired_decay = 1e-4

        self.action_buffer_safe = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer_safe = np.zeros( [self.buffer_size, self.dim_observation] )   

        self.count_CALF = 0    
        self.count_safe = 0 

        ## Debugging
        self.debug_print_counter = 0  

        ######################################################################### /rc

    ######################################################################### rc

    def run_obj(self, observation, action):
      
        observation_action = np.hstack([observation, np.array( [[action]] ) ])
        
        result = observation_action @ self.run_obj_pars @ observation_action.T

        return to_scalar(result)

    def critic_model(self, observation, action, critic_weight_tensor):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action) ])
        
        if self.critic_struct == 'quad-lin':
            feature_tensor = np.hstack([
                uptria2vec( np.outer(observation_action, observation_action), force_row_vec=True ), 
                observation_action
                ])
        elif self.critic_struct == 'quadratic':
            feature_tensor = uptria2vec( np.outer(observation_action, observation_action), force_row_vec=True  ) 
        elif self.critic_struct == 'quad-nomix':
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == 'quad-mix':
            feature_tensor = np.hstack([
                to_row_vec(observation)**2,
                np.kron(to_row_vec(observation), to_row_vec(action)),
                to_row_vec(action)**2
                ]) 

        result = critic_weight_tensor @ feature_tensor.T      

        return to_scalar(result)

    def get_safe_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )

    ######################################################################### /rc

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        # Update score (cumulative objective)
        self.score += self.run_obj(observation, action) * self.action_sampling_time

        ############################################ DEBUG

        if self.debug_print_counter % 50 == 0:
            print("--DEBUG-- reward: %4.2f score: %4.2f" % (-self.run_obj(observation, action), -self.score) )
            print("--DEBUG-- critic: %4.2f " % self.critic_model(observation, action, self.critic_weight_tensor_init) )

        self.debug_print_counter += 1

        ############################################ /DEBUG

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )