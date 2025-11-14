import rclpy
import numpy as np
import casadi as ca

from rclpy.node import Node,Publisher, Subscription
from rclpy.clock import Clock
from rclpy.time import Time, Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from atmos_fmq_msgs.msg import RobotStateResponse, ControlRequest, MultiRobotStateResponse, MultiControlRequest
from px4_msgs.msg import OffboardControlMode, VehicleThrustSetpoint, VehicleTorqueSetpoint

import random
from functools import partial

##############################################################
# named state indices
X     = 0
Y     = 1
THETA = 2
VX    = 3
VY    = 4
OMEGA = 5

FX   = 0
FY   = 1
ALPHA= 2

###############################################################

def wrap_to_pi(x_):
    """Wrap angle to [-pi, pi]"""
    x = x_
    while True:
        if x > np.pi:
            x -= np.pi * 2.0
        elif x < -np.pi:
            x += np.pi * 2.0
        else:
            break
    return x

class SpacecraftModel:
    def __init__(self):

        """
        Simple 2D spacecraft model with position, heading, linear velocity, angular velocity states
        State vector: [x, y, theta, vx, vy, omega]
        Control inputs: [Fx, Fy, alpha] (force in x, force in y, angular acceleration)
        
        Dynamics:
            dx/dt = vx
            dy/dt = vy
            dtheta/dt = omega
            dvx/dt = Fx/mass
            dvy/dt = Fy/mass
            domega/dt = alpha/inertia
        """


        # model parameters
        self.dt         = 0.01
        self.mass       = 16.8
        self.inertia    = 0.1594
        self.max_force  = 1.5
        self.max_torque = 0.5

        # delay model
        self.delay_min = 0.1 # minimum delay in seconds, one-way
        self.delay_max = 0.5 # maximum delay in seconds, one-way


    def f(self, x):
        return np.array([x[VX], 
                         x[VY], 
                         x[OMEGA], 
                         0.0, 
                         0.0, 
                         0.0])

    def g(self, x):
        return np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.cos(x[THETA]) / self.mass, -np.sin(x[THETA]) / self.mass, 0.0               ],
            [np.sin(x[THETA]) / self.mass,  np.cos(x[THETA]) / self.mass, 0.0               ],
            [0.0                     ,                               0.0, 1.0 / self.inertia]])
    
    def f_casadi(self, x):
        return ca.vertcat(x[VX], 
                          x[VY], 
                          x[OMEGA], 
                          0.0, 
                          0.0, 
                          0.0)
    def g_casadi(self, x):
        return ca.vertcat(
            ca.horzcat(0.0, 0.0, 0.0),
            ca.horzcat(0.0, 0.0, 0.0),
            ca.horzcat(0.0, 0.0, 0.0),
            ca.horzcat(ca.cos(x[THETA]) / self.mass, -ca.sin(x[THETA]) / self.mass, 0.0               ),
            ca.horzcat(ca.sin(x[THETA]) / self.mass,  ca.cos(x[THETA]) / self.mass, 0.0               ),
            ca.horzcat(0.0                     ,                               0.0, 1.0 / self.inertia))

    def dynamics(self, x, u):
        return self.f(x) + self.g(x) @ u


    def predict(self, x, u, duration):
        """
        Predict the next state using one Runge-Kutta 4 integration step. 

        Parameters
        ----------
        x : ndarray
            Current state vector.
        u : ndarray
            Control input vector.
        duration : float
            Duration over which to predict the state.

        Returns
        -------
        ndarray
            Predicted next state after one RK4 step.
        """

        x_pred        = x.copy()
        x_pred[THETA] = wrap_to_pi(x[THETA])
        dt            = self.dt
        steps         = int(duration / dt) + 1

        for _ in range(steps):
            k1 = self.dynamics(x_pred, u)
            k2 = self.dynamics(x_pred + 0.5 * dt * k1, u)
            k3 = self.dynamics(x_pred + 0.5 * dt * k2, u)
            k4 = self.dynamics(x_pred + dt * k3, u)
            x_pred += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        x_pred[THETA] = wrap_to_pi(x_pred[THETA])

        return x_pred


    def control(self, x, x_ref):

        ##### 250924: xy pd gain tune
        kx      = 1.0 # 0.47
        kxdot   = 4.02
        ky      = 1.0 # 0.47
        kydot   = 4.02
        ktheta  = 0.2 # 0.08
        kw      = 0.3 # 0.13
        dx      = x - x_ref

        dx[THETA] = wrap_to_pi(dx[THETA])
        vxdot_des = -kx * dx[X]         - kxdot * dx[VX]
        vydot_des = -ky * dx[Y]         - kydot * dx[VY]
        alpha_des = -ktheta * dx[THETA] - kw * dx[OMEGA]

        u = np.zeros(3)
        u[FX]    = vxdot_des * np.cos(x[2]) + vydot_des * np.sin(x[2])
        u[FY]    = -vxdot_des * np.sin(x[2]) + vydot_des * np.cos(x[2])
        u[ALPHA] = alpha_des

        u[FX]    = np.clip(u[FX], -self.max_force, self.max_force)
        u[FY]    = np.clip(u[FY], -self.max_force, self.max_force)
        u[ALPHA] = np.clip(u[ALPHA], -self.max_torque, self.max_torque)

        return u


class InputIdPair:
    def __init__(self, control_id: int, control: np.ndarray):
        self.control_id   : int             = control_id
        self.control      : np.ndarray      = control

class Controller:
    """
    Helper controller class for each robot

    Handles state updates, control generation, and setpoint registration for each robot
    """

    def __init__(self, ns: str='', ctrl_dt=0.1):
        """
        Initialize the controller instance.
        :param ns: Namespace / robot name
        :param ctrl_dt: Control update rate (sampling time)
        """

        self.ns      : str              = ns
        self.ctrl_dt : float            = ctrl_dt
        self.model   : SpacecraftModel  = SpacecraftModel()
        self.state_dim : int            = 6
        self.input_dim : int            = 3
        
        self.x_ref   : np.ndarray       = np.zeros(self.state_dim)  # desired state
        self.x_msg   : np.ndarray       = np.zeros(self.state_dim)  # latest state from robot (arrivea from topic)

        self.ctrl_id                    : int                  = 0
        self.pending_input_requests     : list[InputIdPair]    = []  # list of (id, time, input) tuples
        self.delay_history              : list[float]          = []
        
        self.latest_applied_ctrl_id     : int                  = -1
        self.interval_since_last_ctrl   : float                = 0.0
        self.latest_state_msg           : RobotStateResponse   = None

        self.delay_statistics = {
            'min'   : 0.1,
            'max'   : 0.5,
            'mean'  : 0.25,
            'std'   : 0.05
        }


        # lqr model
        self.P = np.array([
                [0.8476,    0.0000,    0.0000,    3.5418,    0.0000,    0.0000],
                [0.0000,    0.8476,   -0.0000,    0.0000,    3.5418,   -0.0000],
                [0.0000,   -0.0000,    0.1737,    0.0000,   -0.0000,    0.1008],
                [3.5418,    0.0000,    0.0000,   30.0183,    0.0000,    0.0000],
                [0.0000,    3.5418,   -0.0000,    0.0000,   30.0183,   -0.0000],
                [0.0000,   -0.0000,    0.1008,    0.0000,   -0.0000,    0.1751]])
        


        # optimization variables 
        self.optimizer       = ca.Opti("conic")
        self.n_samples       = 10 # number of predicted states samples
        self.x_ref_par       = self.optimizer.parameter(self.state_dim)
        self.x_predicted_par = self.optimizer.parameter(self.state_dim,self.n_samples)
        self.u_ref_par       = self.optimizer.parameter(self.input_dim)
        self.u_var           = self.optimizer.variable(self.input_dim)  # control variable
        self.delta_var       = self.optimizer.variable(1)  # slack variable
        
        self.setup_optimizer()
    

    def lyapunov(self):
        # Lyapunov function: V(x) = 0.5 * ((x - x0)^T * P * (x - x0))
        # where P is a positive definite matrix
        # solve LQR in matlab to obtain P. 
        # The angular difference is treated in term of the chord length spanned
        # on the unit circle by the angle difference. This is to convert the angular distance into a 
        # vector space where the LQR is defined. chord = 2 * sin(dtheta/2)
        
        x     = ca.MX.sym('x'    , self.state_dim)
        x_ref = ca.MX.sym('x_ref', self.state_dim)

        delta_x     = x[X] - x_ref[X]
        delta_y     = x[Y] - x_ref[Y]
        delta_theta = 2 * ca.sin((x[THETA]- x_ref[THETA]) / 2)
        delta_vx    = x[VX] - x_ref[VX]
        delta_vy    = x[VY] - x_ref[VY]
        delta_omega = x[OMEGA] - x_ref[OMEGA]

        delta = ca.vertcat( delta_x, delta_y, delta_theta, delta_vx, delta_vy, delta_omega)
        V            = 0.5 * ca.mtimes(delta.T, ca.mtimes(self.P, delta))
        grad         = ca.jacobian(V, x).T
        
        V_fun        = ca.Function('V_fun', [x, x_ref], [V])
        grad_fun     = ca.Function('grad_fun', [x, x_ref], [grad])

        return V_fun, grad_fun

    def setup_optimizer(self):

        n_samples        = self.n_samples
        V_fun , grad_fun = self.lyapunov()
        
        # CLF constraints
        for i in range(n_samples):
            x_pred              = self.x_predicted_par[:, i]
            lyap_val, lyap_grad = V_fun(x_pred, self.x_ref_par), grad_fun(x_pred, self.x_ref_par)
            dynamics            = self.model.f_casadi(x_pred) + self.model.g_casadi(x_pred) @ self.u_var

            self.optimizer.subject_to(lyap_grad.T @ dynamics + 1.0 * lyap_val <= self.delta_var)

        # control input constraints
        self.optimizer.subject_to(self.u_var[FX]    <= self.model.max_force)
        self.optimizer.subject_to(self.u_var[FX]    >= -self.model.max_force)
        self.optimizer.subject_to(self.u_var[FY]    <= self.model.max_force)
        self.optimizer.subject_to(self.u_var[FY]    >= -self.model.max_force)

        self.optimizer.subject_to(self.u_var[ALPHA] <= self.model.max_torque)
        self.optimizer.subject_to(self.u_var[ALPHA] >= -self.model.max_torque)

        self.optimizer.subject_to(self.delta_var >= 0.0) # slack variable non-negative

        Q = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 30.0, 0],
            [0, 0, 0, 1000.0]
        ])

        q = np.array([0.0, 0.0, 0.0, 1.0])[:,np.newaxis]

        uu     = ca.vertcat(self.u_var, self.delta_var)
        uu_ref = ca.vertcat(self.u_ref_par, 0.0)
        cost   = 0.5 * ca.mtimes((uu-uu_ref).T, ca.mtimes(Q, (uu - uu_ref))) + ca.mtimes(q.T, uu)

        self.optimizer.minimize(cost)
        self.optimizer.solver("daqp")
        self.optimal_controller = self.optimizer.to_function("controller", [self.x_predicted_par, self.x_ref_par,self.u_ref_par], [self.u_var, self.delta_var],["x_pred", "x_ref", "u_ref"], ["u_opt", "delta_opt"])


    def handle_state_msg(self, state_msg: RobotStateResponse):
        """
        Handle state messages from the robot. Update internal state and manage pending inputs.
        """
        self.latest_state_msg = state_msg
        self.x_msg[X]         = state_msg.vehicle_local_position.x
        self.x_msg[Y]         = state_msg.vehicle_local_position.y
        self.x_msg[THETA]     = state_msg.vehicle_local_position.heading # if quaternion: 2.0 * np.arctan2(-msg.vehicle_attitude.q[3], -msg.vehicle_attitude.q[0])
        self.x_msg[VX]        = state_msg.vehicle_local_position.vx
        self.x_msg[VY]        = state_msg.vehicle_local_position.vy
        self.x_msg[OMEGA]     = state_msg.vehicle_angular_velocity.xyz[2]

        self.latest_applied_ctrl_id   = state_msg.latest_ctrl_id
        self.interval_since_last_ctrl = state_msg.interval_since_last_ctrl

        # Remove the list of pending inputs that have been already applied at the time the state message was sent (Id smaller than the latest applied control id)
        while len(self.pending_input_requests) > 0 and self.pending_input_requests[0].control_id < self.latest_applied_ctrl_id:
            self.pending_input_requests.pop(0)
        
        # Update delay history
        for dur in state_msg.transmission_delays:
            self.delay_history.append(Duration.from_msg(dur).nanoseconds / 1e9)
            if len(self.delay_history) > 100:
                self.delay_history.pop(0)

        

        # update delay statistics
        self.delay_statistics['min']   = min(self.delay_history) if len(self.delay_history) > 0 else  self.delay_statistics['min']
        self.delay_statistics['max']   = max(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['max']
        self.delay_statistics['mean']  = np.mean(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['mean']
        self.delay_statistics['std']   = np.std(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['std']

        

    def generate_control(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Generate control message based on the current state and desired state.
        1. Predict the state after the delay using the pending inputs.
        2. Compute the control input using the predicted state and desired state.
        3. Package the control input into a ControlRequest message.
        """
        
        if self.latest_state_msg is None:
            u              = np.zeros(3)
            x_average_pred = self.x_msg.copy()

        else:
            x_pred_batch = np.zeros((self.state_dim, self.n_samples))

            for i in range(self.n_samples):
                x_pred  = self.x_msg.copy()

                delay_profile = np.array([ random.gauss(self.delay_statistics['mean'], self.delay_statistics['std']) for _ in range(len(self.pending_input_requests))]) # sample a delay interval for each pending input
                dt_profile    = np.diff(delay_profile) # delta time between two consecutive pending inputs
                dt_profile    = np.insert(dt_profile, 0, self.ctrl_dt-self.interval_since_last_ctrl) # add the time that remains for the last control input

                for dt, request in zip(dt_profile, self.pending_input_requests):

                    ux     = request.control[FX]
                    uy     = request.control[FY]
                    alpha  = request.control[ALPHA]
                    u      = np.array([ux, uy, alpha])
                    x_pred = self.model.predict(x_pred,u, dt)
                    x_pred_batch[:,i] = x_pred

            x_average_pred   =  np.mean(x_pred_batch, axis=1)
            u_ref            = self.model.control(self.x_msg, self.x_ref)
            u_opt, delta_opt = self.optimal_controller(x_pred_batch, self.x_ref, u_ref)

            u                = u_opt.full().flatten()
            delta            = delta_opt.full()
            request_id       = self.ctrl_id
            self.pending_input_requests.append(InputIdPair(self.ctrl_id, u))

            # step control id (reset after 10000)
            if self.ctrl_id >= 10000:
                self.ctrl_id = 0
            else:
                self.ctrl_id += 1 # augment the control request id

            return u, delta, x_average_pred, request_id

    def zero_input_test_control(self) -> tuple[np.ndarray, np.ndarray, int]:

        request_id       = self.ctrl_id
        u                = np.zeros(3)
        delta            = 0.0
        x_average_pred   = self.x_msg.copy()
        self.pending_input_requests.append(InputIdPair(self.ctrl_id, u))

        # step control id (reset after 10000)
        if self.ctrl_id >= 10000:
            self.ctrl_id = 0
        else:
            self.ctrl_id += 1 # augment the control request id

        return u, delta, x_average_pred, request_id



    def register_pose_setpoint(self, pose_sp: PoseStamped):
        if pose_sp is not None:
            self.x_ref[X]     = pose_sp.pose.position.x
            self.x_ref[Y]     = pose_sp.pose.position.y
            self.x_ref[THETA] = 2.0 * np.arctan2(pose_sp.pose.orientation.z, pose_sp.pose.orientation.w)
        else:
            pass # keep pose unchanged
   
    def register_twist_setpoint(self, twist_sp: TwistStamped):
        
        if twist_sp is not None:
            self.x_ref[VX]    = twist_sp.twist.linear.x
            self.x_ref[VY]    = twist_sp.twist.linear.y
            self.x_ref[OMEGA] = twist_sp.twist.angular.z
        else:
            pass # keep twist unchanged



class RiskController:
    """
    Helper controller class for each robot

    Handles state updates, control generation, and setpoint registration for each robot
    """

    def __init__(self, ns: str='', ctrl_dt=0.1):
        """
        Initialize the controller instance.
        :param ns: Namespace / robot name
        :param ctrl_dt: Control update rate (sampling time)
        """

        self.ns      : str              = ns
        self.ctrl_dt : float            = ctrl_dt
        self.model   : SpacecraftModel  = SpacecraftModel()
        self.state_dim : int            = 6
        self.input_dim : int            = 3
        
        self.x_ref   : np.ndarray       = np.zeros(self.state_dim)  # desired state
        self.x_msg   : np.ndarray       = np.zeros(self.state_dim)  # latest state from robot (arrivea from topic)

        self.ctrl_id                    : int                  = 0
        self.pending_input_requests     : list[InputIdPair]    = []  # list of (id, time, input) tuples
        self.delay_history              : list[float]          = []
        
        self.latest_applied_ctrl_id     : int                  = -1
        self.interval_since_last_ctrl   : float                = 0.0
        self.latest_state_msg           : RobotStateResponse   = None

        self.delay_statistics = {
            'min'   : 0.1,
            'max'   : 0.5,
            'mean'  : 0.25,
            'std'   : 0.05
        }


        # lqr model
        self.P = np.array([
                [0.8476,    0.0000,    0.0000,    3.5418,    0.0000,    0.0000],
                [0.0000,    0.8476,   -0.0000,    0.0000,    3.5418,   -0.0000],
                [0.0000,   -0.0000,    0.1737,    0.0000,   -0.0000,    0.1008],
                [3.5418,    0.0000,    0.0000,   30.0183,    0.0000,    0.0000],
                [0.0000,    3.5418,   -0.0000,    0.0000,   30.0183,   -0.0000],
                [0.0000,   -0.0000,    0.1008,    0.0000,   -0.0000,    0.1751]])
        


        # optimization variables 
        self.optimizer       = ca.Opti("conic")
        self.n_samples       = 10 # number of predicted states samples
        self.x_ref_par       = self.optimizer.parameter(self.state_dim)
        self.x_predicted_par = self.optimizer.parameter(self.state_dim,self.n_samples)
        self.u_ref_par       = self.optimizer.parameter(self.input_dim)
        self.u_var           = self.optimizer.variable(self.input_dim)  # control variable
        
        self.setup_optimizer()
    

    def lyapunov(self):
        # Lyapunov function: V(x) = 0.5 * ((x - x0)^T * P * (x - x0))
        # where P is a positive definite matrix
        # solve LQR in matlab to obtain P. 
        # The angular difference is treated in term of the chord length spanned
        # on the unit circle by the angle difference. This is to convert the angular distance into a 
        # vector space where the LQR is defined. chord = 2 * sin(dtheta/2)
        
        x     = ca.MX.sym('x'    , self.state_dim)
        x_ref = ca.MX.sym('x_ref', self.state_dim)

        delta_x     = x[X] - x_ref[X]
        delta_y     = x[Y] - x_ref[Y]
        delta_theta = 2 * ca.sin((x[THETA]- x_ref[THETA]) / 2)
        delta_vx    = x[VX] - x_ref[VX]
        delta_vy    = x[VY] - x_ref[VY]
        delta_omega = x[OMEGA] - x_ref[OMEGA]

        delta = ca.vertcat( delta_x, delta_y, delta_theta, delta_vx, delta_vy, delta_omega)
        V            = 0.5 * ca.mtimes(delta.T, ca.mtimes(self.P, delta))
        grad         = ca.jacobian(V, x).T
        
        V_fun        = ca.Function('V_fun', [x, x_ref], [V])
        grad_fun     = ca.Function('grad_fun', [x, x_ref], [grad])

        return V_fun, grad_fun

    def setup_optimizer(self):

        n_samples        = self.n_samples
        V_fun , grad_fun = self.lyapunov()

        # weights (tune these as you like)
        alpha = 0.98                # CVaR level
        cvar_weight = 1.0           # weight for CVaR term in cost
        u_norm_weight = 0.1         # weight for explicit ||u||^2 term
        clf_decay = 1.0             # the scalar multiplying V in the CLF expression 

        # decision variables for CVaR formulation
        losses = self.optimizer.variable(n_samples)   # loss_i = max(0, lyapunov_sampled_i)
        t_vars = self.optimizer.variable(n_samples)   # auxiliary vars for CVaR
        zeta   = self.optimizer.variable()            # CVaR threshold (zeta)

        cost = 0

        # collect Lyapunov sampled expressions and add constraints to link to 'losses'
        for i in range(n_samples):
            x_pred              = self.x_predicted_par[:, i]

            # compute V and grad at the sample
            lyap_val = V_fun(x_pred, self.x_ref_par)           # 1x1
            lyap_grad = grad_fun(x_pred, self.x_ref_par)       # state_dim x 1

            # system dynamics at sample (affine in u)
            dynamics = self.model.f_casadi(x_pred) + self.model.g_casadi(x_pred) @ self.u_var

            # CLF / Lyapunov derivative expression (scalar)
            lyap_dot = lyap_grad.T @ dynamics + clf_decay * lyap_val

            # enforce losses[i] >= lyap_dot and losses[i] >= 0  (so losses[i] = max(0, lyap_dot) in optimal solution)
            self.optimizer.subject_to(losses[i] >= lyap_dot)
            self.optimizer.subject_to(losses[i] >= 0)

            # CVaR aux constraints: t_i >= losses_i - zeta and t_i >= 0
            self.optimizer.subject_to(t_vars[i] >= losses[i] - zeta)
            self.optimizer.subject_to(t_vars[i] >= 0)

            # control input constraints (keep yours)
            self.optimizer.subject_to(self.u_var[FX]    <= self.model.max_force)
            self.optimizer.subject_to(self.u_var[FX]    >= -self.model.max_force)
            self.optimizer.subject_to(self.u_var[FY]    <= self.model.max_force)
            self.optimizer.subject_to(self.u_var[FY]    >= -self.model.max_force)

            self.optimizer.subject_to(self.u_var[ALPHA] <= self.model.max_torque)
            self.optimizer.subject_to(self.u_var[ALPHA] >= -self.model.max_torque)

            # original (u - u_ref) quadratic penalty you had
            Q = ca.DM([
                [1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 30.0]
            ])

            uu     = ca.vertcat(self.u_var)
            uu_ref = ca.vertcat(self.u_ref_par)
            control_tracking_cost = 0.5 * ca.mtimes((uu-uu_ref).T, ca.mtimes(Q, (uu - uu_ref)))

            # explicit control-norm regularization
            control_norm_cost = 0.5 * u_norm_weight * ca.mtimes(uu.T, uu)

            cost += control_tracking_cost
            cost += control_norm_cost

            # CVaR computation (Rockafellar–Uryasev):
            # CVaR_alpha = zeta + (1 / ((1 - alpha) * n_samples)) * sum_i t_i
            sum_t = ca.sum1(t_vars)  # returns 1x1
            cvar = zeta + (1.0 / ((1.0 - alpha) * n_samples)) * sum_t

            # add weighted CVaR to cost
            cost += cvar_weight * cvar

            # finalize optimizer
            self.optimizer.minimize(cost)
            # keep your chosen solver
            self.optimizer.solver("osqp")

            # function mapping: inputs are your parameters, output is the optimal u
            self.optimal_controller = self.optimizer.to_function("controller",
                                                                [self.x_predicted_par, self.x_ref_par, self.u_ref_par],
                                                                [self.u_var],
                                                                ["x_pred", "x_ref", "u_ref"],
                                                                ["u_opt"])


    def handle_state_msg(self, state_msg: RobotStateResponse):
        """
        Handle state messages from the robot. Update internal state and manage pending inputs.
        """
        self.latest_state_msg = state_msg
        self.x_msg[X]         = state_msg.vehicle_local_position.x
        self.x_msg[Y]         = state_msg.vehicle_local_position.y
        self.x_msg[THETA]     = state_msg.vehicle_local_position.heading # if quaternion: 2.0 * np.arctan2(-msg.vehicle_attitude.q[3], -msg.vehicle_attitude.q[0])
        self.x_msg[VX]        = state_msg.vehicle_local_position.vx
        self.x_msg[VY]        = state_msg.vehicle_local_position.vy
        self.x_msg[OMEGA]     = state_msg.vehicle_angular_velocity.xyz[2]

        self.latest_applied_ctrl_id   = state_msg.latest_ctrl_id
        self.interval_since_last_ctrl = state_msg.interval_since_last_ctrl

        # Remove the list of pending inputs that have been already applied at the time the state message was sent (Id smaller than the latest applied control id)
        while len(self.pending_input_requests) > 0 and self.pending_input_requests[0].control_id < self.latest_applied_ctrl_id:
            self.pending_input_requests.pop(0)
        
        # Update delay history
        for dur in state_msg.transmission_delays:
            self.delay_history.append(Duration.from_msg(dur).nanoseconds / 1e9)
            if len(self.delay_history) > 100:
                self.delay_history.pop(0)

        

        # update delay statistics
        self.delay_statistics['min']   = min(self.delay_history) if len(self.delay_history) > 0 else  self.delay_statistics['min']
        self.delay_statistics['max']   = max(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['max']
        self.delay_statistics['mean']  = np.mean(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['mean']
        self.delay_statistics['std']   = np.std(self.delay_history) if len(self.delay_history) > 0 else self.delay_statistics['std']

        

    def generate_control(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Generate control message based on the current state and desired state.
        1. Predict the state after the delay using the pending inputs.
        2. Compute the control input using the predicted state and desired state.
        3. Package the control input into a ControlRequest message.
        """
        
        if self.latest_state_msg is None:
            u              = np.zeros(3)
            x_average_pred = self.x_msg.copy()

        else:
            x_pred_batch = np.zeros((self.state_dim, self.n_samples))

            for i in range(self.n_samples):
                x_pred  = self.x_msg.copy()

                delay_profile = np.array([ random.gauss(self.delay_statistics['mean'], self.delay_statistics['std']) for _ in range(len(self.pending_input_requests))]) # sample a delay interval for each pending input
                dt_profile    = np.diff(delay_profile) # delta time between two consecutive pending inputs
                dt_profile    = np.insert(dt_profile, 0, self.ctrl_dt-self.interval_since_last_ctrl) # add the time that remains for the last control input

                for dt, request in zip(dt_profile, self.pending_input_requests):

                    ux     = request.control[FX]
                    uy     = request.control[FY]
                    alpha  = request.control[ALPHA]
                    u      = np.array([ux, uy, alpha])
                    x_pred = self.model.predict(x_pred,u, dt)
                    x_pred_batch[:,i] = x_pred

            x_average_pred   =  np.mean(x_pred_batch, axis=1)
            u_ref            = self.model.control(self.x_msg, self.x_ref)
            u_opt            = self.optimal_controller(x_pred_batch, self.x_ref, u_ref)

            u                = u_opt.full().flatten()
            delta            = 0.0
            request_id       = self.ctrl_id
            self.pending_input_requests.append(InputIdPair(self.ctrl_id, u))

            # step control id (reset after 10000)
            if self.ctrl_id >= 10000:
                self.ctrl_id = 0
            else:
                self.ctrl_id += 1 # augment the control request id

            return u, delta, x_average_pred, request_id

    def zero_input_test_control(self) -> tuple[np.ndarray, np.ndarray, int]:

        request_id       = self.ctrl_id
        u                = np.zeros(3)
        delta            = 0.0
        x_average_pred   = self.x_msg.copy()
        self.pending_input_requests.append(InputIdPair(self.ctrl_id, u))

        # step control id (reset after 10000)
        if self.ctrl_id >= 10000:
            self.ctrl_id = 0
        else:
            self.ctrl_id += 1 # augment the control request id

        return u, delta, x_average_pred, request_id



    def register_pose_setpoint(self, pose_sp: PoseStamped):
        if pose_sp is not None:
            self.x_ref[X]     = pose_sp.pose.position.x
            self.x_ref[Y]     = pose_sp.pose.position.y
            self.x_ref[THETA] = 2.0 * np.arctan2(pose_sp.pose.orientation.z, pose_sp.pose.orientation.w)
        else:
            pass # keep pose unchanged
   
    def register_twist_setpoint(self, twist_sp: TwistStamped):
        
        if twist_sp is not None:
            self.x_ref[VX]    = twist_sp.twist.linear.x
            self.x_ref[VY]    = twist_sp.twist.linear.y
            self.x_ref[OMEGA] = twist_sp.twist.angular.z
        else:
            pass # keep twist unchanged
        


class MultiWrenchControl(Node):
    def __init__(self):
        super().__init__('multi_wrench_control')

        self.namespaces      = self.declare_parameter('namespaces', ['']).value
        self.simulated_delay = self.declare_parameter('simulated_delay', False).value
        
        qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                                 history     = QoSHistoryPolicy.KEEP_LAST,
                                 depth       = 10)

        self.ctrl_dt              : float                 = 0.05 # 0.1
        self.robots               : dict[str, Controller] = {}
        
        self.predicted_pose_pub   : dict[str, Publisher] = {}
        self.predicted_twist_pub  : dict[str, Publisher] = {}
        self.control_pub          : dict[str, Publisher] = {}
        
        self.state_sub            : dict[str, Subscription] = {}
        self.control_on_sub       : dict[str, Subscription] = {}
        self.setpoint_subs        : dict[str, Subscription] = {}
        self.twist_setpoint_subs  : dict[str, Subscription] = {}
        self.pose_setpoint_subs   : dict[str, Subscription] = {}
        self.is_control_on        : dict[str, bool]         = {}

        
        for ns in self.namespaces:

            # Create publisher of predicted states and control commands from each robot (the predicted state is based on the given control commands and delay model)
            self.predicted_pose_pub[ns]   = self.create_publisher(PoseStamped, f'/{ns}/predicted_pose', qos_profile)
            self.predicted_twist_pub[ns]  = self.create_publisher(TwistStamped, f'/{ns}/predicted_twist', qos_profile)
            self.control_pub[ns]          = self.create_publisher(MultiControlRequest, f'/{ns}/fmq/control', qos_profile) # switch to delayed topic is simulated delay is used

            # Create subscriber to get state messages from each robot
            self.control_on_sub[ns]       = self.create_subscription(Bool, f'/{ns}/control_on', partial(self.control_on_callback, namespace=ns), qos_profile)
            self.state_sub[ns]            = self.create_subscription(MultiRobotStateResponse, f'/{ns}/fmq/state' if not self.simulated_delay else f'/{ns}/fmq/state/delayed', self.state_callback, qos_profile) # switch to delayed topic is simulated delay is used

            self.twist_setpoint_subs[ns]  = self.create_subscription(TwistStamped, f'/{ns}/setpoint_twist', partial(self.twist_setpoint_callback, namespace=ns), qos_profile)
            self.pose_setpoint_subs[ns]   = self.create_subscription(PoseStamped, f'/{ns}/setpoint_pose', partial(self.pose_setpoint_callback, namespace=ns), qos_profile)

            self.is_control_on[ns]        = True

            # create controller instance for each robot
            # self.robots[ns]  = Controller(ns = ns)
            self.robots[ns]  = RiskController(ns = ns)

        self.pub_timer     = self.create_timer(self.ctrl_dt, self.publish_control_and_predicted_state_msgs) # control publish timer
        self.setpoint_subs = []        

    def state_callback(self, msg: MultiRobotStateResponse):
        """Get latest state messages from all robots and update controllers."""
        for robot_state in msg.robot_states:
            controller = self.robots.get(robot_state.robot_name)
            controller.handle_state_msg(robot_state)
            # print delay startistics
            self.get_logger().debug(f'Robot: {robot_state.robot_name}, \n Delay Statistics: min={controller.delay_statistics["min"]:.3f}\n, max={controller.delay_statistics["max"]:.3f}\n, mean={controller.delay_statistics["mean"]:.3f}\n, std={controller.delay_statistics["std"]:.3f}\n')

    def publish_control_and_predicted_state_msgs(self):
        msg = MultiControlRequest()

        for (ns, controller) in self.robots.items():
            current_time   = self.get_clock().now()
            if controller.latest_state_msg is not None and self.is_control_on[ns]:
                u, delta, x_average_pred, request_id = controller.generate_control()

            elif controller.latest_state_msg is not None and not self.is_control_on[ns]:
                self.get_logger().warning(f'Control is OFF for robot: {ns}. Sending zero control input.')
                u, delta, x_average_pred, request_id = controller.zero_input_test_control()
            else:
                self.get_logger().warning(f'No state message received yet for robot: {ns}. Sending zero control input.')
                u, delta, x_average_pred, request_id = controller.zero_input_test_control()

            self.get_logger().debug(f'Robot: {ns}, Control Input: Fx={u[FX]}, Fy={u[FY]}, Alpha={u[ALPHA]}, Delta={delta}')

            pose_pred_msg                    = PoseStamped()
            pose_pred_msg.header.stamp       = current_time.to_msg()
            pose_pred_msg.header.frame_id    = 'world'
            pose_pred_msg.pose.position.x    =  x_average_pred[X]
            pose_pred_msg.pose.position.y    =  x_average_pred[Y]
            pose_pred_msg.pose.position.z    = 0.0
            pose_pred_msg.pose.orientation.w = np.cos(x_average_pred[THETA] / 2)
            pose_pred_msg.pose.orientation.x = 0.0
            pose_pred_msg.pose.orientation.y = 0.0
            pose_pred_msg.pose.orientation.z = np.sin(x_average_pred[THETA] / 2)

            twist_pred_msg                 = TwistStamped()
            twist_pred_msg.header.stamp    = current_time.to_msg()
            twist_pred_msg.header.frame_id = 'world'
            twist_pred_msg.twist.linear.x  = x_average_pred[VX]
            twist_pred_msg.twist.linear.y  = x_average_pred[VY]
            twist_pred_msg.twist.linear.z  = 0.0
            twist_pred_msg.twist.angular.x = 0.0
            twist_pred_msg.twist.angular.y = 0.0
            twist_pred_msg.twist.angular.z = x_average_pred[OMEGA]

            px4_timestamp                                       = int(current_time.nanoseconds / 1e3)

            control_msg                                         = ControlRequest()
            control_msg.robot_name                              = ns
            control_msg.request_time                            = current_time.to_msg()
            control_msg.id                                      = request_id
            
            control_msg.offboard_control_mode                   = OffboardControlMode()
            control_msg.offboard_control_mode.timestamp         = px4_timestamp
            control_msg.offboard_control_mode.position          = False
            control_msg.offboard_control_mode.velocity          = False
            control_msg.offboard_control_mode.acceleration      = False
            control_msg.offboard_control_mode.attitude          = False
            control_msg.offboard_control_mode.body_rate         = False
            control_msg.offboard_control_mode.direct_actuator   = False
            control_msg.offboard_control_mode.thrust_and_torque = True
            
            control_msg.vehicle_thrust_setpoint                 = VehicleThrustSetpoint()
            control_msg.vehicle_torque_setpoint                 = VehicleTorqueSetpoint()
            control_msg.vehicle_thrust_setpoint.timestamp       = px4_timestamp
            control_msg.vehicle_thrust_setpoint.xyz             = [u[FX], u[FY], 0.0] # worked on Gazebo Sim so far.
            control_msg.vehicle_torque_setpoint.timestamp       = px4_timestamp
            control_msg.vehicle_torque_setpoint.xyz             = [0.0, 0.0, u[ALPHA]]


            msg.wrench_controls.append(control_msg)

            self.predicted_pose_pub[ns].publish(pose_pred_msg)
            self.predicted_twist_pub[ns].publish(twist_pred_msg)
            self.control_pub[ns].publish(msg)

    def pose_setpoint_callback(self, msg: PoseStamped, namespace: str):
        if self.robots[namespace] is not None:
            self.robots[namespace].register_pose_setpoint(pose_sp=msg)

    def twist_setpoint_callback(self, msg: TwistStamped, namespace: str):
        if self.robots[namespace] is not None:
            self.robots[namespace].register_twist_setpoint(twist_sp=msg)

    def control_on_callback(self, msg: Bool, namespace: str):
        self.is_control_on[namespace] = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = MultiWrenchControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
