import rclpy
import numpy as np
from qpsolvers import solve_qp

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


# state indices
X     = 0
Y     = 1
THETA = 2
VX    = 3
VY    = 4
OMEGA = 5

FX   = 0
FY   = 1
ALPHA= 2

def wrap_to_pi(x_):
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

        # lqr model
        self.P = np.array([
                [0.8476,    0.0000,    0.0000,    3.5418,    0.0000,    0.0000],
                [0.0000,    0.8476,   -0.0000,    0.0000,    3.5418,   -0.0000],
                [0.0000,   -0.0000,    0.1737,    0.0000,   -0.0000,    0.1008],
                [3.5418,    0.0000,    0.0000,   30.0183,    0.0000,    0.0000],
                [0.0000,    3.5418,   -0.0000,    0.0000,   30.0183,   -0.0000],
                [0.0000,   -0.0000,    0.1008,    0.0000,   -0.0000,    0.1751]])

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

    # inputs should be a list of (input, duration) tuples
    def predict(self, inputs, x0):
        x = x0.copy()
        for duration, u in inputs:
            steps = int(duration / self.dt) + 1
            dt = duration / steps
            for _ in range(steps):
                # do prediction for the spacecraft model
                # u[0] is forward force, u[1] is lateral force, u[2] is angular acceleration
                x += (self.f(x) + self.g(x) @ u) * dt
                x[2] = wrap_to_pi(x[2])
        return x

    def lyapunov(self, x, x0):
        # Lyapunov function: V(x) = 0.5 * ((x - x0)^T * P * (x - x0))
        # where P is a positive definite matrix
        # solve LQR in matlab to obtain P. 
        # The angular difference is treated in term of the chord length spanned
        # on the unit circle by the angle difference. This is to convert the angular distance into a 
        # vector space where the LQR is defined. chord = 2 * sin(dtheta/2)

        dx           = x - x0
        dtheta       = wrap_to_pi(dx[THETA])      # store the true angle difference
        dx[THETA]    = 2 * np.sin(dtheta / 2)     # transformed angular deviation
        grad         = np.dot(self.P, dx)         # P * f(x)
        grad[THETA] *= np.cos(dtheta / 2)         # multiply by cos(Δθ/2)
        V            = 0.5 * np.dot(dx, np.dot(self.P, dx))
        
        return V, grad

    def control(self, x, x0):
        ##### 250924: xy pd gain tune
        kx      = 1.0 # 0.47
        kxdot   = 4.02
        ky      = 1.0 # 0.47
        kydot   = 4.02
        ktheta  = 0.2 # 0.08
        kw      = 0.3 # 0.13
        dx      = x - x0

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


class InputTuple:
    def __init__(self, control_id: int, timestamp: Time, input: np.ndarray):
        self.control_id = control_id  # unique control input id
        self.timestamp  = timestamp   # time when the input was sent
        self.input      = input       # input vector

class Controller:
    """
    Helper controller class for each robot

    Handles state updates, control generation, and setpoint registration for each robot
    """

    def __init__(self, ns: str='', x = np.zeros(6), ctrl_dt=0.1):
        """
        Initialize the controller instance.
        :param ns: Namespace / robot name
        :param x: Initial state
        :param ctrl_dt: Control update rate (sampling time)
        """


        self.ns                = ns
        self.x0                = x.copy() # desired state, set to initial state at start
        self.x0[THETA]         = wrap_to_pi(self.x0[THETA])
        self.x                 = x.copy() # state from the robot
        self.x[THETA]          = wrap_to_pi(self.x[THETA])
        self.latest_state_msg  = None
        self.model             = SpacecraftModel()


        self.last_setpoint_stamp : Time             = None
        self.ctrl_id             : int              = 0
        self.pending_inputs      : list[InputTuple] = []  # list of (id, time, input) tuples
        self.delay_history       : list[float]      = []
        self.ctrl_dt             : float            = ctrl_dt

    def handle_state_msg(self, state_msg: RobotStateResponse):
        """
        Handle state messages from the robot. Update internal state and manage pending inputs.
        """
        self.latest_state_msg = state_msg
        self.x[X]             = state_msg.vehicle_local_position.x
        self.x[Y]             = state_msg.vehicle_local_position.y
        self.x[THETA]         = state_msg.vehicle_local_position.heading # if quaternion: 2.0 * np.arctan2(-msg.vehicle_attitude.q[3], -msg.vehicle_attitude.q[0])
        self.x[VX]            = state_msg.vehicle_local_position.vx
        self.x[VY]            = state_msg.vehicle_local_position.vy
        self.x[OMEGA]         = state_msg.vehicle_angular_velocity.xyz[2]


        # remove all the inputs that have already been applied
        while len(self.pending_inputs) > 0 and self.pending_inputs[0].control_id < self.latest_state_msg.latest_ctrl_id:
            self.pending_inputs.pop(0)

        for dur in state_msg.transmission_delays:
            self.delay_history.append(dur)
            if len(self.delay_history) > 100:
                self.delay_history.pop(0)

    def generate_control(self, stamp: Time, control_on: bool ) -> tuple[ControlRequest, PoseStamped, TwistStamped]:
        """
        Generate control message based on the current state and desired state.
        1. Predict the state after the delay using the pending inputs.
        2. Compute the control input using the predicted state and desired state.
        3. Package the control input into a ControlRequest message.
        """
        x_for_control = np.zeros(6)
        
        if self.latest_state_msg is None:
            u = np.zeros(3)
        
        else:
            # do prediction
            worst_lyap_val = -np.inf
            Aineq          = []
            bineq          = []
            n_preds        = 10
            
            for _ in range(n_preds):
                x0 = self.x0.copy()
                # delay_profile = [np.clip(self.ctrl_dt + random.uniform(self.model.delay_min, self.model.delay_max) - random.uniform(self.model.delay_min, self.model.delay_max), 0.0, np.inf) for _ in range(len(self.pending_inputs))]

                if len(self.delay_history) >= len(self.pending_inputs):
                    td_profile = [Duration.from_msg(random.choice(self.delay_history)).nanoseconds / 1e9 for _ in range(len(self.pending_inputs))]
                else:
                    td_profile = [random.uniform(self.model.delay_min, self.model.delay_max) for _ in range(len(self.pending_inputs))]
                td_profile.insert(0, -self.latest_state_msg.sec_since_latest_ctrl)
                delay_profile = [np.clip(td_profile[i + 1] - td_profile[i] + self.ctrl_dt,
                                         0,
                                        self.model.delay_max + self.ctrl_dt 
                                        #  np.inf if i < len(td_profile) - 2 else self.model.delay_max + self.ctrl_dt
                                        ) for i in range(len(td_profile) - 1)]

                inputs = [(delay, pending_input[2]) for delay, pending_input in zip(delay_profile, self.pending_inputs)]

                x_pred = self.model.predict(inputs, self.x)

                lyap_val, lyap_grad = self.model.lyapunov(x_pred, x0)

                LfV = np.dot(lyap_grad, self.model.f(x_pred))
                LfG = list(lyap_grad.T @ self.model.g(x_pred))
                LfG.append(-1.0)
                bineq.append(-LfV - 1.0 * lyap_val)
                Aineq.append(LfG)

                x_for_control += x_pred / n_preds
                # lyap_grad
                if lyap_val > worst_lyap_val:
                    worst_lyap_val = lyap_val

            P = np.array([
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 30.0, 0],
                [0, 0, 0, 0.0]
            ])
            # P = np.array([
            #     [1.0, 0, 0, 0],
            #     [0, 1.0, 0, 0],
            #     [0, 0, 10.0, 0],
            #     [0, 0, 0, 0.0]
            # ])
            q = np.array([0.0, 0.0, 0.0, 1.0])
            Aineq = np.array(Aineq)
            bineq = np.array(bineq)
            lb = np.array([-self.model.max_force, -self.model.max_force, -self.model.max_torque, 0.0])
            ub = np.array([self.model.max_force, self.model.max_force, self.model.max_torque, np.inf])
            # sol = solvers.qp(Q, p, Aineq, bineq)
            if control_on:
                u_ref = self.model.control(self.x, self.x0)
                # u_ref = np.array([0.0, 0.0, 0.0])

                sol = solve_qp(P=P, q=np.array([-P[0,0]*u_ref[0], -P[1,1]*u_ref[1], -P[2,2]*u_ref[2], 1.0]), G=Aineq, h=bineq, lb=lb, ub=ub, solver='cvxopt')
                u = sol[0:3]
            else:
                u = np.zeros(3)
            # u = self.model.control(x_for_control, self.x0)
            # u = self.model.control(self.x, self.x0)

        pose_pred_msg                    = PoseStamped()
        pose_pred_msg.header.stamp       = stamp.to_msg()
        pose_pred_msg.header.frame_id    = 'world'
        pose_pred_msg.pose.position.x    = x_for_control[X]
        pose_pred_msg.pose.position.y    = x_for_control[Y]
        pose_pred_msg.pose.position.z    = 0.0
        pose_pred_msg.pose.orientation.w = np.cos(x_for_control[THETA] / 2)
        pose_pred_msg.pose.orientation.x = 0.0
        pose_pred_msg.pose.orientation.y = 0.0
        pose_pred_msg.pose.orientation.z = np.sin(x_for_control[THETA] / 2)

        twist_pred_msg                 = TwistStamped()
        twist_pred_msg.header.stamp    = stamp.to_msg()
        twist_pred_msg.header.frame_id = 'world'
        twist_pred_msg.twist.linear.x  = x_for_control[VX]
        twist_pred_msg.twist.linear.y  = x_for_control[VY]
        twist_pred_msg.twist.linear.z  = 0.0
        twist_pred_msg.twist.angular.x = 0.0
        twist_pred_msg.twist.angular.y = 0.0
        twist_pred_msg.twist.angular.z = x_for_control[OMEGA]

        px4_timestamp                                       = int(stamp.nanoseconds / 1e3)

        control_msg                                         = ControlRequest()
        control_msg.request_time                            = stamp.to_msg()
        control_msg.id                                      = self.ctrl_id
        control_msg.latest_state_message_time               = self.latest_state_msg.response_delivery_time if self.latest_state_msg is not None else Time().to_msg() # time in which the state message was sent by the robot
        control_msg.offboard_control_mode                   = OffboardControlMode()
        control_msg.offboard_control_mode.timestamp         = px4_timestamp
        control_msg.offboard_control_mode.position          = False
        control_msg.offboard_control_mode.velocity          = False
        control_msg.offboard_control_mode.acceleration      = False
        control_msg.offboard_control_mode.attitude          = False
        control_msg.offboard_control_mode.body_rate         = False
        control_msg.offboard_control_mode.direct_actuator   = False
        control_msg.offboard_control_mode.thrust_and_torque = True
        
        control_msg.vehicle_thrust_setpoint           = VehicleThrustSetpoint()
        control_msg.vehicle_torque_setpoint           = VehicleTorqueSetpoint()
        control_msg.vehicle_thrust_setpoint.timestamp = px4_timestamp
        control_msg.vehicle_thrust_setpoint.xyz       = [u[FX], u[FY], 0.0] # worked on Gazebo Sim so far.
        control_msg.vehicle_torque_setpoint.timestamp = px4_timestamp
        control_msg.vehicle_torque_setpoint.xyz       = [0.0, 0.0, u[ALPHA]]
        control_msg.robot_name                        = self.ns

        self.pending_inputs.append((self.ctrl_id, stamp, u))
        self.ctrl_id += 1

        return control_msg, pose_pred_msg, twist_pred_msg

    def register_setpoint(self, pose_sp: PoseStamped, twist_sp: TwistStamped, no_twist=True):
        if pose_sp is not None:
            self.x0[X] = pose_sp.pose.position.x
            self.x0[Y] = pose_sp.pose.position.y
            self.x0[THETA] = 2.0 * np.arctan2(pose_sp.pose.orientation.z, pose_sp.pose.orientation.w)
        else:
            pass # keep pose unchanged

        if twist_sp is not None:
            self.x0[VX]    = twist_sp.twist.linear.x
            self.x0[VY]    = twist_sp.twist.linear.y
            self.x0[OMEGA] = twist_sp.twist.angular.z
        elif no_twist:
            self.x0[VX]    = 0.0
            self.x0[VY]    = 0.0
            self.x0[OMEGA] = 0.0
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
        self.has_twist            : dict[str, bool]         = {}

        
        for ns in self.namespaces:

            # Create publisher of predicted states and control commands from each robot (the predicted state is based on the given control commands and delay model)
            self.predicted_pose_pub[ns]   = self.create_publisher(PoseStamped, f'/{ns}/predicted_pose', qos_profile)
            self.predicted_twist_pub[ns]  = self.create_publisher(TwistStamped, f'/{ns}/predicted_twist', qos_profile)
            self.control_pub[ns]          = self.create_publisher(MultiControlRequest, f'/{ns}/fmq/control', qos_profile)
            
            # Create subscriber to get state messages from each robot
            self.control_on_sub[ns]       = self.create_subscription(Bool, f'/{ns}/control_on', partial(self.control_on_callback, namespace=ns), qos_profile)
            self.state_sub[ns]            = self.create_subscription(MultiRobotStateResponse, f'/{ns}/fmq/state', self.state_callback, qos_profile)

            self.twist_setpoint_subs[ns]  = self.create_subscription(TwistStamped, f'/{ns}/fmq/setpoint_twist', partial(self.twist_setpoint_callback, namespace=ns), qos_profile)
            self.pose_setpoint_subs[ns]   = self.create_subscription(PoseStamped, f'/{ns}/fmq/setpoint_pose', partial(self.pose_setpoint_callback, namespace=ns), qos_profile)

            self.is_control_on[ns]        = True
            self.has_twist[ns]            = False

            # create controller instance for each robot
            self.robots[ns]  = Controller(ns = ns, x = np.zeros(6))


        self.pub_timer     = self.create_timer(self.ctrl_dt, self.publish_control_msgs) # control publish timer
        self.setpoint_subs = []        

    def state_callback(self, msg: MultiRobotStateResponse):
        """Get latest state messages from all robots and update controllers."""
        for robot_state in msg.robot_states:
            controller = self.robots.get(robot_state.robot_name)
            controller.handle_state_msg(robot_state)

    def publish_control_msgs(self):
        msg = MultiControlRequest()

        for (ns, controller) in self.robots.items():
            current_time                = self.get_clock().now()
            wrench_control, pose, twist = controller.generate_control(current_time, self.is_control_on[ns])

            msg.wrench_controls.append(wrench_control)
            self.predicted_pose_pub[ns].publish(pose)
            self.predicted_twist_pub[ns].publish(twist)
            self.control_pub[ns].publish(msg)

    def pose_setpoint_callback(self, msg: PoseStamped, namespace: str):
        if self.robots[namespace] is not None:
            self.robots[namespace].register_setpoint(pose_sp=msg, twist_sp=None, no_twist=not self.has_twist[namespace])

    def twist_setpoint_callback(self, msg: TwistStamped, namespace: str):
        if self.robots[namespace] is not None:
            self.has_twist[namespace] = True
            self.robots[namespace].register_setpoint(pose_sp=None, twist_sp=msg, no_twist=False)

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
