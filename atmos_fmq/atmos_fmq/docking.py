import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import casadi as ca

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped

DOCKING_ORIENTATION = np.pi*0.0 
DOCKING_VELOCITY    = 0. 

X     = 0
Y     = 1
VX    = 2
VY    = 3

FX   = 0
FY   = 1

def zoh_dyn(x, u, dt):
    """
    Zero-order hold dynamics for 2D translation with acceleration inputs.
    State: [x, y, vx, vy]
    Input: [fx, fy] (forces)
    Exact integration for double integrator
    """
    
    mass    = 16.8
    inertia = 0.1594
    vx = x[VX]
    vy = x[VY]
    ax = u[FX] / mass
    ay = u[FY] / mass

    x_next  = vx*dt + 0.5 * dt**2 * ax + x[X]
    y_next  = vy*dt + 0.5 * dt**2 * ay + x[Y]
    vx_next = ax * dt + x[VX]
    vy_next = ay * dt + x[VY]

    return ca.vcat([x_next, y_next, vx_next, vy_next])


class MPCPlanner:
    def __init__(self, max_force : float, N, dt_mpc : float):
        
        ############################################
        # Setup MPC solver (Here it is a Double integrator with 4 states)
        ############################################
        
        # define indices of the last 5 seconds
        _5s = int(5.0 / dt_mpc)
        self.umin = np.tile([-max_force, -max_force], (N, 1)).T # size (2, N)
        self.umax = np.tile([max_force, max_force], (N, 1)).T   # size (2, N)

        # Constrain input in a funnel set that shrinks to zero in the last 5 seconds
        self.umin[:,-_5s:] = self.umin[:,-_5s:] * np.linspace(1, 0.01, _5s)
        self.umax[:,-_5s:] = self.umax[:,-_5s:] * np.linspace(1, 0.01, _5s)

        # Cost weights (increases over the horizon)
        ramp       = np.linspace(1.0, 2.0, N)
        self.R_seq = [np.diag([1.0, 1.0]) * r**2 for r in ramp]
        
        # define MPC solver variables
        self.mpc_solver = ca.Opti("conic")
        self.X          = self.mpc_solver.variable(4, N + 1) # x, y, vx, vy
        self.U          = self.mpc_solver.variable(2, N)     # ax, ay
        self.X0_param   = self.mpc_solver.parameter(4)
        self.XF_param   = self.mpc_solver.parameter(4)
        
        # define MPC solver constraints
        self.mpc_solver.subject_to(self.X[:, 0]  == self.X0_param)
        self.mpc_solver.subject_to(self.X[:, -1] == self.XF_param)

        for k in range(N):
            x_next = zoh_dyn(self.X[:, k], self.U[:, k], dt_mpc)
            self.mpc_solver.subject_to(self.X[:, k + 1] == x_next)      # dynamic constraint
            self.mpc_solver.subject_to(self.U[:, k] >= self.umin[:, k]) # input constraint
            self.mpc_solver.subject_to(self.U[:, k] <= self.umax[:, k]) # input constraint

        # Cost: sum u^T R_k u (only input)
        self.J = 0
        for k in range(N):
            uk     = self.U[:, k]
            Rk     = self.R_seq[k]
            stage  = ca.mtimes([uk.T, Rk, uk])
            self.J = self.J + stage * dt_mpc

        self.mpc_solver.minimize(self.J)
        self.mpc_solver.solver("osqp") # "osqp", "daqp", "qpoases"
        # get MPC as function for performance 
        self.mpc_function = self.mpc_solver.to_function("mpc_solver", [self.X0_param, self.XF_param], [self.U, self.X], ["x0", "xf"], ["u_opt", "x_opt"])

    def solve(self, x0 : np.ndarray, xf : np.ndarray):

        u_opt, x_opt = self.mpc_function(x0, xf)
        return u_opt.full(), x_opt.full()
   


class Docking(Node):
    def __init__(self):
        """ 
        Node to compute and publish docking setpoints. 

        The first set point provided is a "parking" pose in front of the docking station,
        then when close enough to the parking pose, an MPC is solved to dock the vehicle.
        The MPC uses a simple 2D double integrator model with force inputs, and tries
        to reach the docking pose with zero velocity. The MPC has a time-varying input
        constraint that linearly decays to zero in the last 5 seconds of the horizon.
        The cost is only on the input, with a time-varying weight that increases over
        the horizon to reduce magnitude of the input over the horizon.
        """


        super().__init__('docking_setpoints')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Get namespace names
        self.namespace = self.declare_parameter('namespace', '').value
        self.namespace = 'pop'

        self.docking_target_pose = np.array([-1.33, 1.7968, DOCKING_ORIENTATION]) # x, y, yaw;       (pose to be reached at the end of the docking maneuver)
        self.docking_target_vel  = np.array([0.0, 0.00    , DOCKING_VELOCITY])    # vx, vy, yaw_rate (velocity to be reached at the end of the docking maneuver)
        self.parking_pose        = np.array([-0.8, 1.7968 , DOCKING_ORIENTATION]) # x, y, yaw;       (pose to be reached before the frontal docking starts)
        self.parking_vel         = np.array([0.0, 0.0     , 0.0])                 # vx, vy, yaw_rate (velocity to be reached before the frontal docking starts)
        # if slower speed needed, set parking x position close to the docking pose

        self.pose_setpoint_pub   = self.create_publisher(PoseStamped    , f'/{self.namespace}/setpoint_pose', qos_profile)
        self.twist_setpoint_pub  = self.create_publisher(TwistStamped   , f'/{self.namespace}/setpoint_twist', qos_profile)
        self.control_on_pub      = self.create_publisher(Bool           , f'/{self.namespace}/control_on', qos_profile)

        self.predicted_pose_sub   = self.create_subscription(PoseStamped , f'/{self.namespace}/predicted_pose', self.pose_callback, qos_profile)
        self.predicted_twist_sub  = self.create_subscription(TwistStamped, f'/{self.namespace}/predicted_twist', self.twist_callback, qos_profile)

        self.mpc_timer = self.create_timer(1.0, self.compute_traj)
        self.pub_timer = self.create_timer(0.1, self.publish_setpoints)

        self.x_pred = np.zeros(6) # x, y, yaw, vx, vy, yaw_rate (full state)

        self.is_pred_pose_available  = False   # Flag to mark the correct acquisition of the pose
        self.is_pred_twist_available = False   # Flag to mark the correct acquisition of the twist
        
        self.already_parking_pose_reached = False
        self.docking_success = False
        self.traj            = None
        self.traj_start_time = None

        self.max_force  = 0.5 # 1.5
        self.max_torque = 0.2 # 0.5
        
        ############################################
        # Setup MPC solver (Here it is a Double integrator with 4 states)
        ############################################
        
        self.dt_mpc       = 0.5
        self.N            = 40
        self.horizon      = self.dt_mpc * self.N
        self.mpc_planner  = MPCPlanner(self.max_force, self.N, self.dt_mpc)

    def pose_callback(self, msg: PoseStamped):
        self.is_pred_pose_available  = True
        self.x_pred[0] = msg.pose.position.x
        self.x_pred[1] = msg.pose.position.y

        # yaw from quaternion
        q              = msg.pose.orientation
        siny_cosp      = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp      = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.x_pred[2] = np.arctan2(siny_cosp, cosy_cosp)

    def twist_callback(self, msg: TwistStamped):
        self.is_pred_twist_available = True
        self.x_pred[3] = msg.twist.linear.x
        self.x_pred[4] = msg.twist.linear.y
        self.x_pred[5] = msg.twist.angular.z

    def compute_traj(self):
        
        if not (self.is_pred_pose_available and self.is_pred_twist_available):
            self.get_logger().info('Waiting for predicted pose and twist from the wrench controller...')
            self.traj = None
            return

        # do not start solving mpc if too far away from parking pose
        pos_tolerance  = 0.2
        vel_tolerance  = 0.2
        predicted_pose = self.x_pred[0:3]
        predicted_vel  = self.x_pred[3:6]

        is_parking_pose_reached = (np.linalg.norm(predicted_pose - self.parking_pose) <= pos_tolerance and np.linalg.norm(predicted_vel - self.parking_vel) <= vel_tolerance)
                    
        if is_parking_pose_reached and not self.already_parking_pose_reached:
            self.already_parking_pose_reached = True # set to true first time the parking pose is reached
        
        
        if self.already_parking_pose_reached and self.traj is None: # only computed once when parking pose is reached
            self.get_logger().info('Reached parking pose, computing docking trajectory...')

            # extract indices 0, 1, 3, 4 from x_pred (x,y,vx,vy)
            x0_mpc = np.array([self.x_pred[0], self.x_pred[1], self.x_pred[3], self.x_pred[4]])
            xf_mpc = np.array([self.docking_target_pose[0], self.docking_target_pose[1], self.docking_target_vel[0], self.docking_target_vel[1]])
            
            try:
                u_trj,x_trj = self.mpc_planner.solve(x0_mpc, xf_mpc)
                self.traj = x_trj.T
                self.traj_start_time = self.get_clock().now()
                self.get_logger().info('Found feasible docking trajectory')

            except RuntimeError as e:
                self.get_logger().error('MPC planning failed. The following error was returned: %s' % e)
                return
           


    def publish_setpoints(self):
        """
        Set point publisher.  

        Four states :
        1) before parking pose is reached: publish parking pose and turn off control
        2) after parking pose is reached, before docking trajectory is computed: publish parking pose and turn on control
        3) after docking trajectory is computed: publish interpolated trajectory setpoints
        4) after docking trajectory is completed: publish docking pose and turn off control
        
        """

        if not self.already_parking_pose_reached:
            # publish parking pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = self.parking_pose[0]
            pose_msg.pose.position.y = self.parking_pose[1]
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = np.cos(self.parking_pose[2] / 2)
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = np.sin(self.parking_pose[2] / 2)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'world'
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0

            predicted_pose = self.x_pred[0:3]
            predicted_vel  = self.x_pred[3:6]

            self.pose_setpoint_pub.publish(pose_msg)
            self.twist_setpoint_pub.publish(twist_msg)
            self.get_logger().info('going toward parking pose ... current error to parking pose: position %.2f m' % np.linalg.norm(predicted_pose - self.parking_pose) + ' velocity %.2f m/s' % np.linalg.norm(predicted_vel - self.parking_vel))
        

        elif self.already_parking_pose_reached : # then the trajectory is available
            if self.docking_success :
                # docking succeeded, no control
                #  To be considered
                self.control_on_pub.publish(Bool(data=False))
                self._logger.info(f'Docking maneuver already completed, no further setpoints published. Setting control to OFF.')
                
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'world'
                pose_msg.pose.position.x = self.docking_target_pose[0]
                pose_msg.pose.position.y = self.docking_target_pose[1]
                pose_msg.pose.orientation.w = np.cos(self.docking_target_pose[2] / 2)
                pose_msg.pose.orientation.z = np.sin(self.docking_target_pose[2] / 2)
                self.pose_setpoint_pub.publish(pose_msg)
                
                twist_msg = TwistStamped()
                twist_msg.header.stamp    = self.get_clock().now().to_msg()
                twist_msg.header.frame_id = 'world'
                twist_msg.twist.linear.x  = self.docking_target_vel[0]
                twist_msg.twist.linear.y  = self.docking_target_vel[1]
                twist_msg.twist.angular.z = self.docking_target_vel[2]
                self.twist_setpoint_pub.publish(twist_msg)
                return
            else:
                # linear interpolate the trajectory and publish the current pose.
                # attitude is the same to target attitude
                t_now = self.get_clock().now()
                t_elapsed = (t_now - self.traj_start_time).nanoseconds / 1.0e9

                # find the segment
                idx = int(t_elapsed / self.dt_mpc)
                if idx >= self.N:
                    idx = self.N - 1
                t_segment = t_elapsed - idx * self.dt_mpc
                ratio = t_segment / self.dt_mpc

                x0 = self.traj[idx]
                x1 = self.traj[idx + 1]
                x_interp = (1 - ratio) * x0 + ratio * x1

                # publish next docking setpoint

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'world'
                pose_msg.pose.position.x = x_interp[0]
                pose_msg.pose.position.y = x_interp[1]
                pose_msg.pose.position.z = 0.0
                pose_msg.pose.orientation.w = np.cos(self.docking_target_pose[2] / 2)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = np.sin(self.docking_target_pose[2] / 2)

                twist_msg = TwistStamped()
                twist_msg.header.stamp    = self.get_clock().now().to_msg()
                twist_msg.header.frame_id = 'world'
                twist_msg.twist.linear.x  = x_interp[2]
                twist_msg.twist.linear.y  = x_interp[3]
                twist_msg.twist.linear.z  = 0.0
                twist_msg.twist.angular.x = 0.0
                twist_msg.twist.angular.y = 0.0
                twist_msg.twist.angular.z = 0.0


                self.pose_setpoint_pub.publish(pose_msg)
                self.twist_setpoint_pub.publish(twist_msg)
                predicted_pose = self.x_pred[0:3]
                predicted_vel  = self.x_pred[3:6]
                self.get_logger().info('Docking trajectory already computed, predicted error to docking pose: position %.2f m' % np.linalg.norm(predicted_pose - self.docking_target_pose) + ' velocity %.2f m/s' % np.linalg.norm(predicted_vel - self.docking_target_vel))
                
                if t_elapsed >= self.horizon:
                    self.get_logger().info('Docking maneuver completed successfully!')
                    self.docking_success = True


def main(args=None):
    rclpy.init(args=args)
    node = Docking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
