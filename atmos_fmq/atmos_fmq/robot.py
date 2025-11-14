from typing import NamedTuple

from arrow import now
import rclpy

from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from atmos_fmq_msgs.msg import RobotStateResponse, ControlRequest, MultiRobotStateResponse, MultiControlRequest
from px4_msgs.msg import VehicleTorqueSetpoint, VehicleThrustSetpoint, OffboardControlMode, VehicleAngularVelocity, VehicleAttitude, VehicleLocalPosition
from std_msgs.msg import Bool

from copy import deepcopy
from functools import partial



class ControlFeeder(Node):
    def __init__(self):
        """
        This node acts as a bridge between the remote operators and the robot.
        The remote operator sends control inputs via the /ns/fmq/control_topic,
        and this node does :
            1. Publishes the control inputs to the onboard PX4 controller topics.
            2. Subscribes to the PX4 state topics and republishes them to the /fmq/state as a feedback to the remote operator.
        """

        super().__init__('control_feeder')

        self.state_msg_id : int = 0

        qos_profile      = QoSProfile(
            reliability  = QoSReliabilityPolicy.BEST_EFFORT,
            history      = QoSHistoryPolicy.KEEP_LAST,
            depth        = 10,
        )

        # Create timers: state and control are published to onboard fmu and remote controller
        rate = 10.0 # Hz
        self.pub_to_robot_timer = self.create_timer(1.0 / rate, self.publish_to_robots)
        self.pub_to_ctrl_timer  = self.create_timer(1.0 / rate, self.publish_to_remote_operator)


        # Get namespace names
        self.namespaces = self.declare_parameter('namespaces', ['']).value
        self.simulated_delay = self.declare_parameter('simulated_delay', False).value

        # Create pub/sub and internal states for each namespace
        self.thrust_setpoint_pubs       : dict[str, Publisher] = {}
        self.torque_setpoint_pubs       : dict[str, Publisher] = {}
        self.offboard_control_mode_pubs : dict[str, Publisher] = {}

        self.angular_velocity_subs      : dict[str, list[Subscription]] = {}
        self.attitude_subs              : dict[str, list[Subscription]] = {}
        self.local_position_subs        : dict[str, list[Subscription]] = {}
        self.control_on_sub             : dict[str, Subscription] = {}

        self.state_pubs                 : dict[str, Publisher]    = {}
        self.control_subs               : dict[str, Subscription] = {}         

        self.control_requests_queue        : dict[str, list[ControlRequest]]  = {} # for each control input, record (control_request, arrival_time)
        self.latest_control_id             : dict[str, int]  = {}
        self.latest_control_execution_time : dict[str, Time] = {}
        self.transmission_delays           : list[Duration]  = []

        self.control_on                     : dict[str, bool] = {}
        self.angular_velocity_on            : dict[str, bool] = {}
        self.attitude_on                    : dict[str, bool] = {}
        self.local_position_on              : dict[str, bool] = {}

        self.vehicle_angular_velocity       : dict[str, VehicleAngularVelocity] = {}
        self.vehicle_attitude               : dict[str, VehicleAttitude]        = {}
        self.vehicle_local_position         : dict[str, VehicleLocalPosition]   = {}
        self.delay_robot_state_msgs         : dict[str, RobotStateResponse]     = {}

        for ns in self.namespaces:
            msg_prefix = '' if ns == '' else f'/{ns}'
            self.thrust_setpoint_pubs[ns]       = self.create_publisher(VehicleThrustSetpoint, f'{msg_prefix}/fmu/in/vehicle_thrust_setpoint', qos_profile)
            self.torque_setpoint_pubs[ns]       = self.create_publisher(VehicleTorqueSetpoint, f'{msg_prefix}/fmu/in/vehicle_torque_setpoint', qos_profile)
            self.offboard_control_mode_pubs[ns] = self.create_publisher(OffboardControlMode, f'{msg_prefix}/fmu/in/offboard_control_mode', qos_profile)

            # FleetMQ messages pub/sub
            self.state_pubs[ns]    = self.create_publisher(MultiRobotStateResponse, f'{msg_prefix}/fmq/state', qos_profile)
            self.control_subs[ns]  = self.create_subscription(MultiControlRequest, f'{msg_prefix}/fmq/control' if not self.simulated_delay else f'{msg_prefix}/fmq/control/delayed' , self.control_request_callback, qos_profile)

            # subscription to px4 topics
            # added version v1 for compatibility with older PX4 versions
            self.angular_velocity_subs[ns] = [self.create_subscription( VehicleAngularVelocity, f'{msg_prefix}/fmu/out/vehicle_angular_velocity', partial(self.angular_velocity_callback, namespace=ns), qos_profile)]
            self.angular_velocity_subs[ns] += [self.create_subscription( VehicleAngularVelocity, f'{msg_prefix}/fmu/out/vehicle_angular_velocity_v1', partial(self.angular_velocity_callback, namespace=ns), qos_profile)]
            
            self.attitude_subs[ns]         = [self.create_subscription(VehicleAttitude, f'{msg_prefix}/fmu/out/vehicle_attitude', partial(self.attitude_callback, namespace=ns), qos_profile)]
            self.attitude_subs[ns]         += [self.create_subscription(VehicleAttitude, f'{msg_prefix}/fmu/out/vehicle_attitude_v1', partial(self.attitude_callback, namespace=ns), qos_profile)]
            
            self.local_position_subs[ns]   = [self.create_subscription(VehicleLocalPosition,f'{msg_prefix}/fmu/out/vehicle_local_position',partial(self.local_position_callback, namespace=ns), qos_profile)]
            self.local_position_subs[ns]   += [self.create_subscription(VehicleLocalPosition,f'{msg_prefix}/fmu/out/vehicle_local_position_v1',partial(self.local_position_callback, namespace=ns), qos_profile)]

            self.control_on_sub[ns]       = self.create_subscription(Bool, f'/{ns}/control_on', partial(self.control_on_callback, namespace=ns), qos_profile)

            self.control_on[ns]             = True
            self.angular_velocity_on[ns]    = False
            self.attitude_on[ns]            = False
            self.local_position_on[ns]      = False

            self.control_requests_queue[ns]        = [] # for each control input, record (control_request, arrival_time)
            self.latest_control_id[ns]             = -1
            self.latest_control_execution_time[ns] = self.get_clock().now()


            self.get_logger().info(f'Initialized pub/sub for robot name = \'{ns}\'')

    def publish_to_robots(self):
        """
        Publish latest control input received to onboard fmu. This is done at a fixed rate. Once a control request is sent to the robot is popped from the queue if more than one control request is in the queue.
        """
        for ns in self.namespaces:
            if self.control_on[ns]:
                
                if len(self.control_requests_queue[ns]) == 0:
                    self.get_logger().info(f'Control ON but no control requests in the queue for robot namespace {ns}. Last request will be delivered to the robot again.')
                    continue
                
                cur_time_stamp                                     = self.get_clock().now()
                latest_control_request                             = self.control_requests_queue[ns][-1]

                latest_thrust_setpoint                             = latest_control_request.vehicle_thrust_setpoint
                latest_torque_setpoint                             = latest_control_request.vehicle_torque_setpoint
                latest_offboard_control_mode                       = latest_control_request.offboard_control_mode

                latest_thrust_setpoint.timestamp                   = int(cur_time_stamp.nanoseconds / 1e3)
                latest_torque_setpoint.timestamp                   = int(cur_time_stamp.nanoseconds / 1e3)
                latest_offboard_control_mode.timestamp             = int(cur_time_stamp.nanoseconds / 1e3)

                # command to px4
                self.thrust_setpoint_pubs[ns].publish(latest_thrust_setpoint)
                self.torque_setpoint_pubs[ns].publish(latest_torque_setpoint)
                self.offboard_control_mode_pubs[ns].publish(latest_offboard_control_mode)

                self.latest_control_execution_time[ns]    = cur_time_stamp
                self.latest_control_id[ns]                = latest_control_request.id


                if len(self.control_requests_queue[ns]) > 1:
                    # pop latest control request from the queue
                    self.control_requests_queue[ns].pop(0)
            else :
                self.get_logger().debug(f'Control OFF for robot namespace {ns}, not sending any control commands to the robot.')

    def publish_to_remote_operator(self):
        """
        Publish latest robot state to remote operators for feedback. This is done at a fixed rate
        """

        msg              = MultiRobotStateResponse()
        msg.robot_states = []
        self.get_logger().info('Publishing robot states to remote controller. List of namespaces: ' + ', '.join(self.namespaces))

        for ns in self.namespaces:
            cur_time = self.get_clock().now()

            if self.angular_velocity_on[ns] and self.attitude_on[ns] and self.local_position_on[ns] and self.control_on[ns]:

                msg_robot                          = RobotStateResponse()
                msg_robot.robot_name               = ns
                msg_robot.response_time            = cur_time.to_msg()

                msg_robot.latest_ctrl_id             = self.latest_control_id[ns]
                msg_robot.interval_since_last_ctrl   = (cur_time - self.latest_control_execution_time[ns]).nanoseconds / 1e9
                msg_robot.transmission_delays        = self.transmission_delays

                # save the state in the message
                msg_robot.vehicle_angular_velocity = self.vehicle_angular_velocity[ns]
                msg_robot.vehicle_attitude         = self.vehicle_attitude[ns]
                msg_robot.vehicle_local_position   = self.vehicle_local_position[ns]

                msg.robot_states.append(msg_robot)


            else :
                self.get_logger().info(f'Not publishing state for robot namespace {ns} because not all required topics are available yet. Control on: {self.control_on[ns]}, Angular velocity on: {self.angular_velocity_on[ns]}, Attitude on: {self.attitude_on[ns]}, Local position on: {self.local_position_on[ns]}')
        

        for ns in self.namespaces:
            self.state_pubs[ns].publish(msg)


    def control_request_callback(self, msg: MultiControlRequest):
        """ 
        Callback for receiving control inputs from the remote operator. Every time a message is received it is saved as the latest control input to be sent to the robot.
        Each message is identified by an unique id. 

        The callback keeps track of the latest control inputs and their arrival times.
        """
        
        new_requests : list[ControlRequest] = msg.wrench_controls
        self.get_logger().info(f'Received {len(new_requests)} new control requests from remote operator.')

        for control_request in new_requests:
            ns = control_request.robot_name
            self.get_logger().debug(f'Received control request for robot namespace: {ns}, control id: {control_request.id}, control: {control_request.vehicle_thrust_setpoint}.')
            
            if ns not in self.namespaces:
                self.get_logger().warning(f'Unknown robot name: {ns}')
                continue

            self.control_requests_queue[ns].append(control_request)
            # transmission delay computation
            now       = self.get_clock().now()

            stamp = Time.from_msg(control_request.request_time)
            delay = (now - stamp).to_msg()  # type: Duration

            self.get_logger().info(
                f"Control request for namespace {ns} has delay {delay} s"
            )

            self.transmission_delays.append(delay)
            if len(self.transmission_delays) > 100:
                self.transmission_delays.pop(0)

    def angular_velocity_callback(self, msg: VehicleAngularVelocity, namespace):
        if not self.angular_velocity_on[namespace]:
            self.angular_velocity_on[namespace] = True
        self.vehicle_angular_velocity[namespace] = deepcopy(msg)

    def attitude_callback(self, msg: VehicleAttitude, namespace):
        if not self.attitude_on[namespace]:
            self.attitude_on[namespace] = True
        self.vehicle_attitude[namespace] = deepcopy(msg)

    def local_position_callback(self, msg: VehicleLocalPosition, namespace):
        if not self.local_position_on[namespace]:
            self.local_position_on[namespace] = True
        self.vehicle_local_position[namespace] = deepcopy(msg)

    def control_on_callback(self, msg: Bool, namespace: str):
        self.control_on[namespace] = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = ControlFeeder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
