import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16
import numpy as np
import random
import time
import copy
from collections import deque

from px4_msgs.msg import VehicleAngularVelocity, VehicleAttitude, VehicleLocalPosition # subscribed by controller
from px4_msgs.msg import OffboardControlMode, VehicleThrustSetpoint, VehicleTorqueSetpoint # published by controller
from atmos_fmq_msgs.msg import RobotStateResponse, ControlRequest, MultiControlRequest, MultiRobotStateResponse

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy


'''
/fmu/out/vehicle_angular_velocity
/fmu/out/vehicle_attitude
/fmu/out/vehicle_local_position
/fmu/in/offboard_control_mode
/fmu/in/vehicle_thrust_setpoint
/fmu/in/vehicle_torque_setpoint
'''


RELEASE_TIME_INDEX = 1
MESSAGE_INDEX      = 0

class DelayWrapper(Node):
    def __init__(self):

        super().__init__('delay_wrapper')
        self.namespaces      = self.declare_parameter('namespaces', ['']).value
        self.mean_delay     = self.declare_parameter('mean_delay', 100.0).value # ms
        self.std_delay      = self.declare_parameter('std_delay', 20.0).value  # ms

        self.message_drop_rate = 20 # percentage [0- 100]
        self.delay_stats       = {"mean":  self.mean_delay, 
                                   "std":  self.std_delay} # ms

        self.topic_msg_dict = {}
        for ns in self.namespaces:

            self.topic_msg_dict.update(
                                    {f'{ns}/fmq/state'  : (MultiRobotStateResponse, f'{ns}/fmq/state/delayed'),
                                     f'{ns}/fmq/control': (MultiControlRequest    , f'{ns}/fmq/control/delayed'),
                                    })

        self.pubs = {}
        self.subs = {}
        self.msg_queues   = {}

        qos_profile = QoSProfile(
                                  reliability = QoSReliabilityPolicy.BEST_EFFORT,
                                  history     = QoSHistoryPolicy.KEEP_LAST,
                                  depth       = 10,
        )

        # 각 토픽에 대한 publisher/subscriber 등록
        for topic_name, (msg_type, delay_pub_topic) in self.topic_msg_dict.items():

            # publisher: /msg1 
            self.pubs[delay_pub_topic]       = self.create_publisher(msg_type, delay_pub_topic, qos_profile)
            self.msg_queues[delay_pub_topic] = deque() # queue of messages to send over the network
            
            # subscriber: /prefix/msg1 
            self.subs[delay_pub_topic] = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, topic=delay_pub_topic: self.delay_callback(msg, topic),
                qos_profile
            )

        self.create_timer(0.01, self.delayed_publish) 

    def delay_callback(self, msg, delay_pub_topic):
        if np.random.rand() < self.message_drop_rate / 100.0:
            return
        # random gaussian delay
        delay_ms = max(0.0, np.random.normal(self.delay_stats["mean"], self.delay_stats["std"]))
        delay_sec = delay_ms / 1000.0
        self._logger.debug(f'Delaying message on topic {delay_pub_topic} by {delay_sec:.3f} seconds')

        now          = self.get_clock().now().nanoseconds / 1e9
        release_time = (now + delay_sec) 
        self.msg_queues[delay_pub_topic].append((copy.deepcopy(msg), release_time))


    def delayed_publish(self):
        now = self.get_clock().now().nanoseconds / 1e9
        for delay_pub_topic, queue in self.msg_queues.items():
            while queue and queue[0][RELEASE_TIME_INDEX] <= now:
                msg, _ = queue.popleft()
                self.pubs[delay_pub_topic].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DelayWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

