import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped

STABLE_POINT_HEADING = np.pi*0.0 
STABLE_POINT_X       = -0.8
STABLE_POINT_Y       = 1.7968




class FixedSetpoint(Node):
    def __init__(self):
        """ 
        Publishes a fixed setpoint for docking maneuver.
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

        self.parking_pose        = np.array([STABLE_POINT_X, STABLE_POINT_Y, STABLE_POINT_HEADING]) # x, y, yaw;       (pose to be reached before the frontal docking starts)
        self.parking_vel         = np.array([0.0, 0.0     , 0.0])                 # vx, vy, yaw_rate (velocity to be reached before the frontal docking starts)
        # if slower speed needed, set parking x position close to the docking pose

        self.pose_setpoint_pub   = self.create_publisher(PoseStamped    , f'/{self.namespace}/setpoint_pose', qos_profile)
        self.twist_setpoint_pub  = self.create_publisher(TwistStamped   , f'/{self.namespace}/setpoint_twist', qos_profile)
        self.control_on_pub      = self.create_publisher(Bool           , f'/{self.namespace}/control_on', qos_profile)

        self.predicted_pose_sub   = self.create_subscription(PoseStamped , f'/{self.namespace}/predicted_pose', self.pose_callback, qos_profile)
        self.predicted_twist_sub  = self.create_subscription(TwistStamped, f'/{self.namespace}/predicted_twist', self.twist_callback, qos_profile)

        self.pub_timer = self.create_timer(0.1, self.publish_setpoints)

        self.x_pred = np.zeros(6) # x, y, yaw, vx, vy, yaw_rate (full state)

        self.is_pred_pose_available  = False   # Flag to mark the correct acquisition of the pose
        self.is_pred_twist_available = False   # Flag to mark the correct acquisition of the twist
        

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


    def publish_setpoints(self):
        """
        Set point publisher.          
        """
    
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
        self.get_logger().info('going toward parking stable configuration ... current error to parking pose: position %.2f m' % np.linalg.norm(predicted_pose - self.parking_pose) + ' velocity %.2f m/s' % np.linalg.norm(predicted_vel - self.parking_vel))
        



def main(args=None):
    rclpy.init(args=args)
    node = FixedSetpoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
