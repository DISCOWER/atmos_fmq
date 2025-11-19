import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped


CENTER_X = -0.1
CENTER_Y = 1.7968
RADIUS   = 1.0
PERIOD   = 60.0  # seconds for one full circle


class CircularSetpoint(Node):
    def __init__(self):
        """
        Publishes a circular trajectory setpoint.
        """

        super().__init__('circular_setpoints')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # namespace
        self.namespace = self.declare_parameter('namespace', 'pop').value

        # Circle parameters (change here)
        self.cx     = CENTER_X
        self.cy     = CENTER_Y
        self.radius = RADIUS

        # Speed: one full revolution every 20 seconds
        self.T     = PERIOD
        self.omega = 2 * np.pi / self.T   # angular speed

        # publishers
        self.pose_setpoint_pub = self.create_publisher(
            PoseStamped, f'/{self.namespace}/setpoint_pose', qos_profile)
        self.twist_setpoint_pub = self.create_publisher(
            TwistStamped, f'/{self.namespace}/setpoint_twist', qos_profile)
        self.control_on_pub = self.create_publisher(
            Bool, f'/{self.namespace}/control_on', qos_profile)

        # subscribers for predicted robot state
        self.predicted_pose_sub = self.create_subscription(
            PoseStamped, f'/{self.namespace}/predicted_pose', self.pose_callback, qos_profile)
        self.predicted_twist_sub = self.create_subscription(
            TwistStamped, f'/{self.namespace}/predicted_twist', self.twist_callback, qos_profile)

        # internal state
        self.x_pred = np.zeros(6)
        self.is_pred_pose_available = False
        self.is_pred_twist_available = False

        # timer
        self.start_time = self.get_clock().now()
        self.pub_timer = self.create_timer(0.1, self.publish_setpoints)


    def pose_callback(self, msg):
        self.is_pred_pose_available = True
        self.x_pred[0] = msg.pose.position.x
        self.x_pred[1] = msg.pose.position.y

        # yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.x_pred[2] = np.arctan2(siny_cosp, cosy_cosp)


    def twist_callback(self, msg):
        self.is_pred_twist_available = True
        self.x_pred[3] = msg.twist.linear.x
        self.x_pred[4] = msg.twist.linear.y
        self.x_pred[5] = msg.twist.angular.z


    def publish_setpoints(self):
        # time since start
        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # circular trajectory
        theta = self.omega * t

        x = self.cx + self.radius * np.cos(theta)
        y = self.cy + self.radius * np.sin(theta)
        yaw = theta + np.pi/2    # tangent direction

        # velocities
        vx = -self.radius * self.omega * np.sin(theta)
        vy =  self.radius * self.omega * np.cos(theta)
        yaw_rate = self.omega

        # Pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = np.cos(yaw / 2)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = np.sin(yaw / 2)

        # Twist message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'world'
        twist_msg.twist.linear.x = vx
        twist_msg.twist.linear.y = vy
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = yaw_rate

        # predicted states
        predicted_pose = self.x_pred[0:3]
        predicted_vel  = self.x_pred[3:6]
        desired_pose   = np.array([x, y, yaw])
        desired_vel    = np.array([vx, vy, yaw_rate])

        # publish
        self.pose_setpoint_pub.publish(pose_msg)
        self.twist_setpoint_pub.publish(twist_msg)

        # logging, same style as your original node
        self.get_logger().info(
            'circle tracking... position error %.2f m, velocity error %.2f m/s' %
            (np.linalg.norm(predicted_pose - desired_pose),
             np.linalg.norm(predicted_vel - desired_vel))
        )


def main(args=None):
    rclpy.init(args=args)
    node = CircularSetpoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
