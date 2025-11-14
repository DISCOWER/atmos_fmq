string                        robot_name                # name of the controlled robot
builtin_interfaces/Time       response_time             # absolute time at which the response was sent

int64                         latest_ctrl_id            # id of the control request that is being actively imparted to the system
float32                       interval_since_last_ctrl  # time that passed between the last received input and the response time
builtin_interfaces/Duration[] transmission_delays       # recorded transmission delay

px4_msgs/VehicleAngularVelocity vehicle_angular_velocity
px4_msgs/VehicleAttitude        vehicle_attitude
px4_msgs/VehicleLocalPosition   vehicle_local_position
