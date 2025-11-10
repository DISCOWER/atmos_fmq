std_msgs/Header header
string          robot_name            # name of the controlled robot
int64           latest_ctrl_id        # id of the last control input given
float32         sec_since_latest_ctrl # unit is sec

builtin_interfaces/Duration[] transmission_delays 

px4_msgs/VehicleAngularVelocity vehicle_angular_velocity
px4_msgs/VehicleAttitude        vehicle_attitude
px4_msgs/VehicleLocalPosition   vehicle_local_position
