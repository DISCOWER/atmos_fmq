string                  robot_name                        # name to which the control input should be applied to
int64                   id                                # Id of the given control input
builtin_interfaces/Time request_time                      # Absolute time at which the request is made

px4_msgs/OffboardControlMode   offboard_control_mode
px4_msgs/VehicleThrustSetpoint vehicle_thrust_setpoint
px4_msgs/VehicleTorqueSetpoint vehicle_torque_setpoint
