# ATMOS FleetMQ Integration Package


### Dependencies
Make sure to have the following dependencies installed in your ROS2 workspace:
- `px4_msgs` from [DISCOWER/px4_msgs](https://github.com/DISCOWER/px4_msgs)
- `px4_mpc` from [DISCOWER/px4-mpc](https://github.com/DISCOWER/px4-mpc)
- `ros_fmq_bridge` privately hosted in [DISCOWER/ros_fmq_bridge](https://github.com/DISCOWER/ros_fmq_bridge)
- `qpsolvers` with the `cvxopt` option (for delay compensation control)

### Running the Interface with Hardware
After building the workspace, you can run the interface using the following command:
```bash
ros2 launch atmos_fmq robot_launch.py namespace:=crackle remote_control:=<true|false> 
```


**NOTE**: As of now, only the `crackle` namespace is supported. The `remote_control` parameter can be set to `true` or `false` depending on whether you want to run the controller from the FMQ operator or locally. If set to `false`, a PX4-MPC instance will be launched to control the vehicle towards the prescribed setpoint in `/{namespace}/px4_mpc/setpoint_pose`.

**WARNING**: The `remote_control:=true` was not yet tested on hardware.

### Running the Interface with Simulation
To run the interface in simulation, ensure that you are running the PX4 SITL simulation with `crackle` namespace. The remaining steps will be the same.

### Running the experiments

## Local Operator (Vehicle)

Run the bridge node 

```
ros2 run ros_fmq_bridge bridge_node
```
Run the Docker file. From the package ros_fmq_bridge, within the folder docker, run (you may have to use sudo):
```
docker compose -f docker-compose-vehicle-linux-{name_of_the_robot}.yml up
```
where in place of {name_of_the_robot} you should put the name of the robot you are using. Check that the corresponding file is available in the docker folder.

Finally, launch the control feeder node 

```
ros2 launch atmos_fmq multirobot_test_robot.launch.py namespaces:=pop
```

## Remote operator

Run the Docker file. From the package ros_fmq_bridge, within the folder docker, run (you may have to use sudo):
```
docker compose -f docker-compose-operator-linux-{name_of_the_robot}.yml up
```
where in place of {name_of_the_robot} you should put the name of the robot you are using. Check that the corresponding file is available in the docker folder.


Launch the bridge


```
ros2 run ros_fmq_bridge bridge_node
```

Launch the controller

```
ros2 launch atmos_fmq multirobot_test_controller.launch.py namespaces:=pop
```


