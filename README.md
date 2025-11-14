# ATMOS FleetMQ Integration Package
Remote operations with CLF controller and Cvar controller.


### Dependencies
Make sure to have the following dependencies installed in your ROS2 workspace:
- `px4_msgs` from [DISCOWER/px4_msgs](https://github.com/DISCOWER/px4_msgs)
- `px4_mpc` from [DISCOWER/px4-mpc](https://github.com/DISCOWER/px4-mpc)
- `ros_fmq_bridge` privately hosted in [DISCOWER/ros_fmq_bridge](https://github.com/DISCOWER/ros_fmq_bridge)
- `qpsolvers` with the `cvxopt` option (for delay compensation control)


### Running the Interface with Simulation
To run the simulation make sure that you have all the available dependancies as listed in the [Discower page](https://atmos.discower.io/pages/Simulation/). The follow each of the following steps in a different terminal.


##### Setting up the simulator
Step1 : Start px4

from within the folder `/PX4-Autopilot`
```
PX4_UXRCE_DDS_NS=pop make px4_sitl_spacecraft gz_atmos_kthspacelab
```

Step 2: Run dds agent

```
micro-xrce-dds-agent udp4 -p 8888
```

Step 3: Run QGround Control

```
./QGroundControl-x86_64.AppImage 
```

Note: If dds does not connect make sure that the following dds_ip is correctly set in your QGround control
```
vehicle setup -> parameters -> UXRCE_DDS_AG_IP -> set to 2130706433 
```

##### Setting up the controller and robot nodes

Step 4: Launch the controller node

```
ros2 launch atmos_fmq multirobot_test_controller.launch.py namespaces:=pop simulated_delay:=True mean_delay:=100 std_delay:=20 controller_type:=cvar
```

options :

`simulated_delay` (True/False) : Simulate delay in the fmq/control and fmq/state topics
`std_delay` (float) : standard deviation of the delay
`mean_delay` (float) : mean of the delay
`controller_type`(cvar of clf): which controller to use. CVar minimizer (KTH under testing) of CLF controller (SNU)


Step 5: Launch the robot node

```
ros2 launch atmos_fmq multirobot_test_robot.launch.py namespaces:=pop simulated_delay:=True
```
`simulated_delay` (True/False) : Simulate delay in the fmq/control and fmq/state topics


NOTE: If you want to avoid having delay you need to set set `simulated_delay=False` to both the launch files. Otherwise you will have a one directional delay in eaither of the two nodes.










To run the interface in simulation ensure that you are running the PX4 SITL simulation with `crackle` namespace. The remaining steps will be the same.






### Running the Interface with Hardware