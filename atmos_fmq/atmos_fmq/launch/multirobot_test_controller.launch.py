from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    namespaces_arg = DeclareLaunchArgument(
        'namespaces',
        description='robot names as a list of strings'
    )
    simulated_delay_arg = DeclareLaunchArgument(
        'simulated_delay',
        description='use simulated delay',
        default_value='False',
        choices=['True', 'False']
    )

    mean_delay_arg = DeclareLaunchArgument(
        'mean_delay',
        description='mean delay for message simulation (ms)',
        default_value='100.0'
    )

    std_delay_arg = DeclareLaunchArgument(
        'std_delay',
        description='standard deviation for message simulation (ms)',
        default_value='30.0'
    )



    namespaces = LaunchConfiguration('namespaces')
    namespaces_list = PythonExpression([
        "str('", namespaces, "').split(',')"
    ])

    simulated_delay_ = LaunchConfiguration('simulated_delay')
    simulated_delay  = PythonExpression([
        "bool(", simulated_delay_, ")"
    ])

    mean_delay_ = LaunchConfiguration('mean_delay')
    mean_delay  = PythonExpression([
        "float(", mean_delay_, ")"
    ])

    std_delay_ = LaunchConfiguration('std_delay')
    std_delay  = PythonExpression([
        "float(", std_delay_, ")"
    ])

    nodes = [
        Node(
            package='atmos_fmq',
            namespace='',
            executable='wrench_control',
            name='remote_control',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    'namespaces': namespaces_list,
                    'simulated_delay': simulated_delay
                },
            ],
        ),
        Node(
            package='atmos_fmq',
            namespace='',
            executable='docking',
            name='docking_setpoint_planner',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespaces': namespaces_list, 
                 'simulated_delay': simulated_delay}
            ],
        ),
]
    
    if simulated_delay:
        nodes += [ 
            Node(
                package='atmos_fmq',
                namespace='',
                executable='delay_simulator',
                name='delay_simulator',
                output='screen',
                emulate_tty=True,
                parameters=[
                    {'namespaces': namespaces_list,
                     'mean_delay': mean_delay,
                     'std_delay': std_delay},
                ],
            )
        ]


    return LaunchDescription([
        namespaces_arg,
        simulated_delay_arg,
        mean_delay_arg,
        std_delay_arg,
        *nodes,

    ])
