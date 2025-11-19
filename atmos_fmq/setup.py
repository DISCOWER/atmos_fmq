from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'atmos_fmq'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('atmos_fmq/launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Inkyu Jang',
    maintainer_email='janginkyu.larr@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot              = atmos_fmq.robot:main',
            'wrench_control     = atmos_fmq.wrench_control:main',
            'publish_setpoints  = atmos_fmq.publish_setpoints:main',
            'docking            = atmos_fmq.docking:main',
            'delay_simulator    = atmos_fmq.delay_simulator:main',
            'stable_setpoint    = atmos_fmq.stable_setpoint:main',
            'circular_setpoint  = atmos_fmq.circular_setpoint:main',
        ],
    },
)
