from setuptools import setup
import os

package_name = 'my_robot_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'torch', 'transformers', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='Bin Hyeon-uk',
    maintainer_email='researcher@todo.todo',
    description='Ablation study package for RGB and RGB-D semantic segmentation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = my_robot_segmentation.segmentation_node:main'
        ],
    },
)