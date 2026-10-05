#!/bin/bash

echo "======================================"
echo "Switch Z1 Gazebo to PURE EFFORT mode"
echo "======================================"

source /opt/ros/noetic/setup.bash
source "${ROS_WS_SETUP:-$HOME/z1_ws/devel/setup.bash}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../config/z1_effort_controllers.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Missing controller configuration: $CONFIG_PATH" >&2
    exit 1
fi

echo "[1] Unpause Gazebo physics..."
rosservice call /gazebo/unpause_physics

echo "[2] Stop original UnitreeJointControllers..."
rosservice call /z1_gazebo/controller_manager/switch_controller "start_controllers: []
stop_controllers:
- 'Joint01_controller'
- 'Joint02_controller'
- 'Joint03_controller'
- 'Joint04_controller'
- 'Joint05_controller'
- 'Joint06_controller'
strictness: 2
start_asap: false
timeout: 0.0"

echo "[3] Load effort controller yaml..."
rosparam load "$CONFIG_PATH" /z1_gazebo

echo "[4] Load effort controllers if needed..."
for controller in \
joint1_effort_controller \
joint2_effort_controller \
joint3_effort_controller \
joint4_effort_controller \
joint5_effort_controller \
joint6_effort_controller
do
    rosservice call /z1_gazebo/controller_manager/load_controller "name: '$controller'" || true
done

echo "[5] Start effort controllers..."
rosservice call /z1_gazebo/controller_manager/switch_controller "start_controllers:
- 'joint1_effort_controller'
- 'joint2_effort_controller'
- 'joint3_effort_controller'
- 'joint4_effort_controller'
- 'joint5_effort_controller'
- 'joint6_effort_controller'
stop_controllers:
- 'Joint01_controller'
- 'Joint02_controller'
- 'Joint03_controller'
- 'Joint04_controller'
- 'Joint05_controller'
- 'Joint06_controller'
strictness: 1
start_asap: true
timeout: 5.0"

echo "[6] Current controller status:"
rosservice call /z1_gazebo/controller_manager/list_controllers

echo "======================================"
echo "Done. Now you can run pure computed torque."
echo "======================================"
