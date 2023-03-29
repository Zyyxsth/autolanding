roslaunch mapping run.launch & sleep 1; # 测试完成，没有问题
roslaunch planning run.launch & sleep 1; # 启动 planning nodelet 以及 traj server
rosrun sim_node traj_vis_node & sleep 1; # 可视化轨迹的速度和加速度
roslaunch mapping rviz_real.launch & sleep 1 ; # 启动rviz结点

wait;