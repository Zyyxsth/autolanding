# 修改mavlink接收imu频率
rosrun mavros mavcmd long 511 105 4000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 31 4000 0 0 0 0 0 & sleep 1; 
roslaunch px4ctrl run_ctrl.launch & sleep 1;
wait;