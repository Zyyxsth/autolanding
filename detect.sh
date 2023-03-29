# apriltag 检测
source /home/anifan/apriltag_ws/devel_isolated/setup.bash; 
roslaunch apriltag_ros sim_double_detection.launch & sleep 2;


# 启动ekf
source /home/anifan/hiauto-landing/autolanding_ws/devel/setup.bash & sleep 1;
roslaunch target_ekf target_ekf.launch  & sleep 1; 


wait;