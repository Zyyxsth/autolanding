
sudo udevadm trigger & sleep 1;
roslaunch usb_cam usb_cam_double.launch & sleep 1; # 启用摄像头
source devel/setup.bash;
roslaunch target_ekf target_ekf.launch  & sleep 1; # 使用ekf
source /home/fast/apriltag_ws/devel/setup.bash;
roslaunch apriltag_ros double_detection.launch; # 使用apriltag检测




