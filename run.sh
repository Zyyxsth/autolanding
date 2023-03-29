sudo cpufreq-set -g performance;
sudo chmod 777 /dev/ttyACM0;
sudo chmod 777 /dev/ttyUSB_relay;
sudo chmod 777 /dev/ttyUSB_laser;
sudo udevadm trigger & sleep 1;


#roslaunch usb_cam usb_cam_double.launch & sleep 1;

source devel/setup.bash;

roslaunch uavsensor rccmd.launch & sleep 1;

#roslaunch uavsensor relay.launch & sleep 0.5;
#roslaunch uavsensor laser.launch & sleep 0.5;

roslaunch realsense2_camera rs_camera.launch & sleep 4;
roslaunch mavros px4.launch & sleep 3;
rosrun mavros mavcmd long 511 105 800 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 31 800 0 0 0 0 0 & sleep 1;
roslaunch vins vins_fast330_v5_rs435.launch & sleep 6;
roslaunch px4ctrl run_ctrl.launch & sleep 1;

#roslaunch mapping run.launch & sleep 1;
#roslaunch planning run.launch & sleep 1;

#roslaunch target_ekf target_ekf.launch  & sleep 1;

#source /home/fast/apriltag_ws/devel/setup.bash;
#roslaunch apriltag_ros double_detection.launch;

wait;
