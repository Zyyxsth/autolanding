roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch mavros px4.launch & sleep 10;

rosrun mavros mavcmd long 511 105 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 1;

roslaunch vins vins_fast330_v5_rs435.launch
wait;
