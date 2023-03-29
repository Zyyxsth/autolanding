sudo cpufreq-set -g performance;
sudo chmod 777 /dev/ttyACM0;
sudo chmod 777 /dev/ttyUSB_relay;
sudo chmod 777 /dev/ttyUSB_laser;
sudo udevadm trigger & sleep 1;