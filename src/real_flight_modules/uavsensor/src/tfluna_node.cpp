#include <ros/ros.h>  
#include <std_msgs/Bool.h> 
#include <std_msgs/Int32MultiArray.h> 

#include <uavsensor/safeserial.hpp>
#include <uavsensor/tfluna_laser.hpp>

SafeSerial::SafeSerialHandle hport;
TFLuna::TFLunaHandle hluna;

int looprate;

void node_param_init(ros::NodeHandle &n)
{
    std::string serial_port;
    int bandrate;
    n.getParam("serial_port", serial_port);
    n.getParam("bandrate", bandrate);
    n.getParam("looprate", looprate);

    hport.setport(serial_port, bandrate);
    hport.init();
}



int main(int argc, char **argv)
{

    ros::init(argc, argv, "tfluna_node");
    ros::NodeHandle n("~");
    node_param_init(n);

    ros::Rate loop_rate(looprate);

    ros::Publisher pub_laser = n.advertise<std_msgs::Int32MultiArray>("data", 10);


    while (ros::ok())
    {
        ros::spinOnce();
        if(hport.check())
        {
            std::string databuff;
            size_t datalen = hport.available(); 
            if(datalen != 0) 
            {
                databuff = hport.read(datalen);

                for(int i = 0; i <= (int)datalen - 1; i++)
                {
                    // printf("%02x ", databuff[i]);
                    hluna.pushbyte((uint8_t)databuff[i]);
                    if(hluna.update_state_from_buff())
                    {
                        std_msgs::Int32MultiArray msg_laser;
                        msg_laser.data.resize(3);
                        msg_laser.data[0] = hluna.dis_mm;
                        msg_laser.data[1] = hluna.amp;
                        msg_laser.data[2] = hluna.temp;
                        
                        pub_laser.publish(msg_laser);
                        // printf("%d\r\n", hluna.dis_mm);
                    
                    }

                }
                // printf("\r\n");

            }
        }

        loop_rate.sleep();
    }
    

    hport.close();

    return 0;
}




