#include <ros/ros.h>  
#include <std_msgs/Bool.h> 

#include <uavsensor/safeserial.hpp>
#include <uavsensor/ykusrelay.hpp>


SafeSerial::SafeSerialHandle hport;
YKUSRelay::RelayHandle hrelay;
int looprate;
bool cmd = false;
ros::Time last_cmd_time;

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

void cmdCallback(const std_msgs::Bool& msg)
{
    cmd = msg.data;
    last_cmd_time = ros::Time::now();
}


void setrelay(bool cmd)
{
    uint8_t pack[16];
    int len = 0;
    if(cmd == true)
    {
        hrelay.generate_uartpack(pack, &len, YKUSRelay::CMD_ON_REPLY);
    }
    if(cmd == false)
    {
        hrelay.generate_uartpack(pack, &len, YKUSRelay::CMD_OFF_REPLY);
    }   
    hport.write(pack, len);
}

void request(void)
{
    uint8_t pack[16];
    int len = 0;
    hrelay.generate_uartpack(pack, &len, YKUSRelay::CMD_REQUEST);
    hport.write(pack, len);
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "relayctrl_node");
    ros::NodeHandle n("~");
    node_param_init(n);

    ros::Rate loop_rate(looprate);

    ros::Publisher pub_state = n.advertise<std_msgs::Bool>("state", 10);
    ros::Subscriber sub_com = n.subscribe("cmd", 10, cmdCallback, ros::TransportHints().tcpNoDelay());


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
                    hrelay.pushbyte((uint8_t)databuff[i]);
                    if(hrelay.update_state_from_buff())
                    {
                        std_msgs::Bool msg_state;
                        msg_state.data = hrelay.state;
                        pub_state.publish(msg_state);
                    }
                }
            }

            if( (ros::Time::now() - hrelay.last_update_time).toSec() > 0.005 )
            {
                request();
            }

            if
            (
                hrelay.state != cmd
                && hrelay.last_update_time > last_cmd_time
            )
            {
                setrelay(cmd);
            }
        }

        loop_rate.sleep();
    }
    
    setrelay(false);
    sleep(1);

    hport.close();
    
    return 0;
}




