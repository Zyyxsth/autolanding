#include <ros/ros.h>  
#include <std_msgs/Bool.h> 
#include <std_msgs/Int32MultiArray.h> 

#include <quadrotor_msgs/TakeoffLand.h> 
#include <geometry_msgs/PoseStamped.h> 
#include <mavros_msgs/RCIn.h> 

// 原来在这里


typedef enum
{
    CH9_NONE,
    CH9_UP,
    CH9_MID,
    CH9_DOWN
} ch9_state_t;

int looprate;

void node_param_init(ros::NodeHandle &n)
{
    n.getParam("looprate", looprate);
}

// 初始状态
ch9_state_t ch9_state = CH9_NONE;
ch9_state_t ch9_last_state = CH9_NONE;

ros::Publisher pub_takeoff;
ros::Publisher pub_track_trigger;
ros::Publisher pub_land_trigger;
ros::Subscriber sub_rc;


void rc_callback(const mavros_msgs::RCIn::ConstPtr& msg) 
{
    int value = msg->channels[8]; // 获取通道9

    ch9_last_state = ch9_state; // 记录上次状态

    // 他是上下颠倒的
    if(value < 1200)
    {
        ch9_state = CH9_UP; // 上
    }
    else if(value >1800)
    {
        ch9_state = CH9_DOWN; // 下
    }
    else
    {
        ch9_state = CH9_MID; // 中
    }



    if(ch9_last_state == CH9_UP && ch9_state == CH9_MID) //由下到上 起飞
    {
        ROS_WARN("[RCCMD] takeoff! \r\n");
        quadrotor_msgs::TakeoffLand takeoff_msg;
        takeoff_msg.takeoff_land_cmd = quadrotor_msgs::TakeoffLand::TAKEOFF;
        pub_takeoff.publish(takeoff_msg);
    }
    else if(ch9_last_state == CH9_MID && ch9_state == CH9_DOWN) //由中间提到上面 跟踪
    {
        ROS_WARN("[RCCMD] track!\r\n");
        geometry_msgs::PoseStamped track_msg;
        pub_track_trigger.publish(track_msg);
    }
    else if(ch9_last_state == CH9_DOWN && ch9_state == CH9_MID) // 由上面提到中间 降落
    {
        ROS_WARN("[RCCMD] land!\r\n");
        geometry_msgs::PoseStamped track_msg;
        pub_land_trigger.publish(track_msg);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tfluna_node");
    ros::NodeHandle n("~");
    node_param_init(n);

    ros::Rate loop_rate(looprate);  // 更新周期

    // 发布，所以直接发在这里就能控制他的行为
    pub_takeoff = n.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 10); // 用来起飞
    pub_track_trigger = n.advertise<geometry_msgs::PoseStamped>("/triger", 10);
    pub_land_trigger = n.advertise<geometry_msgs::PoseStamped>("/land_triger", 10);

    // 接受遥控期输入
    sub_rc = n.subscribe<mavros_msgs::RCIn>("/mavros/rc/in", 1, &rc_callback, ros::TransportHints().tcpNoDelay());



    while (ros::ok())
    {
        ros::spinOnce();
        

        loop_rate.sleep(); 
    }
    


    return 0;
}




