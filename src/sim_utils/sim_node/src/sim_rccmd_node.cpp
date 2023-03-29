#include <ros/ros.h>  
#include <std_msgs/Bool.h> 
#include <std_msgs/Int32MultiArray.h> 

#include <quadrotor_msgs/TakeoffLand.h> 
#include <geometry_msgs/PoseStamped.h> 
#include <mavros_msgs/RCIn.h> 
// #include <sim_node/sim_rccmd.h>
#include <sim_msgs/sim_rccmd.h>

#define LAND_STAIGHT_DOWNWARD 4

// 原来在这里


// typedef enum
// {
//     CH9_NONE,
//     CH9_UP,
//     CH9_MID,
//     CH9_DOWN
// } ch9_state_t;

int looprate;

void node_param_init(ros::NodeHandle &n)
{
    n.getParam("looprate", looprate);
}

// // 初始状态
// ch9_state_t ch9_state = CH9_NONE;
// ch9_state_t ch9_last_state = CH9_NONE;

ros::Publisher pub_takeoff;
ros::Publisher pub_track_trigger;
ros::Publisher pub_land_trigger;
// ros::Subscriber sub_rc;
ros::Subscriber sub_sim_rc;


void sim_rc_callback(const sim_msgs::sim_rccmdConstPtr& msg)  // 就少了个引用容易报错
{
    

    if(msg->sim_rccmd_ch9==sim_msgs::sim_rccmd::TAKEOFF)
    {
        ROS_WARN("[RCCMD] takeoff! \r\n");
        quadrotor_msgs::TakeoffLand takeoff_msg;
        // 给/px4ctrl/takeoff_land 发个1
        takeoff_msg.takeoff_land_cmd = quadrotor_msgs::TakeoffLand::TAKEOFF;
        pub_takeoff.publish(takeoff_msg);
    }
    else if(msg->sim_rccmd_ch9==sim_msgs::sim_rccmd::TRACK)
    {
        ROS_WARN("[RCCMD] track!\r\n");
        // 给/triger发个消息
        geometry_msgs::PoseStamped track_msg;
        pub_track_trigger.publish(track_msg);
    }
    else if(msg->sim_rccmd_ch9==sim_msgs::sim_rccmd::LAND)
    {
        ROS_WARN("[RCCMD] land on the car!\r\n");
        // 给/land_triger发个消息
        geometry_msgs::PoseStamped track_msg;
        pub_land_trigger.publish(track_msg);
    }
    else if(msg->sim_rccmd_ch9==LAND_STAIGHT_DOWNWARD)
    {
        ROS_WARN("[RCCMD] land straight downward!\r\n");
        // 给/land_triger发个消息
        quadrotor_msgs::TakeoffLand takeoff_msg;
        takeoff_msg.takeoff_land_cmd = quadrotor_msgs::TakeoffLand::LAND;
        pub_takeoff.publish(takeoff_msg);
    }
    else
    {
        ROS_WARN("[RCCMD] not a valid ch9 rc command ... ");
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "tfluna_node");
    ros::NodeHandle n("~");
    node_param_init(n);

    ros::Rate loop_rate(looprate);  // 手动指定更新周期

    // 发布，所以直接发在这里就能控制他的行为
    pub_takeoff = n.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 10); // 用来起飞
    pub_track_trigger = n.advertise<geometry_msgs::PoseStamped>("/triger", 10);
    pub_land_trigger = n.advertise<geometry_msgs::PoseStamped>("/land_triger", 10);

    // 接受遥控器输入
    // 不写<消息类型>也性，或者callback中要加引用
    sub_sim_rc=n.subscribe<sim_msgs::sim_rccmd>("/sim_cmd/rc/in", 1, &sim_rc_callback, ros::TransportHints().tcpNoDelay());

    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep(); 
    }


    return 0;
}




