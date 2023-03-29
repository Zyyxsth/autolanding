/*
 * @Author: Jimazeyu
 * @Date: 2022-02-18 03:19:20
 * @LastEditors: OBKoro1
 * @LastEditTime: 2022-02-18 04:38:48
 * @FilePath: /catkin_ws/src/map_tutorials/src/pose_publisher.cpp
 */
#include"ros/ros.h"
#include"iostream"
#include"gazebo_msgs/GetModelState.h"
#include"geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
using namespace std;


//*********************
//小车base和apriltag的相对位置
// At time 0.000
// - Translation: [0.081, 0.000, 0.265]
// - Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
//             in RPY (radian) [0.000, -0.000, 0.000]
//             in RPY (degree) [0.000, -0.000, 0.000]


int main(int argc,char**argv)
{
    ros::init(argc, argv, "true_pose_publisher");
    ros::NodeHandle n("~");
    
    int pub_rate=100;
    n.getParam("/vins_fusion/true_odom_pub_rate", pub_rate);
    ros::Rate loop_rate(pub_rate);

    //获取机器人名字
    string robot_name;
    n.param<std::string>("robot_name", robot_name, "iris");
    cout<<"true odom publish : "<<robot_name<<endl;
    //用于发布机器人真实坐标
    // ros::Publisher pose_publisher=n.advertise<geometry_msgs::PoseStamped>("true_pose",10);
    ros::Publisher odom_publisher=n.advertise<nav_msgs::Odometry>("/vins_fusion/imu_propagate",10);
    ros::Publisher car_odom_publisher = n.advertise<nav_msgs::Odometry>("husky_odom",10);
    ros::Publisher apritag_odom_publisher = n.advertise<nav_msgs::Odometry>("apriltag_odom",10);

    //用于获得真实坐标
    ros::ServiceClient client=n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    int seq=0;//发布序号
    while(ros::ok())
    {
        gazebo_msgs::GetModelState srv;
        srv.request.model_name = robot_name; //指定要获取的机器人在gazebo中的名字；
        if (client.call(srv))
        {
            nav_msgs::Odometry odom;
            odom.header.frame_id="world";
            odom.header.seq=seq++;
            odom.header.stamp=ros::Time::now();
            odom.pose.pose.position=srv.response.pose.position;
            odom.pose.pose.orientation=srv.response.pose.orientation;
            odom.twist.twist.linear=srv.response.twist.linear;
            odom.twist.twist.angular=srv.response.twist.angular;
            odom_publisher.publish(odom);

            // geometry_msgs::PoseStamped pose;
            // pose.header.frame_id="odom";
            // pose.header.seq=seq++;
            // pose.header.stamp=ros::Time::now();
            // pose.pose=srv.response.pose;
            // pose_publisher.publish(pose);
        }
        else
        {
            ROS_ERROR("Failed to call service /gazebo/get_model_state for iris");
        }
        srv.request.model_name="husky";
        if(client.call(srv))
        {
            nav_msgs::Odometry odom;
            odom.header.frame_id="world";
            odom.header.seq=seq++;
            odom.header.stamp=ros::Time::now();
            odom.pose.pose.position=srv.response.pose.position;
            odom.pose.pose.orientation=srv.response.pose.orientation;
            odom.twist.twist.linear=srv.response.twist.linear;
            odom.twist.twist.angular=srv.response.twist.angular;
            car_odom_publisher.publish(odom);
            odom.pose.pose.position.x=srv.response.pose.position.x+0.081;
            odom.pose.pose.position.z=srv.response.pose.position.z+0.265;
            apritag_odom_publisher.publish(odom);
        }
        else{
           ROS_ERROR("Failed to call service /gazebo/get_model_state for husky");

        }
        loop_rate.sleep();
    }

    return 0;
}
