
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <iostream>
#include <quadrotor_msgs/PolyTraj.h>
#include <Eigen/Eigen>
#include <traj_opt/poly_traj_utils.hpp>
#include"gazebo_msgs/GetModelState.h"



//**************** 全局变量 **************//
//* 可视化参数
double _vis_traj_width =0.1;
ros::Publisher _vel_traj_vis_pub;
ros::Publisher _acc_traj_vis_pub;
ros::Publisher real_traj_vis_pub;
ros::Publisher real_traj_line_vis_pub;
ros::Subscriber traj_msg_sub;
ros::Subscriber robot_state_sub; // 获取飞机的位置
visualization_msgs::Marker pos_marker;
visualization_msgs::Marker vel_marker;
visualization_msgs::Marker acc_marker;
visualization_msgs::Marker real_pos_marker;
visualization_msgs::Marker real_traj_marker;

quadrotor_msgs::PolyTraj trajMsg;
ros::ServiceClient state_client;
int real_traj_vis_hz = 10;
ros::Timer get_state_timer;


// 类型定义
typedef Eigen::Matrix<double, 3, 6> CoefficientMat;


//* 声明

// 获取规划出来的轨迹消息
void TrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr);
void setMarkerScale(visualization_msgs::Marker & marker,double x, double y ,double z);
void setMarkerColor(visualization_msgs::Marker & marker,double a,double r,double g,double b);
void setMarkerOrientation(visualization_msgs::Marker & marker,double x, double y ,double z,double w);
void setMarkerPosition(visualization_msgs::Marker & marker,double x,double y,double z);
void setMarkerArrowSize(visualization_msgs::Marker & marker,double size);
void setMarkerArrowOrientation(visualization_msgs::Marker & marker,Eigen::Vector3d start_pos,Eigen::Vector3d end_pos);

Eigen::Vector3d getPosPoly( Eigen::MatrixXd polyCoeff, double t );
void real_state_timer_callback(const ros::TimerEvent& event);




int main(int argc, char *argv[])
{
    /* code */
    ros::init(argc,argv,"traj_vis_node");
    ros::NodeHandle nh;

    // 订阅轨迹
    traj_msg_sub = nh.subscribe("/drone0/trajectory", 5, &TrajCallback); //订阅规划出来的多项式

    // 发布消息
    _vel_traj_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("traj_vis_vel",1);
    _acc_traj_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("traj_vis_acc",1);

    get_state_timer = nh.createTimer(ros::Duration(1.0 / real_traj_vis_hz), &real_state_timer_callback);
    state_client=nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

    real_traj_vis_pub  = nh.advertise<visualization_msgs::Marker>("traj_vis_real",1);
    real_traj_line_vis_pub  = nh.advertise<visualization_msgs::Marker>("traj_vis_real_line",1);


    // 初始化marker
    // real traj
    real_pos_marker.header.stamp = ros::Time::now();
    real_pos_marker.header.frame_id = "world";
    real_pos_marker.ns="traj_node/trajectory_waypoints";
    real_pos_marker.id = 0;
    real_pos_marker.type = visualization_msgs::Marker::SPHERE;
    real_pos_marker.action = visualization_msgs::Marker::ADD;
    setMarkerScale(real_pos_marker, 0.02,0.02,0.02);
    setMarkerColor(real_pos_marker,1,1,0.5,0 ); // orange

    real_traj_marker.header.stamp = ros::Time::now();
    real_traj_marker.header.frame_id = "world";
    real_traj_marker.ns="traj_node/real_traj";
    real_traj_marker.id = 0;
    real_traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
    real_traj_marker.action = visualization_msgs::Marker::ADD;
    setMarkerScale(real_traj_marker, 0.02,0.02,0.02);
    setMarkerColor(real_traj_marker,1,0,1,0 ); // green

    real_traj_marker.points.clear();


    // velocity
    vel_marker.header.stamp = ros::Time::now();
    vel_marker.header.frame_id = "world";
    vel_marker.ns="traj_node/trajectory_waypoints";
    vel_marker.id=0;
    vel_marker.type = visualization_msgs::Marker::ARROW;
    setMarkerColor(vel_marker,1,0,1,0); // green
    setMarkerArrowSize(vel_marker,0.5);


    // 功能1 发布速度和加速度轨迹
    //     traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("trajectory", 1); // 发布轨迹消息给traj server，traj_pub_

    // 功能2 订阅目标位置，直接在planning_nodelet上实现吧
    ros::spin();

}

void real_state_timer_callback(const ros::TimerEvent& event)
{
    //TODO 获取位置

    gazebo_msgs::GetModelState srv;
    srv.request.model_name = "iris"; //指定要获取的机器人在gazebo中的名字；
    if (state_client.call(srv))
    {
        //TODO 发布marker

        
        setMarkerPosition(real_pos_marker, srv.response.pose.position.x, srv.response.pose.position.y, srv.response.pose.position.z);
        real_pos_marker.id +=1;
        real_traj_vis_pub.publish(real_pos_marker);


        geometry_msgs::Point pos;
        pos.x=srv.response.pose.position.x;
        pos.y=srv.response.pose.position.y;
        pos.z=srv.response.pose.position.z;
        real_traj_marker.points.push_back(pos);
        real_traj_line_vis_pub.publish(real_traj_marker);
        
        // ROS_INFO("publish real pos !");
    }
    else
    {
        ROS_ERROR("Failed to call service /gazebo/get_model_state for iris");
    }

}



void TrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr) {
    trajMsg = *msgPtr; // 更新轨迹消息
    
    ROS_WARN("[traj_vis] traj_msg received");

    // 根据轨迹和时间获取位置
    int piece_nums = trajMsg.duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<CoefficientMat> cMats(piece_nums);
        for (int i = 0; i < piece_nums; ++i) {
      int i6 = i * 6;
      cMats[i].row(0) << trajMsg.coef_x[i6 + 0], trajMsg.coef_x[i6 + 1], trajMsg.coef_x[i6 + 2],
          trajMsg.coef_x[i6 + 3], trajMsg.coef_x[i6 + 4], trajMsg.coef_x[i6 + 5];
      cMats[i].row(1) << trajMsg.coef_y[i6 + 0], trajMsg.coef_y[i6 + 1], trajMsg.coef_y[i6 + 2],
          trajMsg.coef_y[i6 + 3], trajMsg.coef_y[i6 + 4], trajMsg.coef_y[i6 + 5];
      cMats[i].row(2) << trajMsg.coef_z[i6 + 0], trajMsg.coef_z[i6 + 1], trajMsg.coef_z[i6 + 2],
          trajMsg.coef_z[i6 + 3], trajMsg.coef_z[i6 + 4], trajMsg.coef_z[i6 + 5];

    dura[i] = trajMsg.duration[i];
    }

    Trajectory traj(dura,cMats);
    auto duration = traj.getTotalDuration(); // 获取总时长

    Eigen::Vector3d p, v, a; // 获取当前时间的p,v,a,j

   
    visualization_msgs::Marker clear_previous_msg;
    clear_previous_msg.action=visualization_msgs::Marker::DELETEALL;  // 删除所有arrow

    vel_marker.type = visualization_msgs::Marker::ARROW;
    vel_marker.action = visualization_msgs::Marker::ADD;
    vel_marker.header.frame_id = "world";
    vel_marker.id = 0;
    vel_marker.points.resize(2);
    setMarkerPosition(vel_marker, 0, 0, 0);
    setMarkerScale(vel_marker, 0.02,0.05,0.01);
    setMarkerColor(vel_marker,1,0,1,0 ); // green

    acc_marker.type=visualization_msgs::Marker::ARROW;
    acc_marker.action=visualization_msgs::Marker::ADD;
    acc_marker.header.frame_id="world";
    acc_marker.id=0;
    acc_marker.points.resize(2);
    setMarkerPosition(acc_marker, 0, 0, 0);
    setMarkerScale(acc_marker, 0.02,0.05,0.01);
    setMarkerColor(acc_marker,1,1,0,0 ); // red


    visualization_msgs::MarkerArray vel_list_msg;
    vel_list_msg.markers.clear();
    vel_list_msg.markers.push_back(clear_previous_msg);

    visualization_msgs::MarkerArray acc_list_msg;
    acc_list_msg.markers.clear();
    acc_list_msg.markers.push_back(clear_previous_msg);

    for (double t = 0; t < duration; t += 0.3) {   
        // 在这里进行可视化
        // path.push_back(traj.getPos(t)); // 按时间获取获取点
        p=traj.getPos(t); //TODO 位置还没写，不用写
        v=traj.getVel(t);
        a=traj.getAcc(t);
        ROS_WARN("test p : %f ",p.x());
        ROS_WARN("test v : %f ",v.x());


        vel_marker.points[0].x=p.x();
        vel_marker.points[0].y=p.y() - 1.0;  // NOTE : 在旁边可视化出来
        vel_marker.points[0].z=p.z();
        vel_marker.points[1].x= p.x()+ 0.05 * v.x();
        vel_marker.points[1].y= p.y()+ 0.05 * v.y() - 1.0;
        vel_marker.points[1].z= p.z()+ 0.05 * v.z();

        acc_marker.points[0].x=p.x();
        acc_marker.points[0].y=p.y() - 2.0;
        acc_marker.points[0].z=p.z();
        acc_marker.points[1].x= p.x()+ 0.05 * a.x();
        acc_marker.points[1].y= p.y()+ 0.05 * a.y() - 2.0;
        acc_marker.points[1].z= p.z()+ 0.05 * a.z();
        
        vel_list_msg.markers.push_back(vel_marker);
        acc_list_msg.markers.push_back(acc_marker);
        vel_marker.id+=1;
        acc_marker.id+=1;
        // vel_marker.points.clear();

    }
    _vel_traj_vis_pub.publish(vel_list_msg);
    ROS_WARN("[traj_vis] vel_list_msg published");

    _acc_traj_vis_pub.publish(acc_list_msg);
    ROS_WARN("[traj_vis] acc_list_msg published");
}


// marker处理工具
void setMarkerScale(visualization_msgs::Marker & marker,double x, double y ,double z)
{
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = "world";
    marker.scale.x=x;
    marker.scale.y=y;
    marker.scale.z=z;
}

void setMarkerOrientation(visualization_msgs::Marker & marker,double x, double y ,double z,double w)
{
    marker.pose.orientation.x=x;
    marker.pose.orientation.y=y;
    marker.pose.orientation.z=z;
    marker.pose.orientation.w=w;
}


void setMarkerPosition(visualization_msgs::Marker & marker,double x,double y,double z)
{
    marker.pose.position.x=x;
    marker.pose.position.y=y;
    marker.pose.position.z=z;
}

void setMarkerColor(visualization_msgs::Marker & marker,double a,double r,double g,double b)
{
    marker.color.a=a;
    marker.color.g=g;
    marker.color.r=r;
    marker.color.b=b;
}

void setMarkerArrowSize(visualization_msgs::Marker & marker,double size)
{
    marker.scale.x = size * 0.5;
    marker.scale.y = size * 1;
    marker.scale.z = 0.5; 
}

void setMarkerArrowOrientation(visualization_msgs::Marker & marker,Eigen::Vector3d start_pos,Eigen::Vector3d end_pos)
{
    marker.points.clear();
    geometry_msgs::Point start_pt,end_pt;
    start_pt.x=start_pos.x();
    start_pt.y=start_pos.y();
    start_pt.z=start_pos.z();
    end_pt.x=end_pos.x();
    end_pt.y=end_pos.y();
    end_pt.z=end_pos.z();
    marker.points.push_back(start_pt);
    marker.points.push_back(end_pt);
}
