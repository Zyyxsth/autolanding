/**
 * @file planning_nodelet.cpp
 * @author your name (you@domain.com)
 * @brief 自动降落的主要代码
 * @version 0.1
 * @date 2023-02-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include <geometry_msgs/PoseStamped.h>
#include <mapping/mapping.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <nodelet/nodelet.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/ReplanState.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/Bool.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <traj_opt/traj_opt.h>

#include <Eigen/Core>
#include <atomic>
#include <env/env.hpp>
#include <prediction/prediction.hpp>
#include <thread>
#include <visualization/visualization.hpp>
#include <wr_msg/wr_msg.hpp>
#include <std_msgs/Int32MultiArray.h> 
#include "nlink_parser/TofsenseFrame0.h"  //distance detection

// OBVP
#include "RapidTrajectoryGenerator.h"
// using namespace OBVP;

// 任务
#include <std_msgs/String.h>
// #include "sim_node/task_state.h"
#include <sim_msgs/task_state.h>

sim_msgs::task_state task_state_msg;              //任务状态的信息
sim_msgs::task_state set_task_msg;               //仿真时的任务状态信息
std::atomic_flag set_task_lock_ = ATOMIC_FLAG_INIT; // clear状态,既不是true，也不是false


// log added
// 任务流程状态机
// Finite State Machine 
enum TASK_STATE{
    NOT_READY = 0, 
    READY , // 起飞结束切换到READY
    TRACK  , // 跟踪
    LANDING, // 过渡
    CALI , // 末端校正
    DEBUG , // 用来debug
    LANDED,
  };
// 任务状态
TASK_STATE task_state = NOT_READY;


namespace planning {

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

class Nodelet : public nodelet::Nodelet {     //节点
 private:
  std::thread initThread_;
  ros::Subscriber gridmap_sub_, odom_sub_, target_sub_, triger_sub_, land_triger_sub_;
  ros::Subscriber laser_sub_;
  ros::Timer plan_timer_;
  ros::Timer ctrl_timer_; // 控制降落
  ros::Publisher ctrl_heartbeat_pub_;

  ros::Publisher traj_pub_, heartbeat_pub_, replanState_pub_;
  ros::Publisher relay_pub_, motors_pub_;   // relay 继电器

  ros::Publisher task_state_pub_;     //任务的发布者
  ros::Publisher cali_pos_pub_;
  ros::Subscriber set_task_sub_;      //任务的接收者




  double last_yaw_;

  std::shared_ptr<mapping::OccGridMap> gridmapPtr_;                       //shared_ptr共享指针，地图的指针
  std::shared_ptr<env::Env> envPtr_;
  std::shared_ptr<visualization::Visualization> visPtr_;                //可视化指针，用于检测无人机和车辆之间有没有障碍物
  std::shared_ptr<traj_opt::TrajOpt> trajOptPtr_;                       //路径的指针
  std::shared_ptr<prediction::Predict> prePtr_;                          //预测的指针

  // NOTE planning for fake target
  bool fake_ = false;
  Eigen::Vector3d goal_;                       //目标位置；
  Eigen::Vector3d land_p_;                   //Eigen：Vector3d 指的是三维向量，land_p_应该是降落的位置
  Eigen::Quaterniond land_q_;              //Eigen::Quaterniond指的是四元数，land_q_应该指的是降落下的姿态

  // NOTE just for debug
  bool debug_ = false;
  quadrotor_msgs::ReplanState replanStateMsg_;              //quadrotor应该就是四旋翼的一种类型
  ros::Publisher gridmap_pub_, inflate_gridmap_pub_;       //gridmap_pub_  应该就是建图的信息发布结点   inflate_gridmap_pub_ 讲的是膨胀后的建图信息发布
  quadrotor_msgs::OccMap3d occmap_msg_;                      //occ应该指的是occlusion  遮挡信息，根据p122那么应该指的是遮挡地图的信息。

  double tracking_dur_, tracking_dist_, tolerance_d_;     //torlerance_d 能够允许的最小的距离
  double tracking_z_; //跟踪的高度


  Trajectory traj_poly_; // 上一次的轨迹,不随着循环而消失，所以设置成全局变量   Trajectory是由pieces组成的。
  ros::Time replan_stamp_; // 规划的时间
  int traj_id_ = 0;                   //轨迹段的编号
  bool wait_hover_ = true;        //hover悬停,等待悬停标志
  bool force_hover_ = true;        //强制悬停标志
  bool landing_finished_ = false;          //降落成功标志

  nav_msgs::Odometry odom_msg_, target_msg_;        //里程计上的信息，odom_msg_应该是指的飞机里程计上的信息， target_msg_应该指的是目标物里程计上的信息。
  quadrotor_msgs::OccMap3d map_msg_; // 栅格地图参数以及数据（一个数组）

// 看成布尔类型即可
  std::atomic_flag odom_lock_ = ATOMIC_FLAG_INIT; // clear状态,既不是true，也不是false
  std::atomic_flag target_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag gridmap_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_bool odom_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool map_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool triger_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool target_received_ = ATOMIC_VAR_INIT(false);              //车辆的信息
  std::atomic_bool land_triger_received_ = ATOMIC_VAR_INIT(false);         //表明是否收到降落的信息
  std::atomic_bool trigger_cali_ctrl_flag = ATOMIC_VAR_INIT(false);

  // 启动继电器
  void turn_on_relay() {
    std_msgs::Bool msg;   //例如ros入门教程中的节点talker和listener之间通过话题”chatter”来传递数据流，而数据流的具体数据类型则是std_msgs/String。
    msg.data = true;
    relay_pub_.publish(msg); 
    return;
  }

  // 停止电机
  void stop_motors() {
    quadrotor_msgs::TakeoffLand msg;
    msg.takeoff_land_cmd = msg.KILLMOTOR;
    motors_pub_.publish(msg);
    return;
  }

  // 在降落过程中
  void in_landing_process() {
    quadrotor_msgs::TakeoffLand msg;
    msg.takeoff_land_cmd = msg.TRUST_ADJUST_BY_LASER;
    motors_pub_.publish(msg); // 这里应该就交给px4_ctrl控制了
    return;
  }

  // 悬停
  void pub_hover_p(const Eigen::Vector3d& hover_p, const ros::Time& stamp) {        //stamp指的是开始悬停的时间
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = true;
    traj_msg.hover_p.resize(3);                                                 //这个hover_p应该是悬停的位置
    for (int i = 0; i < 3; ++i) {
      traj_msg.hover_p[i] = hover_p[i];
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;                                              //然后到下一条轨迹，所以轨迹的id++。
    traj_pub_.publish(traj_msg);
  }


  void pub_traj(const Trajectory& traj, const double& yaw, const ros::Time& stamp) {    //发布轨迹
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = false;
    traj_msg.order = 5;                  //这个order是什么还没看懂
    Eigen::VectorXd durs = traj.getDurations();                  //Eigen::VectorXd是x维的向量
    int piece_num = traj.getPieceNum();                          //应该是traj分成了几段的数量
    traj_msg.duration.resize(piece_num); // 每段一个duration
    traj_msg.coef_x.resize(6 * piece_num);  // 重定义维度   coefficient系数
    traj_msg.coef_y.resize(6 * piece_num);
    traj_msg.coef_z.resize(6 * piece_num);
    for (int i = 0; i < piece_num; ++i) {
      traj_msg.duration[i] = durs(i);          //把traj的每一段赋值给traj_msg
      CoefficientMat cMat = traj[i].getCoeffMat();                   //这段系数是什么意思暂时还没看懂
      int i6 = i * 6;
      for (int j = 0; j < 6; j++) {
        traj_msg.coef_x[i6 + j] = cMat(0, j);
        traj_msg.coef_y[i6 + j] = cMat(1, j);
    traj_msg.coef_z.resize(6 * piece_num);
    for (int i = 0; i < piece_num; ++i) {
      traj_msg.duration[i] = durs(i);
      CoefficientMat cMat = traj[i].getCoeffMat();
      int i6 = i * 6;
      for (int j = 0; j < 6; j++) {
        traj_msg.coef_x[i6 + j] = cMat(0, j);
        traj_msg.coef_y[i6 + j] = cMat(1, j);
        traj_msg.coef_z[i6 + j] = cMat(2, j);
      }
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    // NOTE yaw
    traj_msg.yaw = yaw;                                  //轨迹的角度
    traj_pub_.publish(traj_msg);
  }
    }}

  void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    goal_ << msgPtr->pose.position.x, msgPtr->pose.position.y, 0.9;
    triger_received_ = true;
  }

  void set_task_callback(const sim_msgs::task_stateConstPtr& msgPtr)   //输入的是一个标志着任务状况的指针msgPtr
  {
    while(set_task_lock_.test_and_set()) ;
    set_task_msg = *msgPtr;

    std::cout<<"setting state: "<< set_task_msg.state<<std::endl;

    if(set_task_msg.state=="NOT_READY")
      set_task_msg.state_id=sim_msgs::task_state::TASK_NOT_READY;
    else if(set_task_msg.state=="READY")
      set_task_msg.state_id=sim_msgs::task_state::TASK_READY;
    else if(set_task_msg.state=="TRACK")
      set_task_msg.state_id=sim_msgs::task_state::TRACK;
    else if(set_task_msg.state=="LANDING")
      set_task_msg.state_id=sim_msgs::task_state::LANDING;
    else if(set_task_msg.state=="LANDED")
      set_task_msg.state_id=sim_msgs::task_state::LANDED;
    else if(set_task_msg.state=="DEBUG")
      set_task_msg.state_id=sim_msgs::task_state::DEBUG;
    else if(set_task_msg.state=="CALI")                     //校准
      set_task_msg.state_id=sim_msgs::task_state::CALIBRATION;
    else
      set_task_msg.state_id=8;


    switch(set_task_msg.state_id)                        //根据修改后的set_task_msg.state_id的信息去修改人物的状态
    {
      case sim_msgs::task_state::TASK_NOT_READY:
        task_state=NOT_READY;
        break;
      case sim_msgs::task_state::TASK_READY:
        task_state=READY;
        break;
      case sim_msgs::task_state::TRACK:
        task_state=TRACK;
        break;
      case sim_msgs::task_state::LANDING:
        task_state=LANDING;
        break;
      case sim_msgs::task_state::CALIBRATION:
        task_state=CALI;
        break;
      case sim_msgs::task_state::LANDED:
        task_state=LANDED;
        break;
      case sim_msgs::task_state::DEBUG:
        task_state=DEBUG;
        break;
      default:
        ROS_ERROR("NOT A VALID TASK STATE!!!!");
        break;
    }

  
    set_task_lock_.clear();
  }

  // 确认降落的位姿
  void land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    land_p_.x() = msgPtr->pose.position.x;                              //降落的位置land_p_    p103行
    land_p_.y() = msgPtr->pose.position.y;
    land_p_.z() = msgPtr->pose.position.z;
    land_q_.w() = msgPtr->pose.orientation.w;                           //降落的姿态
    land_q_.x() = msgPtr->pose.orientation.x;
    land_q_.y() = msgPtr->pose.orientation.y;
    land_q_.z() = msgPtr->pose.orientation.z;
    land_triger_received_ = true;
  }

  void odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (odom_lock_.test_and_set()) // 启动锁
      ;
    odom_msg_ = *msgPtr;                                     //指的就是无人机的位置信息
    odom_received_ = true;
    odom_lock_.clear(); // 清除里程计锁，
  }

  // 
  void target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (target_lock_.test_and_set())
      ;
    target_msg_ = *msgPtr;                                     //指的就是车辆的位置信息
    target_received_ = true;
    target_lock_.clear();
  }


  int dis_laser_mm = 10000; // 初始化100m
  // void laser_callback(const std_msgs::Int32MultiArray::ConstPtr& msgPtr) 
  // {
  //   dis_laser_mm = msgPtr->data[0];
  //   // printf("planner   laser = %d mm\r\n", dis_laser_mm);
  // }

  // 这里要改成 sensor_msgs::
  void sim_laser_callback(const sensor_msgs::LaserScanConstPtr& pMsg)                   //应该一个是仿真用的，一个是实际的用的
  {
    // dis_laser_mm = (int)(pMsg->dis*1000);；  // 用mm表示
    dis_laser_mm = (int)(pMsg->ranges[0]*1000);  // 用mm表示 // 取出第一个数作为激光测距距离
    // printf("planner   laser = %d mm\r\n", dis_laser_mm);
  }
  void laser_callback(const nlink_parser::TofsenseFrame0ConstPtr& pMsg) 
  {
    dis_laser_mm = (int)(pMsg->dis*1000);;  // 用mm表示
    // printf("planner   laser = %d mm\r\n", dis_laser_mm);
  }

  // 栅格地图获取
  void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
    while (gridmap_lock_.test_and_set())
      ;
    map_msg_ = *msgPtr;                            //就是将msgPtr中的栅格地图数据赋值给变量map_msg_     p122
    map_received_ = true;
    gridmap_lock_.clear();
  }


// 重构
// 先把landing和track的部分分开写
// 然后进行
//                       system start
//                            |
//                            |
//                            v
// ------------------- >  NOT_READY 
// |                          |                 DEBUG
// |                          |                   ^
// |                  TAKEOFF |                   |
// |                          |                   |
// |                          |                   |
// |                          v                   |
// |            ------------READY ----------------|
// |           |              |                   | force hover
// |           |              |                   |
// |   setting | tracking     |  setting landing  | 
// |           v              v                   |
// -------- TRACKING  --->  LANDING --------------
//                            |
//                            |
//                            v
//                           CALI  ---> LANDED


bool landed_flag = false;

void ctrl_timer_callback(const ros::TimerEvent& event)
{
    ctrl_heartbeat_pub_.publish(std_msgs::Empty());
    if(!trigger_cali_ctrl_flag)
      return ;
  
   // 获得目标以及target的odom
    //! step 1 : obtain state of odom
    while (odom_lock_.test_and_set()) ;
    auto odom_msg = odom_msg_; 
    odom_lock_.clear();                                                          //下面指的是无人机的相关信息
    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,                       //pose.pose.position.x分别指的就是odom坐标系下x,y,z的位置
                          odom_msg.pose.pose.position.y,
                          odom_msg.pose.pose.position.z);
    Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,                        //Twist.twist.linear.x分别指的就是odom坐标系下x,y,z的速度
                          odom_msg.twist.twist.linear.y,
                          odom_msg.twist.twist.linear.z);
    Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,                 //pose.pose.orientation.w分别指的就是odom坐标系下的四元数
                              odom_msg.pose.pose.orientation.x,
                              odom_msg.pose.pose.orientation.y,
                              odom_msg.pose.pose.orientation.z);

    //! step 2 : obtain state of target 
    while (target_lock_.test_and_set())
      ;
    replanStateMsg_.target = target_msg_; 
    target_lock_.clear();

    Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,          //这个指的是车辆的位置和速度
                            replanStateMsg_.target.pose.pose.position.y,
                            replanStateMsg_.target.pose.pose.position.z);
    Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                            replanStateMsg_.target.twist.twist.linear.y,
                            replanStateMsg_.target.twist.twist.linear.z);
    Eigen::Quaterniond target_q;
    target_q.w() = replanStateMsg_.target.pose.pose.orientation.w;
    target_q.x() = replanStateMsg_.target.pose.pose.orientation.x;
    target_q.y() = replanStateMsg_.target.pose.pose.orientation.y;
    target_q.z() = replanStateMsg_.target.pose.pose.orientation.z;

    
    // 设置target位置为匀速直线运动之后的时间 ，加一点点的预测
    Eigen::Vector3d cali_point = target_p + target_v * 0.01;                         //这是设置的车辆的移动位置，这里就是按照匀速直线规定的，沿着（1， 1，1）的方向。
    target_p=cali_point;

    static Eigen::Vector3d cali_p,cali_v;                                     //校准时的位置和速度

    static bool first_flag=true,not_stable_over_head_flag=true;



    Eigen::Vector3d delta_p=odom_p-target_p;                                      //无人机和车辆的距离差
    Eigen::Vector3d delta_v=odom_v-target_v;                                      //无人机和车辆的速度差

    // judge if landed
    if(delta_p.norm()<=0.1)                                   //sqrt(delta_x^2+delta_y^2+delta_z^2) 指的就是就是距离。
    {
      landed_flag=true;
      ROS_WARN("[ planner ] CALI stage : landed success!!!");
      stop_motors();                                           //p144 停止电机
      return ;
    }


    // if 正上方 稳定跟踪
    delta_p.z()=0;
    if(delta_p.norm()<=0.3 && delta_v.norm()<=0.1){                          //说明在上面校准的很好，速度也差不多
      not_stable_over_head_flag=false;
    }

    if(!not_stable_over_head_flag)
        ROS_WARN("[ planner ] CALI stage : STABLE OVER HEAD  !!! ");

    if(first_flag || not_stable_over_head_flag )
    {
      cali_p.x()=target_p.x();
      cali_p.y()=target_p.y();
      cali_p.z()=odom_p.z();
      cali_v.x()=target_v.x();
      cali_v.y()=target_v.y();
      //TODO yaw 设置为小车的方向
      cali_v.z()=0;
      first_flag = false;
    }
    else
    {
      Eigen::Vector3d down_speed(0,0,-0.005);
      cali_p.x()=target_p.x();
      cali_p.y()=target_p.y();
      // cali_p.z()-=cali_p.z()-0.0005;
      cali_p=cali_p+down_speed;                                    //假如校准较高，就慢慢往下降。
      cali_v.x()=target_v.x();
      cali_v.y()=target_v.y();
      int plan_hz = 20;
      // cali_v.z()=-0.0005 * plan_hz;
    }

    //! step publish the desired position and velocity
    quadrotor_msgs::PositionCommand position_cmd_msg;              //这里就是悬停时期望的无人机的飞行速度，和飞行位置
    position_cmd_msg.position.x=cali_p.x();
    position_cmd_msg.position.y=cali_p.y();
    position_cmd_msg.position.z=cali_p.z();
    position_cmd_msg.velocity.x=cali_v.x();
    position_cmd_msg.velocity.y=cali_v.y();
    position_cmd_msg.velocity.z=cali_v.z();

    ROS_WARN("[ planner ] CALI stage : position_cmd published");
    cali_pos_pub_.publish(position_cmd_msg);
}


//! 重点，由规划定时器触发的回调函数
  // NOTE main callback
void plan_timer_callback(const ros::TimerEvent& event) {                            //规划定时器的回调函数
    heartbeat_pub_.publish(std_msgs::Empty()); // planner heartbeat
    ROS_WARN_ONCE("[ planner ] waken up");
    if (!odom_received_ || !map_received_) { // no odom or no map planner not work           //如果没有收到里程计的信息或者没有收到建图消息的时候退出。
      ROS_INFO("[ planner ] no odom or no map received");
      return;
    }

    // 发布task消息
    task_state_msg.header.stamp=ros::Time::now();
    switch(task_state)
    {
      case NOT_READY:
        task_state_msg.state="NOT_READY";
        break;
      case READY:
        task_state_msg.state="REDAY";
        break;
      case TRACK:
        task_state_msg.state="TRACK";
        break;
      case LANDING:
        task_state_msg.state="LANDING";
        break;
      case CALI:
        task_state_msg.state="CALIBRATION";
        break;
      case LANDED:
        task_state_msg.state="LANDED";
        break;
      case DEBUG:
        task_state_msg.state="DEBUG";
        break;
      default:break;
    }
    task_state_pub_.publish(task_state_msg);
    // ROS_WARN("[ planner ] task state published");

    // if (!triger_received_) { // no track trigger
    //   ROS_INFO("[ planner ] watting for trigger for uav");
    //   task_state = NOT_READY;
    //   return;
    // }

    if (!target_received_) { // no target in view
      ROS_INFO_ONCE("[ planner ] watting for landing target");              //如果没有收到车辆的信息，那么就会返回等待降落移动平台的信息
      return;
    }
      ROS_INFO_ONCE("[ planner ] target in view");                          //如果收到了车辆的信息，那么就可以回应目标已经在视野中


    // 判断状态
    switch(task_state)
    {
      case NOT_READY:
      {
        task_state_msg.state="NOT_READY";
        ROS_INFO_ONCE("[planner] watting for trigger for uav, task state not ready");
        return ;
        break;
      }

      case READY:
      {
        task_state_msg.state="REDAY";
        ROS_INFO_ONCE("[[planner] planner is ready]");
        break;
      }

      case TRACK:
      {
        task_state_msg.state="TRACK";
        ROS_INFO_ONCE("[planner] planner is tracking");

        //! step 1 : obtain state of odom
        while (odom_lock_.test_and_set()) ;
        auto odom_msg = odom_msg_; 
        odom_lock_.clear();
        Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                              odom_msg.pose.pose.position.y,
                              odom_msg.pose.pose.position.z);
        Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                              odom_msg.twist.twist.linear.y,
                              odom_msg.twist.twist.linear.z);
        Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,
                                  odom_msg.pose.pose.orientation.x,
                                  odom_msg.pose.pose.orientation.y,
                                  odom_msg.pose.pose.orientation.z);

        //! step 2 : obtain state of target 
        while (target_lock_.test_and_set())
          ;
        replanStateMsg_.target = target_msg_; 
        target_lock_.clear();

        Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                                replanStateMsg_.target.pose.pose.position.y,
                                replanStateMsg_.target.pose.pose.position.z);
        Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                                replanStateMsg_.target.twist.twist.linear.y,
                                replanStateMsg_.target.twist.twist.linear.z);
        Eigen::Quaterniond target_q;
        target_q.w() = replanStateMsg_.target.pose.pose.orientation.w;
        target_q.x() = replanStateMsg_.target.pose.pose.orientation.x;
        target_q.y() = replanStateMsg_.target.pose.pose.orientation.y;
        target_q.z() = replanStateMsg_.target.pose.pose.orientation.z;

        // NOTE force-hover: waiting for the speed of drone small enough
        if (force_hover_ && odom_v.norm() > 0.1) {                 //如果想要强制悬停的话，需要按下强制悬停的按键force_hover_, 并且要把飞机的速度降到一个足够小的情况下
          return;
        }


        //! step 3 : setting tracking target 
          target_p.z() += tracking_z_;  // 跟踪时候距离小车的垂直距离, 这个tracking_z_不知道是什么意思，可能是车辆的高度

        // NOTE determin whether to replan
        Eigen::Vector3d dp = target_p - odom_p; // 距离目标的位置差向量
        double desired_yaw = std::atan2(dp.y(), dp.x()); // 朝向目标的期望的yaw角
        Eigen::Vector3d project_yaw = odom_q.toRotationMatrix().col(0);  // NOTE ZYX //! 旋转矩阵取第一列     把四元数转换为旋转矩阵，第一列为飞机在各个方向的yaw航向角
        double now_yaw = std::atan2(project_yaw.y(), project_yaw.x()); // 现在的yaw角，std::atan2(project_yaw.y(), project_yaw.x())

        // NOTE : too close
        if (std::fabs((target_p - odom_p).norm() - tracking_dist_) < tolerance_d_ && // 距离小车头顶的点太近了，悬停   而 fabs 函数用于对浮点数 float 或者 double 或者整数 int 取绝对值，也是一个比较常用的函数 ，语法如下：
            odom_v.norm() < 0.1 && target_v.norm() < 0.2 &&
            std::fabs(desired_yaw - now_yaw) < 0.5)  // 角度误差小于0.5    就是指飞机的航向角和车辆的前进方向一致的情况下
          { 
            if (!wait_hover_) {
              pub_hover_p(odom_p, ros::Time::now()); 
              wait_hover_ = true;
            }
            ROS_WARN("[planner] Too close ...  HOVERING...");
            replanStateMsg_.state = -1;
            replanState_pub_.publish(replanStateMsg_); 
            // return;
            break;
          } 
        else 
          {
              wait_hover_ = false; 
          }

        //! step 4 : obtain map 
        while (gridmap_lock_.test_and_set()) 
          ;
        gridmapPtr_->from_msg(map_msg_);
        replanStateMsg_.occmap = map_msg_; 
        gridmap_lock_.clear(); 
        prePtr_->setMap(*gridmapPtr_); 


        // visualize the ray from drone to target
        if (envPtr_->checkRayValid(odom_p, target_p)) { // 检查飞机和目标线段是否有障碍物
            visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::yellow);
        } else {
            visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::red);
          }

        //! step 5 : prediction
        std::vector<Eigen::Vector3d> target_predcit; // 预测路径
        // ros::Time t_start = ros::Time::now();
        bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit);        //前端优化，可能采用了混合A*算法之类的，得到的路径就是target_predict, generate_new_traj_success用来表示生成是否成功
        // 根据目标的为位置和速度得到预测的目标位置 放在target_predict
        if (generate_new_traj_success) {
          Eigen::Vector3d observable_p = target_predcit.back(); // 预测的最终位置
          visPtr_->visualize_path(target_predcit, "car_predict"); //预测路径可视化

        // 最终位置方圆tracking_dist的边缘画出来，可视边缘
        std::vector<Eigen::Vector3d> observable_margin;                  //这个东西可能就是论文中的那个黄色的圆圈
        for (double theta = 0; theta <= 2 * M_PI; theta += 0.01) {
          observable_margin.emplace_back(observable_p + tracking_dist_ * Eigen::Vector3d(cos(theta), sin(theta), 0));
        }
        visPtr_->visualize_path(observable_margin, "observable_margin");  // 在rviz看一下就知道是什么了
      }

        //! step 6 setting replan state                       //后端优化，重规划的状态
        Eigen::MatrixXd iniState;
        iniState.setZero(3, 3); // 三列，p,v,a         iniState 表示的是初始的位置，速度和加速度
        ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);    // 预测终点的时间戳
        double replan_t = (replan_stamp - replan_stamp_).toSec();           // 距离上次重规划的时间长度 replan_stamp_是上一次的时间戳

        // 之前是强制悬停 或者距离上一次规划的时间长度大于轨迹的长度时，当前状态应该是悬停，从悬停状态(当前状态)开始replan
        //  traj_poly_是上一次的轨迹
        if (force_hover_ || replan_t > traj_poly_.getTotalDuration()) {     // 之前是强制悬停 或者距离上一次规划的时间长度大于轨迹的长度时，当前状态应该是悬停，从悬停状态(当前状态)开始replan
          // should replan from the hover state
          iniState.col(0) = odom_p;
          iniState.col(1) = odom_v;
        } else {
          // should replan from the last trajectory
          iniState.col(0) = traj_poly_.getPos(replan_t);  // 从上一次的轨迹开始重规划，获取从上一次重规划经过若干点后的在轨迹上的状态
          iniState.col(1) = traj_poly_.getVel(replan_t);
          iniState.col(2) = traj_poly_.getAcc(replan_t);
        }
        replanStateMsg_.header.stamp = ros::Time::now();
        replanStateMsg_.iniState.resize(9); // 长度为9的数组，将一个3x3的矩阵，变成一个1x9的矩阵
        Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3) = iniState; // 赋值过去，将一个3x3的矩阵，变成一个1x9的矩阵

        //! step 7 : path searching
        Eigen::Vector3d p_start = iniState.col(0); // 开始位置
        std::vector<Eigen::Vector3d> path, way_pts; // 路径和waypoint，waypoint我就翻译成飞行路径上的一个个航点了

      if(generate_new_traj_success)   // 预测成功
        generate_new_traj_success = envPtr_->findVisiblePath(p_start, target_predcit, way_pts, path); //通过这个findVisiblePath函数，把预测的航点连接成一个path路径，

      std::vector<Eigen::Vector3d> visible_ps;   //可以看见的那个点kexi k，其实waypoint和visible_ps很多是重合的吧
      std::vector<double> thetas;               //应该就是论文里说的那个角
      Trajectory traj;            //最终生成的路径
      if (generate_new_traj_success) { // 路径搜索成功
        visPtr_->visualize_path(path, "astar"); 

        //! step 8 : generate visible regions
        target_predcit.pop_back(); // 删除尾端元素
        way_pts.pop_back();
        envPtr_->generate_visible_regions(target_predcit, way_pts, // 生成一个区域，waypoint变成可视范围的中间
                                          visible_ps, thetas);
  
        visPtr_->visualize_pointcloud(visible_ps, "visible_ps"); // 貌似是在追踪距离处能看到target的范围
        visPtr_->visualize_fan_shape_meshes(target_predcit, visible_ps, thetas, "visible_region"); // 生成扇形区

        // TODO change the final state
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
        for (int i = 0; i < (int)way_pts.size(); ++i) {
          rays.emplace_back(target_predcit[i], way_pts[i]); 
        }
        visPtr_->visualize_pointcloud(way_pts, "way_pts"); //其实waypoint和visible_ps很多是重合的吧
        way_pts.insert(way_pts.begin(), p_start); //在开头处插入一个起始坐标
        envPtr_->pts2path(way_pts, path); // waypoint用Atar连起来生成轨迹

        //! step 9 : 生成飞行走廊
        ROS_WARN("corridor generating");  //corridor 走廊
        // NOTE corridor generating
        std::vector<Eigen::MatrixXd> hPolys;    //飞行走廊，应该是指的多边形飞行走廊
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;     //飞行走廊上的关键点吧，但是为什么是两个vector3d不是很清楚，可能包括速度之类的

        envPtr_->generateSFC(path, 2.0, hPolys, keyPts); // 生成飞行走廊
        envPtr_->visCorridor(hPolys); // 可视化飞行走廊
        visPtr_->visualize_pairline(keyPts, "keyPts");

        //! step 10 : trajectory optimization
        ROS_WARN("trajectory optimization");
        Eigen::MatrixXd finState; // 末端状态
        finState.setZero(3, 3);
        finState.col(0) = path.back(); // 轨迹的终点
        finState.col(1) = target_v; // 和目标一样的速度
        // finState.col(1).z()+=-0.2; // 末端添加一个向下的速度
        generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState,
                                                               target_predcit, visible_ps, thetas,
                                                               hPolys, traj);    //这个在traj_opt/src/traj_opt.cc中 然后论文中的那些惩罚函数中也都在里面
        visPtr_->visualize_traj(traj, "traj");


      }
          //! final step 11: collision check
        bool valid = false;
        if (generate_new_traj_success) {
            valid = validcheck(traj, replan_stamp); // 从replan_stemp开始的1s内是否碰撞，已经是之后的事情了
        } else { // not generate successfully
          replanStateMsg_.state = -2;
          replanState_pub_.publish(replanStateMsg_);
        }
        if (valid) { // 从replan_stemp开始的1s内无碰撞
          force_hover_ = false; // not hover
          ROS_WARN("[planner] TRACKING REPLAN SUCCESS");
          replanStateMsg_.state = 0; // 
          replanState_pub_.publish(replanStateMsg_);
          Eigen::Vector3d dp = target_p + target_v * 0.03 - iniState.col(0); // 假设匀速直线运动的目标未来位置到飞机当前位置的位移
          // NOTE : if the trajectory is known, watch that direction
          // Eigen::Vector3d un_known_p = traj.getPos(1.0);
          // if (gridmapPtr_->isUnKnown(un_known_p)) {
          //   dp = un_known_p - odom_p;
          // }
          double yaw = std::atan2(dp.y(), dp.x()); // yaw角对准目标未来位置
          // yaw = 0;
          last_yaw_ = yaw;
          pub_traj(traj, yaw, replan_stamp); //发布轨迹，在这里发布轨迹
          traj_poly_ = traj; // 记录为上一次的轨迹
          replan_stamp_ = replan_stamp; // 记录为上一次的时间戳
        } // 生成的轨迹从replan开始1s内有碰撞 
        else if (force_hover_) { // 如果轨迹有问题，且有force_hover_，则悬停
          ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
          replanStateMsg_.state = 1;
          replanState_pub_.publish(replanStateMsg_);
          return;
        } else if (!validcheck(traj_poly_, replan_stamp_)) { // 当前轨迹有问题，检查上一个轨迹中在1s无碰撞 ，这里应该是写反了
          force_hover_ = true;
          ROS_FATAL("[planner] EMERGENCY STOP!!!"); // 紧急悬停
          replanStateMsg_.state = 2;
          replanState_pub_.publish(replanStateMsg_);
          pub_hover_p(iniState.col(0), replan_stamp);                 
          return;
      } else { // 上一次轨迹没问题
          ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ..."); // 检查生成轨迹上一个轨迹中在1s无碰撞
          replanStateMsg_.state = 3;
          replanState_pub_.publish(replanStateMsg_);
          return;  // current generated traj invalid but last is valid
      }
        visPtr_->visualize_traj(traj, "traj"); // 可视化轨迹，发布的话题名称为traj，对应航点为traj_wpts
        break;
      }
      case LANDING:
      {
        ROS_INFO("[planner] in landing process");

        //! step 1 : obtain state of robot odom
        while (odom_lock_.test_and_set()) ;
        auto odom_msg = odom_msg_; 
        odom_lock_.clear(); 

        // 构造函数，获取里程计的速度，位置和姿态，也就是当前的状态
        Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                              odom_msg.pose.pose.position.y,
                              odom_msg.pose.pose.position.z);
        Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                              odom_msg.twist.twist.linear.y,
                              odom_msg.twist.twist.linear.z);
        Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,
                                  odom_msg.pose.pose.orientation.x,
                                  odom_msg.pose.pose.orientation.y,
                                  odom_msg.pose.pose.orientation.z);


        //! step 2 : obtain state of target 
        while (target_lock_.test_and_set())
          ;
        replanStateMsg_.target = target_msg_; 
        target_lock_.clear();

        Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                                replanStateMsg_.target.pose.pose.position.y,
                                replanStateMsg_.target.pose.pose.position.z);
        Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                                replanStateMsg_.target.twist.twist.linear.y,
                                replanStateMsg_.target.twist.twist.linear.z);
        Eigen::Quaterniond target_q;
        target_q.w() = replanStateMsg_.target.pose.pose.orientation.w;
        target_q.x() = replanStateMsg_.target.pose.pose.orientation.x;
        target_q.y() = replanStateMsg_.target.pose.pose.orientation.y;
        target_q.z() = replanStateMsg_.target.pose.pose.orientation.z;

        if (force_hover_ && odom_v.norm() > 0.1) {
          break;
        }

        //! step 3 : judge if landed
        // static int count_laser = 0;
        // if( dis_laser_mm < 120 && dis_laser_mm > 5 ) // 如果底部激光探测到这个距离，说明接触到了物体，贴上去
        // {
        //   count_laser++;
        //   ROS_WARN("laser count++! count_laser = %d",count_laser);
        // }
        // else
        // {
        //   count_laser = 0; // reset
        // }

        // if (count_laser >= 2) {
        //   landing_finished_ = true;
        //   ROS_WARN("[planner] LANDING FINISHED!");
        // }

        // if (landing_finished_)  // landed
        // {
        //   stop_motors(); //发布停止电机的消息
        //   ROS_WARN("[planner] has landed at the landing position");
        //   task_state=LANDED;
        //   break;
        // }
        // else // landing process
        // {
        //   // log 注释掉降落过程
        //   in_landing_process(); // 将motor的状态改为TRUST_ADJUST_BY_LASER
        // }


      // 足够近了，切换到cali模式
      Eigen::Vector3d landing_dp=target_p-odom_p;
      Eigen::Vector3d landing_dv=target_v-odom_v;
      landing_dp.z()=0;
      landing_dv.z()=0;
      // if(landing_dp.norm()<0.1 && landing_dv.norm()<target_v.norm()*0.3) // 相对速度小于车的速度乘上比例项
      if(landing_dp.norm()<0.2)  //当距离足够近的时候，切换到cali模式
      {
          task_state=CALI;
          ROS_WARN("[ planner ] LANDING stage : switch to CALI !!!");
          return ;
      }


      // 足够近了，悬停
        // if (std::fabs((target_p - odom_p).norm() < 0.1 && odom_v.norm() < 0.1 && target_v.norm() < 0.2)) {
        //   if (!wait_hover_) {
        //     pub_hover_p(odom_p, ros::Time::now());
        //     wait_hover_ = true; // 正在悬停
        //   }
        //   ROS_WARN("[planner] close enough ...  HOVERING...");
        //   break;
        // }

        // TODO get the orientation fo target and calculate the pose of landing point
        // target_p = target_p + target_q * land_p_; // 降落时的轨迹的终点位置，land_p_是相对于车的降落点，只有faketarget时才有后两项
        wait_hover_ = false; // 还没足够进，不要悬停

        target_p.z()+= 1; // 飞到上面1米处

        //! step 4 : obtain map               第四步是得到地图
        while (gridmap_lock_.test_and_set()) 
          ;
        gridmapPtr_->from_msg(map_msg_);
        replanStateMsg_.occmap = map_msg_; 
        gridmap_lock_.clear(); 
        prePtr_->setMap(*gridmapPtr_); 

        // visualize the ray from drone to target
        if (envPtr_->checkRayValid(odom_p, target_p)) { // 检查飞机和目标线段是否有障碍物
          visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::yellow);
        } else {
          visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::red);
        }

        //! step 5 : prediction
        std::vector<Eigen::Vector3d> target_predcit; // 预测路径
        bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit); 
        // 根据目标的为位置和速度得到预测的一堆位置 放在target_predict
        if (generate_new_traj_success) {
        Eigen::Vector3d observable_p = target_predcit.back(); // 预测的最终位置
        visPtr_->visualize_path(target_predcit, "car_predict"); //预测路径可视化

        // 最终位置方圆tracking_dist的边缘画出来，可视边缘
         std::vector<Eigen::Vector3d> observable_margin;
        for (double theta = 0; theta <= 2 * M_PI; theta += 0.01) {
          observable_margin.emplace_back(observable_p + tracking_dist_ * Eigen::Vector3d(cos(theta), sin(theta), 0));
          }
        visPtr_->visualize_path(observable_margin, "observable_margin"); // 在rviz看一下就知道是什么了
        }

        //! step 6 : setting replan state
        Eigen::MatrixXd iniState;
        iniState.setZero(3, 3); // 三列，p,v,a
        ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03); // 预测终点的时间戳
        double replan_t = (replan_stamp - replan_stamp_).toSec();  //toSec应该就是转化为秒的意思

        if (force_hover_ || replan_t > traj_poly_.getTotalDuration()) { // 接收到的路径的
        // should replan from the hover state
        iniState.col(0) = odom_p;
         iniState.col(1) = odom_v;
         } else {
        // should replan from the last trajectory
        iniState.col(0) = traj_poly_.getPos(replan_t);  // 从上一次的轨迹开始重规划，获取从上一次重规划经过若干点后的在轨迹上的状态
        iniState.col(1) = traj_poly_.getVel(replan_t);
        iniState.col(2) = traj_poly_.getAcc(replan_t);
       }
        replanStateMsg_.header.stamp = ros::Time::now();
        replanStateMsg_.iniState.resize(9); // 长度为9的数组
        Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3) = iniState; // 赋值过去


        //! step 7 : path searching
        Eigen::Vector3d p_start = iniState.col(0); // 开始位置
        std::vector<Eigen::Vector3d> path, way_pts; // 路径和waypoint
        
        if (generate_new_traj_success) { // prediction for car
          generate_new_traj_success = envPtr_->short_astar(p_start, target_predcit.back(), path); // 从当前位置到预测的终点直接搜出一条Astar
        ROS_WARN_STREAM("[ planner ] landing stage : path.size() = " << path.size());

        Trajectory traj;
        
        if (generate_new_traj_success) {
              visPtr_->visualize_path(path, "astar"); 

        //! step 8 : corridor generating
         ROS_WARN("corridor generating");
        // NOTE corridor generating
        std::vector<Eigen::MatrixXd> hPolys;      //多边形的飞行走廊
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;

        envPtr_->generateSFC(path, 2.0, hPolys, keyPts); // 生成飞行走廊
        envPtr_->visCorridor(hPolys); // 可视化飞行走廊
        visPtr_->visualize_pairline(keyPts, "keyPts");

        //! step 9 : trajectory optimization
        ROS_WARN("trajectory optimization");
        // NOTE trajectory optimization
        Eigen::MatrixXd finState; // 末端状态
        finState.setZero(3, 3);
        finState.col(0) = path.back(); // 轨迹的终点
        finState.col(1) = target_v; // 和目标一样的速度
        // finState.col(1).z()+=-0.2; // 末端添加一个向下的速度
        finState.col(0) = target_predcit.back(); //如果收到降落信号，则终点是和目标一样的位置
        generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState, target_predcit, hPolys, traj);
          visPtr_->visualize_traj(traj, "traj");
        }

        //! step 10 : collision check 
        bool valid = false;
        if (generate_new_traj_success) {
          valid = validcheck(traj, replan_stamp); // 从replan_stemp开始的1s内是否碰撞，已经是之后的事情了
        } else {
          replanStateMsg_.state = -2;
          replanState_pub_.publish(replanStateMsg_);
       }

        if (valid) { // 从replan_stemp开始的1s内无碰撞
          force_hover_ = false;
          ROS_WARN("[planner] LANDING REPLAN SUCCESS");
          replanStateMsg_.state = 0; // 
          replanState_pub_.publish(replanStateMsg_);
          Eigen::Vector3d dp = target_p + target_v * 0.03 - iniState.col(0); // 假设匀速直线运动的目标未来位置到飞机当前位置的位移
          // NOTE : if the trajectory is known, watch that direction
          double yaw = std::atan2(dp.y(), dp.x()); // yaw角对准目标未来位置
          yaw = last_yaw_;
          last_yaw_ = yaw;
          pub_traj(traj, yaw, replan_stamp); //发布轨迹，在这里发布轨迹
          traj_poly_ = traj; // 记录为上一次的轨迹
          replan_stamp_ = replan_stamp; // 记录为上一次的时间戳
        } // 生成的轨迹从replan开始1s内有碰撞 
        else if (force_hover_) { // 如果轨迹有问题，且有force_hover_，则悬停
          ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
          replanStateMsg_.state = 1;
          replanState_pub_.publish(replanStateMsg_);
          return;
        } else if (!validcheck(traj_poly_, replan_stamp_)) { // 轨迹有问题，检查生成轨迹上一个轨迹中在1s有碰撞 
          force_hover_ = true;
          ROS_FATAL("[planner] EMERGENCY STOP!!!"); // 紧急悬停
          replanStateMsg_.state = 2;
          replanState_pub_.publish(replanStateMsg_);
          pub_hover_p(iniState.col(0), replan_stamp);                 
          break;
        } else {
          ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ..."); // 检查生成轨迹上一个轨迹中在1s无碰撞
          replanStateMsg_.state = 3;
          replanState_pub_.publish(replanStateMsg_);
          break;  // current generated traj invalid but last is valid
        }
        visPtr_->visualize_traj(traj, "traj"); // 可视化轨迹，发布的话题名称为traj，对应航点为traj_wpts
        }
        break;
      }
      case CALI:
      {
        ROS_INFO_ONCE("[ planner ] in calibration process");
        if(!landed_flag)
          trigger_cali_ctrl_flag = true;

        else{
          trigger_cali_ctrl_flag = false; // 关闭ctrl_callback
          ROS_WARN_ONCE("[ planner ] CALI stage : land success "); 
          task_state = READY;
        }

        break;
      }
      case LANDED:
      {
        task_state_msg.state="LANDED";
        ROS_INFO("[planner] ALREADY landed");
        return ;
        break;
      }
      case DEBUG:
      {
        task_state_msg.state="DEBUG";


        // 调节pid

        // step 1 : 飞机目前的状态
        while (odom_lock_.test_and_set()) ;
        auto odom_msg = odom_msg_; 
        odom_lock_.clear(); 

        Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                              odom_msg.pose.pose.position.y,
                              odom_msg.pose.pose.position.z);
        Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                              odom_msg.twist.twist.linear.y,
                              odom_msg.twist.twist.linear.z);
        Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,
                                  odom_msg.pose.pose.orientation.x,
                                  odom_msg.pose.pose.orientation.y,
                                  odom_msg.pose.pose.orientation.z);
        // step 2 : 栅格地图
        while (gridmap_lock_.test_and_set()) 
          ;
        gridmapPtr_->from_msg(map_msg_);
        replanStateMsg_.occmap = map_msg_; 
        gridmap_lock_.clear(); 

        Eigen::MatrixXd iniState,finState;
        iniState.setZero(3,3);
        finState.setZero(3,3);

        Eigen::Vector3d p_start = iniState.col(0);
        Eigen::Vector3d p_end = finState.col(0);
        std::vector<Eigen::Vector3d> path;
        
        bool generate_traj_success = envPtr_->short_astar(p_start,p_end,path);

        Trajectory traj;

        std::vector<Eigen::MatrixXd> hPolys;
        std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> keyPts;

        envPtr_->generateSFC(path,2.0,hPolys,keyPts);

        std::vector<Eigen::Vector3d> target_predict;
        generate_traj_success = trajOptPtr_->generate_traj(iniState,finState,target_predict,hPolys,traj);
        
        ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03); // 预测终点的时间戳

        double yaw=0;
        pub_traj(traj, yaw, replan_stamp); //发布轨迹，在这里发布轨迹






        break;
      }
      default:
        break;
    }
    
    return ;
}

  //! fake 回调函数
  void fake_timer_callback(const ros::TimerEvent& event) {
    // dosomehing
    ROS_WARN("in fake callback");
  }


  //! debug用的回调函数
  void debug_timer_callback(const ros::TimerEvent& event) {
   // do something
    ROS_WARN("in debug callback");
  }

  // 如果发现traj上的T时刻的p的位置在地图上有障碍物占领了，那么便宣布这个轨迹无效，Time是时间戳 ,t_start为从哪里开始检查轨迹， check_dur是持续的时间
  // return true : 安全
  // return false : 不安全
  bool validcheck(const Trajectory& traj, const ros::Time& t_start, const double& check_dur = 1.0) {
    double t0 = (ros::Time::now() - t_start).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration(); // 限制在轨迹时间范围内
    for (double t = t0; t < t0 + delta_t; t += 0.01) {
      Eigen::Vector3d p = traj.getPos(t);
      if (gridmapPtr_->isOccupied(p)) {
        return false;                                          //如果p的位置在地图上有障碍物占领了，那么便宣布这个轨迹无效
      }
    }
    return true;
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of planning
    int plan_hz = 10;    //以plan_hz的频率执行回调函数
    
    nh.getParam("plan_hz", plan_hz);
    nh.getParam("tracking_dur", tracking_dur_); 
    nh.getParam("tracking_dist", tracking_dist_); 
    nh.getParam("tolerance_d", tolerance_d_);
    nh.getParam("tracking_z", tracking_z_); // 跟踪的高度
    nh.getParam("debug", debug_); // 是否debug模式
    nh.getParam("fake", fake_);  // 是否跟踪
    int ctrl_hz = nh.param("ctrl_hz",50); // 校正模式的跟踪频率

    gridmapPtr_ = std::make_shared<mapping::OccGridMap>();      //栅格地图
    envPtr_ = std::make_shared<env::Env>(nh, gridmapPtr_);
    visPtr_ = std::make_shared<visualization::Visualization>(nh); // 可视化
    trajOptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh); // 轨迹优化对象
    prePtr_ = std::make_shared<prediction::Predict>(nh);

    heartbeat_pub_ = nh.advertise<std_msgs::Empty>("heartbeat", 10); // 发布心跳，表示planner还活着
    ctrl_heartbeat_pub_=nh.advertise<std_msgs::Empty>("heartbeat_ctrl",10);
    traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("trajectory", 1); // 发布轨迹消息给traj server，traj_pub_
    replanState_pub_ = nh.advertise<quadrotor_msgs::ReplanState>("replanState", 1); // 发布replanState

    relay_pub_ = nh.advertise<std_msgs::Bool>("/relayctrl_node/cmd", 1); // 控制电磁继电器
    motors_pub_ = nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 1); // 控制马达

    // ***************** 任务控制和反馈
    task_state_pub_ =nh.advertise<sim_msgs::task_state>("/task_state",1);
    set_task_sub_ = nh.subscribe<sim_msgs::task_state>("/set_task_state",1, &Nodelet::set_task_callback, this, ros::TransportHints().tcpNoDelay());

    // ***************** 校正任务
    cali_pos_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd",1);


  

      // 一般模式
    ctrl_timer_ = nh.createTimer(ros::Duration(1.0/ ctrl_hz),&Nodelet::ctrl_timer_callback,this);

    plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::plan_timer_callback, this); // plan_hz的频率执行callback
    
    gridmap_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1, &Nodelet::gridmap_callback, this, ros::TransportHints().tcpNoDelay()); // 膨胀之后的gripmap
    
    // 订阅的里程计信息 /vins/imu_propogate
    odom_sub_ = nh.subscribe<nav_msgs::Odometry>("odom", 10, &Nodelet::odom_callback, this, ros::TransportHints().tcpNoDelay());
    target_sub_ = nh.subscribe<nav_msgs::Odometry>("target", 10, &Nodelet::target_callback, this, ros::TransportHints().tcpNoDelay());
    triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, &Nodelet::triger_callback, this, ros::TransportHints().tcpNoDelay());
    land_triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("land_triger", 10, &Nodelet::land_triger_callback, this, ros::TransportHints().tcpNoDelay());
    // laser_sub_ = nh.subscribe<std_msgs::Int32MultiArray>("/tfluna_node/data", 10, &Nodelet::laser_callback, this, ros::TransportHints().tcpNoDelay());
    // laser_sub_ = nh.subscribe<nlink_parser::TofsenseFrame0>("/nlink_tofsense_frame0", 10, &Nodelet::laser_callback, this, ros::TransportHints().tcpNoDelay()); //

    laser_sub_ = nh.subscribe<sensor_msgs::LaserScan>("/nlink_tofsense_frame0", 10, &Nodelet::sim_laser_callback, this, ros::TransportHints().tcpNoDelay()); //

    ROS_WARN("Planning node initialized!");
  }

 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&Nodelet::init, this, nh));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace planning

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(planning::Nodelet, nodelet::Nodelet);