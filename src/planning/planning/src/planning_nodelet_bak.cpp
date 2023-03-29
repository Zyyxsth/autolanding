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
#include "sim_node/task_state.h"

sim_node::task_state task_state_msg;
sim_node::task_state set_task_msg;
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

class Nodelet : public nodelet::Nodelet {
 private:
  std::thread initThread_;
  ros::Subscriber gridmap_sub_, odom_sub_, target_sub_, triger_sub_, land_triger_sub_;
  ros::Subscriber laser_sub_;
  ros::Timer plan_timer_;

  ros::Publisher traj_pub_, heartbeat_pub_, replanState_pub_;
  ros::Publisher relay_pub_, motors_pub_;   // relay 继电器

  ros::Publisher task_state_pub_;
  ros::Subscriber set_task_sub_;



  double last_yaw_;

  std::shared_ptr<mapping::OccGridMap> gridmapPtr_;
  std::shared_ptr<env::Env> envPtr_;
  std::shared_ptr<visualization::Visualization> visPtr_;
  std::shared_ptr<traj_opt::TrajOpt> trajOptPtr_;
  std::shared_ptr<prediction::Predict> prePtr_;

  // NOTE planning for fake target
  bool fake_ = false;
  Eigen::Vector3d goal_;
  Eigen::Vector3d land_p_;
  Eigen::Quaterniond land_q_;

  // NOTE just for debug
  bool debug_ = false;
  quadrotor_msgs::ReplanState replanStateMsg_;
  ros::Publisher gridmap_pub_, inflate_gridmap_pub_;
  quadrotor_msgs::OccMap3d occmap_msg_;

  double tracking_dur_, tracking_dist_, tolerance_d_, tracking_z_;

  Trajectory traj_poly_; // 上一次的轨迹
  ros::Time replan_stamp_;
  int traj_id_ = 0;
  bool wait_hover_ = true;
  bool force_hover_ = true;
  bool landing_finished_ = false;

  nav_msgs::Odometry odom_msg_, target_msg_;
  quadrotor_msgs::OccMap3d map_msg_; // 栅格地图参数以及数据（一个数组）

// 看成布尔类型即可
  std::atomic_flag odom_lock_ = ATOMIC_FLAG_INIT; // clear状态,既不是true，也不是false
  std::atomic_flag target_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag gridmap_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_bool odom_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool map_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool triger_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool target_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool land_triger_received_ = ATOMIC_VAR_INIT(false);

  // 启动继电器
  void turn_on_relay() {
    std_msgs::Bool msg;
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
  void pub_hover_p(const Eigen::Vector3d& hover_p, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = true;
    traj_msg.hover_p.resize(3);
    for (int i = 0; i < 3; ++i) {
      traj_msg.hover_p[i] = hover_p[i];
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    traj_pub_.publish(traj_msg);
  }


  void pub_traj(const Trajectory& traj, const double& yaw, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = false;
    traj_msg.order = 5;
    Eigen::VectorXd durs = traj.getDurations();
    int piece_num = traj.getPieceNum();
    traj_msg.duration.resize(piece_num); // 每段一个duration
    traj_msg.coef_x.resize(6 * piece_num);  // 重定义维度
    traj_msg.coef_y.resize(6 * piece_num);
    traj_msg.coef_z.resize(6 * piece_num);
    for (int i = 0; i < piece_num; ++i) {
      traj_msg.duration[i] = durs(i);
      CoefficientMat cMat = traj[i].getCoeffMat();
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
    traj_msg.yaw = yaw;
    traj_pub_.publish(traj_msg);
  }
    }}

  void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    goal_ << msgPtr->pose.position.x, msgPtr->pose.position.y, 0.9;
    triger_received_ = true;
  }

  void set_task_callback(const sim_node::task_stateConstPtr& msgPtr)
  {
    while(set_task_lock_.test_and_set()) ;
    set_task_msg = *msgPtr;
    switch(set_task_msg.state_id)
    {
      case sim_node::task_state::TASK_READY:
        task_state=READY;
        break;
      case sim_node::task_state::TRACK:
        task_state=TRACK;
        break;
      case sim_node::task_state::LADNING:
        task_state=LANDING;
        break;
      case sim_node::task_state::CALIBRATION:
        task_state=CALI;
        break;
      case sim_node::task_state::LANDED:
        task_state=LANDED;
        break;
      case sim_node::task_state::DEBUG:
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
    land_p_.x() = msgPtr->pose.position.x;
    land_p_.y() = msgPtr->pose.position.y;
    land_p_.z() = msgPtr->pose.position.z;
    land_q_.w() = msgPtr->pose.orientation.w;
    land_q_.x() = msgPtr->pose.orientation.x;
    land_q_.y() = msgPtr->pose.orientation.y;
    land_q_.z() = msgPtr->pose.orientation.z;
    land_triger_received_ = true;
  }

  void odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (odom_lock_.test_and_set()) // 启动锁
      ;
    odom_msg_ = *msgPtr;
    odom_received_ = true;
    odom_lock_.clear(); // 清楚里程计锁
  }

  // 
  void target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (target_lock_.test_and_set())
      ;
    target_msg_ = *msgPtr;
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
  void sim_laser_callback(const sensor_msgs::LaserScanConstPtr& pMsg)
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
    map_msg_ = *msgPtr;
    map_received_ = true;
    gridmap_lock_.clear();
  }


// 重构
// 先把landing和track的部分分开写
// 然后进行

//! 重点，由规划定时器触发的回调函数
  // NOTE main callback
void plan_timer_callback(const ros::TimerEvent& event) {
    heartbeat_pub_.publish(std_msgs::Empty()); // planner heartbeat
    if (!odom_received_ || !map_received_) { // no odom or no map planner not work
      ROS_INFO("[ planner ] no odom or no map received");
      return;
    }
    

    // obtain state of odom
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


    if (!triger_received_) { // no track trigger
      ROS_INFO("[ planner ] watting for trigger for uav");
      task_state = NOT_READY;
      return;
    }

    task_state = READY ;

    if (!target_received_) { // no target in view
      ROS_INFO("[ planner ] watting for landing target");
      return;
    }
    // NOTE obtain state of target 
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
    if (force_hover_ && odom_v.norm() > 0.1) {
      return;
    }

    // NOTE just for landing on the car of YTK! 
    if (land_triger_received_) {  // landing
        // turn_on_relay(); 
      // NOTE turn on the magnet and stop the motors!
      // ROS_INFO("odom_z = %f,target_z = %f,minus = %f",odom_p.z(),target_p.z(),(odom_p - target_p).z());
      
      //test if landed
      static int count_laser = 0;
      if( dis_laser_mm < 120 && dis_laser_mm > 5 ) // 如果底部激光探测到这个距离，说明接触到了物体，贴上去
      {
        count_laser++;
        ROS_WARN("laser count++! count_laser = %d",count_laser);
      }
      else
      {
        count_laser = 0;
      }

      if 
      (
        // (odom_p - target_p).z() < 0.12
        count_laser >= 2
      ) 
      {
        // 完成降落
        landing_finished_ = true;
        ROS_WARN("[planner] LANDING FINISHED!");
      }

      if (landing_finished_)  // landed
      {
        stop_motors(); //发布停止电机的消息
        ROS_WARN("[planner] has landed at the landing position");
        return;
      }
      else // landing 
      {
        // log 注释掉降落过程
        in_landing_process(); // 将motor的状态改为TRUST_ADJUST_BY_LASER
      }


    // 足够近了，悬停
      if (std::fabs((target_p - odom_p).norm() < 0.1 && odom_v.norm() < 0.1 && target_v.norm() < 0.2)) {
        if (!wait_hover_) {
          pub_hover_p(odom_p, ros::Time::now());
          wait_hover_ = true; // 正在悬停
        }
        ROS_WARN("[planner] HOVERING...");
        return;
      }

      // TODO get the orientation fo target and calculate the pose of landing point
      target_p = target_p + target_q * land_p_; // 降落时的轨迹的终点位置，land_p_是相对于车的降落点，只有faketarget时才有后两项
      wait_hover_ = false; // 还没足够进，不要悬停

      // target_p.z()+= 1; // 飞到上面1米处
      // end of if(land_triger_received_) 
    } 
    
    else { // tracking


      target_p.z() += tracking_z_;  // 跟踪时候距离小车的垂直距离

      //! NOTE determin whether to replan
      Eigen::Vector3d dp = target_p - odom_p; // 距离目标的位置差向量
      // std::cout << "dist : " << dp.norm() << std::endl;
      double desired_yaw = std::atan2(dp.y(), dp.x()); // 朝向目标的期望的yaw角
      Eigen::Vector3d project_yaw = odom_q.toRotationMatrix().col(0);  // NOTE ZYX //! 旋转矩阵取第一列 
      double now_yaw = std::atan2(project_yaw.y(), project_yaw.x()); // 现在的yaw角
      if (std::fabs((target_p - odom_p).norm() - tracking_dist_) < tolerance_d_ && // 距离小车头顶的点太近了，悬停
          odom_v.norm() < 0.1 && target_v.norm() < 0.2 &&
          std::fabs(desired_yaw - now_yaw) < 0.5) { // 角度误差小于0.5
        if (!wait_hover_) {
          pub_hover_p(odom_p, ros::Time::now()); // 在原地悬停
          wait_hover_ = true;
        }
        ROS_WARN("[planner] HOVERING...");
        replanStateMsg_.state = -1;
        replanState_pub_.publish(replanStateMsg_); // 悬停，不replan
        return;
      } else {
        wait_hover_ = false; // replan，不悬停
      }
    }

    // NOTE obtain map 
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

    //! NOTE prediction
    std::vector<Eigen::Vector3d> target_predcit; // 预测路径
    // ros::Time t_start = ros::Time::now();
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit); 
    // 根据目标的为位置和速度得到预测的目标位置 放在target_predict
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


    // NOTE replan state
    Eigen::MatrixXd iniState;
    iniState.setZero(3, 3); // 三列，p,v,a
    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03); // 预测终点的时间戳
    double replan_t = (replan_stamp - replan_stamp_).toSec(); 
    // 距离上次重规划的时间长度 replan_stamp_是上一次的时间戳
    
    // 之前是悬停 或者距离上一次规划的时间长度大于轨迹的长度时，当前状态应该是悬停，从悬停状态(当前状态)开始replan
    //  traj_poly_是上一次的轨迹
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
 
    //! NOTE path searching
    Eigen::Vector3d p_start = iniState.col(0); // 开始位置
    std::vector<Eigen::Vector3d> path, way_pts; // 路径和waypoint


    // 用Astar，得到离散路径path
    if (generate_new_traj_success) { // prediction for car
      if (land_triger_received_) { // landing
        generate_new_traj_success = envPtr_->short_astar(p_start, target_predcit.back(), path); // 从当前位置到预测的终点直接搜出一条Astar
      } else { // 跟踪，找到最佳的观察点
        generate_new_traj_success = envPtr_->findVisiblePath(p_start, target_predcit, way_pts, path); 
    }

    std::vector<Eigen::Vector3d> visible_ps;
    std::vector<double> thetas;
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar"); 
      if (land_triger_received_) { 
      
      } else { // tracking
        // NOTE generate visible regions
        target_predcit.pop_back(); // 删除尾端元素
        way_pts.pop_back();
        envPtr_->generate_visible_regions(target_predcit, way_pts, // 生成一个区域，waypoint变成可视范围的中间
                                          visible_ps, thetas);
        // 这里修改了way_pts
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
      }

    //! 生成飞行走廊
      ROS_WARN("corridor generating");
      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;

      envPtr_->generateSFC(path, 2.0, hPolys, keyPts); // 生成飞行走廊
      envPtr_->visCorridor(hPolys); // 可视化飞行走廊
      visPtr_->visualize_pairline(keyPts, "keyPts");

      //! 轨迹优化
      ROS_WARN("trajectory optimization");
      // NOTE trajectory optimization
      Eigen::MatrixXd finState; // 末端状态
      finState.setZero(3, 3);
      finState.col(0) = path.back(); // 轨迹的终点
      finState.col(1) = target_v; // 和目标一样的速度
      finState.col(1).z()+=-0.2; // 末端添加一个向下的速度

      if (land_triger_received_) {
        finState.col(0) = target_predcit.back(); //如果收到降落信号，则终点是和目标一样的位置
        generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState, target_predcit, hPolys, traj);
      } else {
        // 变成跟踪了
        generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState,
                                                               target_predcit, visible_ps, thetas,
                                                               hPolys, traj);
      }


      visPtr_->visualize_traj(traj, "traj");
    }

    // NOTE collision check
    bool valid = false;
    if (generate_new_traj_success) {
      valid = validcheck(traj, replan_stamp); // 从replan_stemp开始的1s内是否碰撞，已经是之后的事情了
    } else {
      replanStateMsg_.state = -2;
      replanState_pub_.publish(replanStateMsg_);
    }

    if (valid) { // 从replan_stemp开始的1s内无碰撞
      force_hover_ = false;
      ROS_WARN("[planner] REPLAN SUCCESS");
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
      if (land_triger_received_) {
        // yaw = 2 * std::atan2(target_q.z(), target_q.w());
        yaw = last_yaw_;
      }
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
    } else if (!validcheck(traj_poly_, replan_stamp_)) { // 轨迹有问题，检查生成轨迹上一个轨迹中在1s无碰撞 ，这里应该是写反了
      force_hover_ = true;
      ROS_FATAL("[planner] EMERGENCY STOP!!!"); // 紧急悬停
      replanStateMsg_.state = 2;
      replanState_pub_.publish(replanStateMsg_);
      pub_hover_p(iniState.col(0), replan_stamp);                 
      return;
    } else {
      ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ..."); // 检查生成轨迹上一个轨迹中在1s无碰撞
      replanStateMsg_.state = 3;
      replanState_pub_.publish(replanStateMsg_);
      return;  // current generated traj invalid but last is valid
    }
    visPtr_->visualize_traj(traj, "traj"); // 可视化轨迹，发布的话题名称为traj，对应航点为traj_wpts
    

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
    }
  }

  //! fake 回调函数
  void fake_timer_callback(const ros::TimerEvent& event) {
    heartbeat_pub_.publish(std_msgs::Empty()); // 我还活着
    if (!odom_received_ || !map_received_) {
      return;
    }
    // obtain state of odom
    while (odom_lock_.test_and_set())
      ;
    auto odom_msg = odom_msg_;
    odom_lock_.clear();
    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                           odom_msg.pose.pose.position.y,
                           odom_msg.pose.pose.position.z);
    Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                           odom_msg.twist.twist.linear.y,
                           odom_msg.twist.twist.linear.z);
    if (!triger_received_) {
      return;
    }
    // NOTE force-hover: waiting for the speed of drone small enough
    if (force_hover_ && odom_v.norm() > 0.1) {
      return;
    }

    // NOTE local goal
    Eigen::Vector3d local_goal;
    Eigen::Vector3d delta = goal_ - odom_p;
    if (delta.norm() < 15) {
      local_goal = goal_;
    } else {
      local_goal = delta.normalized() * 15 + odom_p; // local范围不超过15
    }

    // NOTE obtain map
    while (gridmap_lock_.test_and_set())
      ;
    gridmapPtr_->from_msg(map_msg_);
    replanStateMsg_.occmap = map_msg_;
    gridmap_lock_.clear();

    // NOTE determin whether to replan
    bool no_need_replan = false;
    if (!force_hover_ && !wait_hover_) {
      double last_traj_t_rest = traj_poly_.getTotalDuration() - (ros::Time::now() - replan_stamp_).toSec();

      // 目标距离轨迹终点大于跟踪距离
      bool new_goal = (local_goal - traj_poly_.getPos(traj_poly_.getTotalDuration())).norm() > tracking_dist_;
      if (!new_goal) { //靠近goal
        if (last_traj_t_rest < 1.0) { // 剩余时间太短了，且靠近goal
          ROS_WARN("[planner] NEAR GOAL...");
          no_need_replan = true;
        } else if (validcheck(traj_poly_, replan_stamp_, last_traj_t_rest)) { // 剩余时间大于1
          ROS_WARN("[planner] NO NEED REPLAN...");
          //! 确定yaw角朝向
          double t_delta = traj_poly_.getTotalDuration() < 1.0 ? traj_poly_.getTotalDuration() : 1.0;
          double t_yaw = (ros::Time::now() - replan_stamp_).toSec() + t_delta;
          Eigen::Vector3d un_known_p = traj_poly_.getPos(t_yaw); //取得1s后的位置
          Eigen::Vector3d dp = un_known_p - odom_p; 
          double yaw = std::atan2(dp.y(), dp.x());
          pub_traj(traj_poly_, yaw, replan_stamp_);
          no_need_replan = true;
        }
      }
    }
    // NOTE determin whether to pub hover
    if ((goal_ - odom_p).norm() < tracking_dist_ + tolerance_d_ && odom_v.norm() < 0.1) {
      if (!wait_hover_) {
        pub_hover_p(odom_p, ros::Time::now());
        wait_hover_ = true;
      }
      ROS_WARN("[planner] HOVERING...");
      replanStateMsg_.state = -1;
      replanState_pub_.publish(replanStateMsg_);
      return;
    } else {
      wait_hover_ = false;
    }
    if (no_need_replan) {
      return;
    }

    // NOTE replan state
    Eigen::MatrixXd iniState;
    iniState.setZero(3, 3);
    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);
    double replan_t = (replan_stamp - replan_stamp_).toSec();
    if (force_hover_ || replan_t > traj_poly_.getTotalDuration()) {
      // should replan from the hover state
      iniState.col(0) = odom_p;
      iniState.col(1) = odom_v;
    } else {
      // should replan from the last trajectory
      iniState.col(0) = traj_poly_.getPos(replan_t);
      iniState.col(1) = traj_poly_.getVel(replan_t);
      iniState.col(2) = traj_poly_.getAcc(replan_t);
    }
    replanStateMsg_.header.stamp = ros::Time::now();
    replanStateMsg_.iniState.resize(9);
    Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3) = iniState;

    // NOTE generate an extra corridor
    Eigen::Vector3d p_start = iniState.col(0);
    bool need_extra_corridor = iniState.col(1).norm() > 1.0;
    Eigen::MatrixXd hPoly;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> line;
    if (need_extra_corridor) {
      Eigen::Vector3d v_norm = iniState.col(1).normalized();
      line.first = p_start;
      double step = 0.1;
      for (double dx = step; dx < 1.0; dx += step) {
        p_start += step * v_norm;
        if (gridmapPtr_->isOccupied(p_start)) {
          p_start -= step * v_norm;
          break;
        }
      }
      line.second = p_start;
      envPtr_->generateOneCorridor(line, 2.0, hPoly);
    }
    // NOTE path searching
    std::vector<Eigen::Vector3d> path;
    bool generate_new_traj_success = envPtr_->astar_search(p_start, local_goal, path);
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar");
      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      if (need_extra_corridor) {
        hPolys.insert(hPolys.begin(), hPoly);
        keyPts.insert(keyPts.begin(), line);
      }
      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      // NOTE trajectory optimization
      Eigen::MatrixXd finState;
      finState.setZero(3, 3);
      finState.col(0) = path.back();
      // return;
      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState, hPolys, traj);
      visPtr_->visualize_traj(traj, "traj");
    }

    // NOTE collision check
    bool valid = false;
    if (generate_new_traj_success) {
      valid = validcheck(traj, replan_stamp);
    } else {
      replanStateMsg_.state = -2;
      replanState_pub_.publish(replanStateMsg_);
    }
    if (valid) {
      force_hover_ = false;
      ROS_WARN("[planner] REPLAN SUCCESS");
      replanStateMsg_.state = 0;
      replanState_pub_.publish(replanStateMsg_);
      // NOTE : if the trajectory is known, watch that direction
      Eigen::Vector3d un_known_p = traj.getPos(traj.getTotalDuration() < 1.0 ? traj.getTotalDuration() : 1.0);
      Eigen::Vector3d dp = un_known_p - odom_p;
      double yaw = std::atan2(dp.y(), dp.x());
      pub_traj(traj, yaw, replan_stamp);
      traj_poly_ = traj;
      replan_stamp_ = replan_stamp;
    } else if (force_hover_) {
      ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
      replanStateMsg_.state = 1;
      replanState_pub_.publish(replanStateMsg_);
      return;
    } else if (validcheck(traj_poly_, replan_stamp_)) {
      force_hover_ = true;
      ROS_FATAL("[planner] EMERGENCY STOP!!!");
      replanStateMsg_.state = 2;
      replanState_pub_.publish(replanStateMsg_);
      pub_hover_p(iniState.col(0), replan_stamp);
      return;
    } else {
      ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ...");
      replanStateMsg_.state = 3;
      replanState_pub_.publish(replanStateMsg_);
      return;  // current generated traj invalid but last is valid
    }
    visPtr_->visualize_traj(traj, "traj");
  }


  //! debug用的回调函数
  void debug_timer_callback(const ros::TimerEvent& event) {
    inflate_gridmap_pub_.publish(replanStateMsg_.occmap);
    Eigen::MatrixXd iniState;
    iniState.setZero(3, 3);
    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);

    iniState = Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 3);
    Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                             replanStateMsg_.target.pose.pose.position.y,
                             replanStateMsg_.target.pose.pose.position.z);
    Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                             replanStateMsg_.target.twist.twist.linear.y,
                             replanStateMsg_.target.twist.twist.linear.z);
    // std::cout << "target_p: " << target_p.transpose() << std::endl;
    // std::cout << "target_v: " << target_v.transpose() << std::endl;

    // visualize the target and the drone velocity
    visPtr_->visualize_arrow(iniState.col(0), iniState.col(0) + iniState.col(1), "drone_vel");
    visPtr_->visualize_arrow(target_p, target_p + target_v, "target_vel");

    // visualize the ray from drone to target
    if (envPtr_->checkRayValid(iniState.col(0), target_p)) {
      visPtr_->visualize_arrow(iniState.col(0), target_p, "ray", visualization::yellow);
    } else {
      visPtr_->visualize_arrow(iniState.col(0), target_p, "ray", visualization::red);
    }

    // NOTE prediction
    std::vector<Eigen::Vector3d> target_predcit;
    if (gridmapPtr_->isOccupied(target_p)) {
      std::cout << "target is invalid!" << std::endl;
      assert(false);
    }
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit);

    if (generate_new_traj_success) {
      Eigen::Vector3d observable_p = target_predcit.back();
      visPtr_->visualize_path(target_predcit, "car_predict");
      std::vector<Eigen::Vector3d> observable_margin;
      for (double theta = 0; theta <= 2 * M_PI; theta += 0.01) {
        observable_margin.emplace_back(observable_p + tracking_dist_ * Eigen::Vector3d(cos(theta), sin(theta), 0));
      }
      visPtr_->visualize_path(observable_margin, "observable_margin");
    }

    // NOTE path searching
    Eigen::Vector3d p_start = iniState.col(0);
    std::vector<Eigen::Vector3d> path, way_pts;
    if (generate_new_traj_success) {
      generate_new_traj_success = envPtr_->findVisiblePath(p_start, target_predcit, way_pts, path);
    }

    std::vector<Eigen::Vector3d> visible_ps;
    std::vector<double> thetas;
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_path(path, "astar");
      // NOTE generate visible regions
      target_predcit.pop_back();
      way_pts.pop_back();
      envPtr_->generate_visible_regions(target_predcit, way_pts,
                                        visible_ps, thetas);
      visPtr_->visualize_pointcloud(visible_ps, "visible_ps");
      visPtr_->visualize_fan_shape_meshes(target_predcit, visible_ps, thetas, "visible_region");
      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      // TODO change the final state
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
      for (int i = 0; i < (int)way_pts.size(); ++i) {
        rays.emplace_back(target_predcit[i], way_pts[i]);
      }
      visPtr_->visualize_pointcloud(way_pts, "way_pts");
      way_pts.insert(way_pts.begin(), p_start);
      envPtr_->pts2path(way_pts, path);
      visPtr_->visualize_path(path, "corridor_path");
      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      // NOTE trajectory optimization
      Eigen::MatrixXd finState;
      finState.setZero(3, 3);
      finState.col(0) = path.back();
      finState.col(1) = target_v;

      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, finState,
                                                             target_predcit, visible_ps, thetas,
                                                             hPolys, traj);
      visPtr_->visualize_traj(traj, "traj");
    }
    if (!generate_new_traj_success) {
      return;
      // assert(false);
    }
    // check
    bool valid = true;
    std::vector<Eigen::Vector3d> check_pts, invalid_pts;
    double t0 = (ros::Time::now() - replan_stamp).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double check_dur = 1.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration();
    for (double t = t0; t < t0 + delta_t; t += 0.1) {
      Eigen::Vector3d p = traj.getPos(t);
      check_pts.push_back(p);
      if (gridmapPtr_->isOccupied(p)) {
        invalid_pts.push_back(p);
      }
    }
    visPtr_->visualize_path(invalid_pts, "invalid_pts");
    visPtr_->visualize_path(check_pts, "check_pts");
    valid = validcheck(traj, replan_stamp);
    if (!valid) {
      std::cout << "invalid!" << std::endl;
    }
  }

  // Time是时间戳 ,t_start为从哪里开始检查轨迹
  // return true : 安全
  // return false : 不安全
  bool validcheck(const Trajectory& traj, const ros::Time& t_start, const double& check_dur = 1.0) {
    double t0 = (ros::Time::now() - t_start).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration(); // 限制在轨迹时间范围内
    for (double t = t0; t < t0 + delta_t; t += 0.01) {
      Eigen::Vector3d p = traj.getPos(t);
      if (gridmapPtr_->isOccupied(p)) {
        return false;
      }
    }
    return true;
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of planning
    int plan_hz = 10;
    nh.getParam("plan_hz", plan_hz);
    nh.getParam("tracking_dur", tracking_dur_); 
    nh.getParam("tracking_dist", tracking_dist_); 
    nh.getParam("tolerance_d", tolerance_d_);
    nh.getParam("tracking_z", tracking_z_); // 跟踪的高度
    nh.getParam("debug", debug_); // 是否debug模式
    nh.getParam("fake", fake_);  // 是否跟踪

    gridmapPtr_ = std::make_shared<mapping::OccGridMap>();
    envPtr_ = std::make_shared<env::Env>(nh, gridmapPtr_);
    visPtr_ = std::make_shared<visualization::Visualization>(nh); // 可视化
    trajOptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh); // 轨迹优化对象
    prePtr_ = std::make_shared<prediction::Predict>(nh);

    heartbeat_pub_ = nh.advertise<std_msgs::Empty>("heartbeat", 10); // 发布心跳，表示planner还活着
    traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("trajectory", 1); // 发布轨迹消息给traj server，traj_pub_
    replanState_pub_ = nh.advertise<quadrotor_msgs::ReplanState>("replanState", 1); // 发布replanState

    relay_pub_ = nh.advertise<std_msgs::Bool>("/relayctrl_node/cmd", 1); // 控制电磁继电器
    motors_pub_ = nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 1); // 控制马达

    // ***************** 任务控制和反馈
    task_state_pub_ =nh.advertise<sim_node::task_state>("/task_state",1);
    set_task_sub_ = nh.subscribe<sim_node::task_state>("/set_task_state",1, &Nodelet::set_task_callback, this, ros::TransportHints().tcpNoDelay());

    if (debug_) {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::debug_timer_callback, this);
      // TODO read debug data from files
      wr_msg::readMsg(replanStateMsg_, ros::package::getPath("planning") + "/../../../debug/replan_state.bin");
      inflate_gridmap_pub_ = nh.advertise<quadrotor_msgs::OccMap3d>("gridmap_inflate", 10);
      gridmapPtr_->from_msg(replanStateMsg_.occmap);
      prePtr_->setMap(*gridmapPtr_);
      std::cout << "plan state: " << replanStateMsg_.state << std::endl;
    } 
    
    // 追踪fake目标，测试先用这个吧
    else if (fake_) {
      // plan定时器
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::fake_timer_callback, this);
    } 
    
    else {
      // 一般模式
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::plan_timer_callback, this); // plan_hz的频率执行callback
    }
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