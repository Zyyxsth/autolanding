#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>

#include <traj_opt/poly_traj_utils.hpp>

ros::Publisher pos_cmd_pub_;
ros::Time heartbeat_time_;
bool receive_traj_ = false;

// 保存两条路径
quadrotor_msgs::PolyTraj trajMsg_, trajMsg_last_;
// 上一次的位置和yaw角
Eigen::Vector3d last_p_;
double last_yaw_ = 0;

// 发布出去给px4ctrl执行
void publish_cmd(int traj_id, // 执行的路径的编号
                 const Eigen::Vector3d &p, // 执行期望的位置
                 const Eigen::Vector3d &v, //执行期望的速度
                 const Eigen::Vector3d &a, // 执行期望的加速度
                 double y, double yd) { // 执行期望的yaw角，yaw角变化率
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY; 
  cmd.trajectory_id = traj_id; // 执行的是哪条轨迹

  cmd.position.x = p(0); // 
  cmd.position.y = p(1);
  cmd.position.z = p(2);
  cmd.velocity.x = v(0);
  cmd.velocity.y = v(1);
  cmd.velocity.z = v(2);
  cmd.acceleration.x = a(0);
  cmd.acceleration.y = a(1);
  cmd.acceleration.z = a(2);
  cmd.yaw = y;
  cmd.yaw_dot = yd;
  pos_cmd_pub_.publish(cmd);
  last_p_ = p;
}

// 执行路径
bool exe_traj(const quadrotor_msgs::PolyTraj &trajMsg) {
  double t = (ros::Time::now() - trajMsg.start_time).toSec(); // 获取当前时间在轨迹上对应的时间长度
  if (t > 0) { // 当前时间在轨迹时间段内
    if (trajMsg.hover) { // 悬停
      if (trajMsg.hover_p.size() != 3) {
        ROS_ERROR("[traj_server] hover_p is not 3d!");
      }
      Eigen::Vector3d p, v0;
      p.x() = trajMsg.hover_p[0]; // 获取悬停的位置
      p.y() = trajMsg.hover_p[1];
      p.z() = trajMsg.hover_p[2];
      v0.setZero();
      publish_cmd(trajMsg.traj_id, p, v0, v0, last_yaw_, 0);  // TODO yaw
      return true;
    }
    if (trajMsg.order != 5) {
      ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
      return false;
    }
    if (trajMsg.duration.size() * (trajMsg.order + 1) != trajMsg.coef_x.size()) {
      ROS_ERROR("[traj_server] WRONG trajectory parameters!");
      return false;
    }
    // end of validation check

    int piece_nums = trajMsg.duration.size(); // 轨迹里的段数
    std::vector<double> dura(piece_nums); // 每段的时间
    std::vector<CoefficientMat> cMats(piece_nums); // 每段的系数
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
    Trajectory traj(dura, cMats); // 生成轨迹对象
    if (t > traj.getTotalDuration()) { // 生成的轨迹持续时间太短了，现在已经过了规划的时间段了
      ROS_ERROR("[traj_server] trajectory too short left!");
      return false;
    }
    Eigen::Vector3d p, v, a; // 获取当前时间的p,v,a,j
    p = traj.getPos(t);
    v = traj.getVel(t);
    a = traj.getAcc(t);
    // NOTE yaw
    double yaw = trajMsg.yaw;
    double d_yaw = yaw - last_yaw_;
    d_yaw = d_yaw >= M_PI ? d_yaw - 2 * M_PI : d_yaw; // 限定在正负pi
    d_yaw = d_yaw <= -M_PI ? d_yaw + 2 * M_PI : d_yaw;
    double d_yaw_abs = fabs(d_yaw);
    if (d_yaw_abs >= 0.02) {
      yaw = last_yaw_ + d_yaw / d_yaw_abs * 0.02; // yaw的变化率不超过0.02
    }
    publish_cmd(trajMsg.traj_id, p, v, a, yaw, 0);  // TODO yaw
    last_yaw_ = yaw;
    return true;
  }
  return false;
}

// 心跳信号，由规划节点发出，如果心跳信号停了，表示规划节点出问题，需要悬停
void heartbeatCallback(const std_msgs::EmptyConstPtr &msg) {
  heartbeat_time_ = ros::Time::now();
}

// 获取规划出来的轨迹消息
void polyTrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr) {
  trajMsg_ = *msgPtr; // 更新轨迹消息
  if (!receive_traj_) { // 第一次接收到消息
    trajMsg_last_ = trajMsg_; // 
    receive_traj_ = true;
  }
}

void cmdCallback(const ros::TimerEvent &e) {
  if (!receive_traj_) { //  没收到规划出来的路径
    // ROS_INFO("[Traj server]:cmdCallback not received traj");
    return;
  }
  ros::Time time_now = ros::Time::now();
  if ((time_now - heartbeat_time_).toSec() > 0.5) { // 失去心跳信号
    ROS_ERROR_ONCE("[traj_server] Lost heartbeat from the planner, is he dead?");
    // 在上一次的位置悬停
    publish_cmd(trajMsg_.traj_id, last_p_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0, 0);  // TODO yaw
    return;
  }
  if (exe_traj(trajMsg_)) { // 执行轨迹
    trajMsg_last_ = trajMsg_;
    return;
  } else if (exe_traj(trajMsg_last_)) { // 执行失败则执行上次节点，失败原因有轨迹有问题，轨迹不够长
    return;
  }
  // ROS_ERROR_ONCE("[traj server] traj received invalid!");
  // publish_cmd(trajMsg_.traj_id, last_p_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0, 0);  // TODO yaw
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  ros::Subscriber poly_traj_sub = nh.subscribe("trajectory", 10, polyTrajCallback); //订阅规划出来的多项式
  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback); // 订阅心跳信号

  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 50); // 发布控制指令

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback); // 每0.01s执行一次

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}