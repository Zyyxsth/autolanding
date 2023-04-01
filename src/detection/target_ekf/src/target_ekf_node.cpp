#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <object_detection_msgs/BoundingBoxes.h>
#include <ros/ros.h>

#include <Eigen/Geometry>

#include "target_ekf/AprilTagDetection.h"
#include "target_ekf/AprilTagDetectionArray.h"



// typedef message_filters::sync_policies::ApproximateTime<object_detection_msgs::BoundingBoxes, nav_msgs::Odometry>
//     YoloOdomSyncPolicy;
// typedef message_filters::Synchronizer<YoloOdomSyncPolicy>
//     YoloOdomSynchronizer;

// by HJ
typedef message_filters::sync_policies::ApproximateTime<target_ekf::AprilTagDetectionArray, nav_msgs::Odometry>
    ApriltagOdomSyncPolicy;
typedef message_filters::Synchronizer<ApriltagOdomSyncPolicy>
    ApriltagOdomSynchronizer;

ros::Publisher target_odom_pub_, yolo_odom_pub_;

Eigen::Matrix3d cam2body_R_front_, cam2body_R_down_;
Eigen::Vector3d cam2body_p_front_, cam2body_p_down_;  //因为有两个摄像头吗，所以一个是前面的，一个是底下的

Eigen::Matrix3d cam2body_R_realsense;
Eigen::Vector3d cam2body_p_realsense;       //深度摄像机realsense的位置
ros::Publisher april_odom_realsense;
double fx_r,fy_r,cx_r,cy_r;


ros::Publisher april_odom_front, april_odom_down;

double fx_, fy_, cx_, cy_;
ros::Time last_update_stamp_;
double pitch_thr_ = 30;

Eigen::Quaterniond q_;

struct Ekf {
  double dt;
  Eigen::MatrixXd A, B, C;               //A就是A，B就是B，C是指的是H
  Eigen::MatrixXd Qt, Rt;                //Qt是过程噪声的协方差矩阵,Rt是测量噪声的协方差矩阵
  Eigen::MatrixXd Sigma, K;            //sigma先验误差的协方差矩阵, K 卡尔曼增益
  Eigen::VectorXd x;                    //前面3个是位置，后面3个是速度

  Ekf(double _dt) : dt(_dt) {
    A.setIdentity(6, 6);                              //6x6
    Sigma.setZero(6, 6); // 
    B.setZero(6, 3);
    C.setZero(3, 6);
    A(0, 3) = dt;
    A(1, 4) = dt;
    A(2, 5) = dt;
    double t2 = dt * dt / 2;
    B(0, 0) = t2;
    B(1, 1) = t2;
    B(2, 2) = t2;
    B(3, 0) = dt;
    B(4, 1) = dt;
    B(5, 2) = dt;
    C(0, 0) = 1;
    C(1, 1) = 1;
    C(2, 2) = 1;
    K = C;
    Qt.setIdentity(3, 3);
    Rt.setIdentity(3, 3);
    Qt(0, 0) = 4;                                  //Q是过程噪声的协方差矩阵      
    Qt(1, 1) = 4;
    Qt(2, 2) = 1;
    Rt(0, 0) = 0.1;                                //R是测量噪声的协方差矩阵
    Rt(1, 1) = 0.1;
    Rt(2, 2) = 0.1;
    x.setZero(6);
  }


  inline void predict() {
    x = A * x;
    Sigma = A * Sigma * A.transpose() + B * Qt * B.transpose();   //先验误差的协方差矩阵为：〖P^−〗_k=A〖P^−〗_(k−1) A^T+B_K  Q〖B_K〗^T
    return;
  }
  inline void reset(const Eigen::Vector3d& z) {
    x.head(3) = z;
    x.tail(3).setZero();
    Sigma.setZero();
  }
  inline bool checkValid(const Eigen::Vector3d& z) const {                      //z明显是测量出来的
    Eigen::MatrixXd K_tmp = Sigma * C.transpose() * (C * Sigma * C.transpose() + Rt).inverse();  //K_k=P^− H^T 〖(HP^− H^T+C_k R〖C_k〗^T)〗^(−1)
    Eigen::VectorXd x_tmp = x + K_tmp * (z - C * x);   //那么再求后验估计：X ̂_k=〖X ̂_k〗^−+K_k (Z_k−hx ̂_k, 0))
    const double vmax = 4;
    if (x_tmp.tail(3).norm() > vmax) {      //速度不能超过临界值
      return false;
    } else {
      return true;
    }
  }
  inline void update(const Eigen::Vector3d& z) {
    K = Sigma * C.transpose() * (C * Sigma * C.transpose() + Rt).inverse();  //更新卡尔曼滤波的增益
    x = x + K * (z - C * x);                             //这就是求出来的后验估计X ̂_k=〖X ̂_k〗^−+K_k (Z_k−hx ̂_k, 0))
    Sigma = Sigma - K * C * Sigma;                       //P =(I−K_k H) P^− 
  }

  // 取前三个作为位置
  inline const Eigen::Vector3d pos() const {
    return x.head(3);
  }

  // 取后三个作为速度
  inline const Eigen::Vector3d vel() const {
    return x.tail(3);
  }
};

std::shared_ptr<Ekf> ekfPtr_;

// 预测得到目标的位置和速度
void predict_state_callback(const ros::TimerEvent& event) {
  double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
  if (update_dt < 3.0) {
    ekfPtr_->predict(); // 调用
  } else {
    ROS_WARN("too long time no update!");
    return;
  }
  // publish target odom
  nav_msgs::Odometry target_odom;             //汽车的里程计的相关信息
  target_odom.header.stamp = ros::Time::now();
  target_odom.header.frame_id = "world";
  target_odom.pose.pose.position.x = ekfPtr_->pos().x();
  target_odom.pose.pose.position.y = ekfPtr_->pos().y();
  target_odom.pose.pose.position.z = ekfPtr_->pos().z();
  target_odom.twist.twist.linear.x = ekfPtr_->vel().x();
  target_odom.twist.twist.linear.y = ekfPtr_->vel().y();
  target_odom.twist.twist.linear.z = ekfPtr_->vel().z();
  target_odom.pose.pose.orientation.w = 1.0; // 姿态不管
  target_odom_pub_.publish(target_odom);
}


void update_state_realsense_callback(const target_ekf::AprilTagDetectionArrayConstPtr& april_tag_msg, const nav_msgs::OdometryConstPtr& odom_msg) {
  // std::cout << "yolo stamp: " << bboxes_msg->header.stamp << std::endl;
  // std::cout << "odom stamp: " << odom_msg->header.stamp << std::endl;

    if (april_tag_msg->detections.size() != 0) {
      ROS_INFO("realsense find apriltag!");
  }

  Eigen::Vector3d odom_p;  // NOTE: (By HJ)这是VIO提供的odom 为机体的位姿，飞机的相关信息， odom下飞机的相关信息
  Eigen::Quaterniond odom_q;
  odom_p(0) = odom_msg->pose.pose.position.x;
  odom_p(1) = odom_msg->pose.pose.position.y;
  odom_p(2) = odom_msg->pose.pose.position.z;
  odom_q.w() = odom_msg->pose.pose.orientation.w;
  odom_q.x() = odom_msg->pose.pose.orientation.x;
  odom_q.y() = odom_msg->pose.pose.orientation.y;
  odom_q.z() = odom_msg->pose.pose.orientation.z;

  // 从VIO odom (飞机坐标系)得到camera的位置
  Eigen::Vector3d cam_p = odom_q.toRotationMatrix() * cam2body_p_realsense + odom_p;

  // 相机在世界系下的姿态
  Eigen::Quaterniond cam_q = odom_q * Eigen::Quaterniond(cam2body_R_realsense);

  // if there is no detection, return!
  if (april_tag_msg->detections.size() == 0) {
    // ROS_ERROR("cannot find apriltag!");
    // ROS_INFO("cannot find apriltag!");
    return;
  }

  // get relative tranlation between camera and tag // 相对位置
  double x = april_tag_msg->detections[0].pose.pose.pose.position.x;
  double y = april_tag_msg->detections[0].pose.pose.pose.position.y;
  double z = april_tag_msg->detections[0].pose.pose.pose.position.z;
  Eigen::Vector3d p(x, y, z); // 这个p是在摄像头下二维码的位置
  // std::cout << "p cam frame: " << p.transpose() << std::endl;

  // get tag's odom in the world frame 位置
  p = cam_q * p + cam_p;  //二维码标志在世界坐标系下的位置
  // 20221021 debug 
  nav_msgs::Odometry pose_april_f;
  pose_april_f.header.frame_id="world";
  pose_april_f.pose.pose.position.x=p(0);
  pose_april_f.pose.pose.position.y=p(1);
  pose_april_f.pose.pose.position.z=p(2);
  april_odom_realsense.publish(pose_april_f);

  q_.w() = april_tag_msg->detections[0].pose.pose.pose.orientation.w;
  q_.x() = april_tag_msg->detections[0].pose.pose.pose.orientation.x;
  q_.y() = april_tag_msg->detections[0].pose.pose.pose.orientation.y;
  q_.z() = april_tag_msg->detections[0].pose.pose.pose.orientation.z;

  // get tag's odom in the world frame 姿态
  q_ = cam_q * q_;

  // update target odom
  double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
  if (update_dt > 3.0) {
    ekfPtr_->reset(p);                     //超过3s二维码速度设为0，不知道为什么
    ROS_WARN("[realsense] ekf reset!");
  } else if (ekfPtr_->checkValid(p)) { 
    ekfPtr_->update(p); // 在这里update
  } else {
    ROS_ERROR("update invalid!");
    return;
  }
  last_update_stamp_ = ros::Time::now();
}



void update_state_front_callback(const target_ekf::AprilTagDetectionArrayConstPtr& april_tag_msg, const nav_msgs::OdometryConstPtr& odom_msg) {
  // std::cout << "yolo stamp: " << bboxes_msg->header.stamp << std::endl;
  // std::cout << "odom stamp: " << odom_msg->header.stamp << std::endl;
  if (april_tag_msg->detections.size() != 0) {
      ROS_INFO("front usb_cam find apriltag!");
  }

  Eigen::Vector3d odom_p;  // NOTE: (By HJ)这是VIO提供的odom 为机体的位姿
  Eigen::Quaterniond odom_q;
  odom_p(0) = odom_msg->pose.pose.position.x;
  odom_p(1) = odom_msg->pose.pose.position.y;
  odom_p(2) = odom_msg->pose.pose.position.z;
  odom_q.w() = odom_msg->pose.pose.orientation.w;
  odom_q.x() = odom_msg->pose.pose.orientation.x;
  odom_q.y() = odom_msg->pose.pose.orientation.y;
  odom_q.z() = odom_msg->pose.pose.orientation.z;

  // 从VIO odom (飞机坐标系)得到camera的位置
  Eigen::Vector3d cam_p = odom_q.toRotationMatrix() * cam2body_p_front_ + odom_p;  //差不多就是求世界系下的摄像头的位置

  // 相机在世界系下的姿态（VIO约等于世界系）
  Eigen::Quaterniond cam_q = odom_q * Eigen::Quaterniond(cam2body_R_front_);          //求世界系下摄像头的姿态

  // if there is no detection, return!
  if (april_tag_msg->detections.size() == 0) {                         //没有找到检测物，返回
    // ROS_ERROR("cannot find apriltag!");
    // ROS_INFO("front camera cannot find apriltag!");
    return;
  }
  

  // get relative tranlation between camera and tag // 相对位置
  double x = april_tag_msg->detections[0].pose.pose.pose.position.x;
  double y = april_tag_msg->detections[0].pose.pose.pose.position.y;
  double z = april_tag_msg->detections[0].pose.pose.pose.position.z;
  Eigen::Vector3d p(x, y, z); // 这个p是在摄像头下二维码的位置
  // std::cout << "p cam frame: " << p.transpose() << std::endl;

  // get tag's odom in the world frame 位置
  p = cam_q * p + cam_p;
  // 20221021 debug 
  nav_msgs::Odometry pose_april_f;
  pose_april_f.header.frame_id="world";
  pose_april_f.pose.pose.position.x=p(0);
  pose_april_f.pose.pose.position.y=p(1);
  pose_april_f.pose.pose.position.z=p(2);
  april_odom_front.publish(pose_april_f);

  q_.w() = april_tag_msg->detections[0].pose.pose.pose.orientation.w;  //这个q_是在摄像头下二维码的姿态
  q_.x() = april_tag_msg->detections[0].pose.pose.pose.orientation.x;
  q_.y() = april_tag_msg->detections[0].pose.pose.pose.orientation.y;
  q_.z() = april_tag_msg->detections[0].pose.pose.pose.orientation.z;

  // get tag's odom in the world frame 姿态
  q_ = cam_q * q_;  //

  // update target odom
  double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
  if (update_dt > 3.0) {
    ekfPtr_->reset(p);
    ROS_WARN("ekf reset!");
  } else if (ekfPtr_->checkValid(p)) {                 //检查是否超过了速度的限制
    ekfPtr_->update(p); // 在这里update
  } else {
    ROS_ERROR("update invalid!");
    return;
  }
  last_update_stamp_ = ros::Time::now();
}


void update_state_down_callback(const target_ekf::AprilTagDetectionArrayConstPtr& april_tag_msg, const nav_msgs::OdometryConstPtr& odom_msg) {
  // std::cout << "yolo stamp: " << bboxes_msg->header.stamp << std::endl;
  // std::cout << "odom stamp: " << odom_msg->header.stamp << std::endl;
  if (april_tag_msg->detections.size() != 0) {
      ROS_INFO("down usb_cam find apriltag!");
  }

  Eigen::Vector3d odom_p;  // NOTE: (By HJ)这是VIO提供的odom
  Eigen::Quaterniond odom_q;
  odom_p(0) = odom_msg->pose.pose.position.x;
  odom_p(1) = odom_msg->pose.pose.position.y;
  odom_p(2) = odom_msg->pose.pose.position.z;
  odom_q.w() = odom_msg->pose.pose.orientation.w;
  odom_q.x() = odom_msg->pose.pose.orientation.x;
  odom_q.y() = odom_msg->pose.pose.orientation.y;
  odom_q.z() = odom_msg->pose.pose.orientation.z;

  // 从VIO odom (飞机坐标系)得到camera的坐标
  Eigen::Vector3d cam_p = odom_q.toRotationMatrix() * cam2body_p_down_ + odom_p;
  Eigen::Quaterniond cam_q = odom_q * Eigen::Quaterniond(cam2body_R_down_);

  // if there is no detection, return!
  if (april_tag_msg->detections.size() == 0) {
    // ROS_ERROR("cannot find apriltag!");
    return;
  }

  // get relative tranlation between camera and tag 相对位置
  double x = april_tag_msg->detections[0].pose.pose.pose.position.x;
  double y = april_tag_msg->detections[0].pose.pose.pose.position.y;
  double z = april_tag_msg->detections[0].pose.pose.pose.position.z;
  Eigen::Vector3d p(x, y, z);
  // std::cout << "p cam frame: " << p.transpose() << std::endl;

  // get tag's odom in the world frame
  p = cam_q * p + cam_p;
  
    // 20221021 debug 
  nav_msgs::Odometry pose_april_d;
  pose_april_d.header.frame_id="world";
  pose_april_d.pose.pose.position.x=p(0);
  pose_april_d.pose.pose.position.y=p(1);
  pose_april_d.pose.pose.position.z=p(2);
  april_odom_down.publish(pose_april_d);

  q_.w() = april_tag_msg->detections[0].pose.pose.pose.orientation.w;
  q_.x() = april_tag_msg->detections[0].pose.pose.pose.orientation.x;
  q_.y() = april_tag_msg->detections[0].pose.pose.pose.orientation.y;
  q_.z() = april_tag_msg->detections[0].pose.pose.pose.orientation.z;

  q_ = cam_q * q_;

  // update target odom
  double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
  if (update_dt > 3.0) {
    ekfPtr_->reset(p);
    ROS_WARN("ekf reset!");
  } else if (ekfPtr_->checkValid(p)) {
    ekfPtr_->update(p);
  } else {
    ROS_ERROR("update invalid!");
    return;
  }
  last_update_stamp_ = ros::Time::now();
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "target_ekf");
  ros::NodeHandle nh("~");
  // last_update_stamp_ = ros::Time::now() - ros::Duration(10.0);
  last_update_stamp_ = ros::Time::now();
  ROS_INFO("[target_ekf] started");

  
  std::vector<double> tmp;
  if (nh.param<std::vector<double>>("cam2body_R_front", tmp, std::vector<double>())) {
    cam2body_R_front_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
  }
  if (nh.param<std::vector<double>>("cam2body_p_front", tmp, std::vector<double>())) {
    cam2body_p_front_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
  }
  if (nh.param<std::vector<double>>("cam2body_R_down", tmp, std::vector<double>())) {
    cam2body_R_down_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
  }
  if (nh.param<std::vector<double>>("cam2body_p_down", tmp, std::vector<double>())) {
    cam2body_p_down_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
  }

  //realsense 
  if (nh.param<std::vector<double>>("cam2body_R_realsense", tmp, std::vector<double>())) {
    cam2body_R_realsense = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
  }
  if (nh.param<std::vector<double>>("cam2body_p_realsense", tmp, std::vector<double>())) {
    cam2body_p_realsense = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
  }

  // 这几个参数似乎没有用
  nh.getParam("cam_fx", fx_);
  nh.getParam("cam_fy", fy_);
  nh.getParam("cam_cx", cx_);
  nh.getParam("cam_cy", cy_);
  nh.getParam("pitch_thr", pitch_thr_);
  nh.getParam("realsense_cam_fx", fx_r);
  nh.getParam("realsense_cam_fy", fy_r);
  nh.getParam("realsense_cam_cx", cx_r);
  nh.getParam("realsense_cam_cy", cy_r);

  // message_filters::Subscriber<object_detection_msgs::BoundingBoxes> yolo_sub_;
  message_filters::Subscriber<target_ekf::AprilTagDetectionArray> apriltag_sub_front_, apriltag_sub_down_,apriltag_sub_realsense;

  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;

  // std::shared_ptr<YoloOdomSynchronizer> yolo_odom_sync_Ptr_;
  std::shared_ptr<ApriltagOdomSynchronizer> apriltag_odom_sync_Ptr_front, apriltag_odom_sync_Ptr_down,apriltag_odom_sync_Ptr_realsense;

  ros::Timer ekf_predict_timer_;
  // ros::Subscriber single_odom_sub = nh.subscribe("odom", 100, &odom_callback, ros::TransportHints().tcpNoDelay());
  target_odom_pub_ = nh.advertise<nav_msgs::Odometry>("target_odom", 1);
  // 发布的里程计
  // yolo_odom_pub_ = nh.advertise<nav_msgs::Odometry>("yolo_odom", 1);

  // 发布的监测信息
  april_odom_front = nh.advertise<nav_msgs::Odometry>("april_odom_front", 1); // 发布
  april_odom_down = nh.advertise<nav_msgs::Odometry>("april_odom_down", 1);
  april_odom_realsense=nh.advertise<nav_msgs::Odometry>("april_odom_realsense",1);
  //advertise()函数是你告诉ROS你想在给定的话题名上发布特定类型的消息。
     //在这个advertise()调用之后，master节点将通知任何试图订阅这个话题名称的节点，然后他们将与这个节点建立一个对等网络（peer to peer / P2P）连接。
      //advertise()括号里面的第一个参数是话题名字，第二个参数是用于发布消息的消息队列的大小。
     //<>里面指定消息的类型

  int ekf_rate = 20;
  nh.getParam("ekf_rate", ekf_rate);
  ekfPtr_ = std::make_shared<Ekf>(1.0 / ekf_rate);

  // yolo_sub_.subscribe(nh, "yolo", 1, ros::TransportHints().tcpNoDelay());
  odom_sub_.subscribe(nh, "/vins_fusion/imu_propagate", 100, ros::TransportHints().tcpNoDelay());
  // odom_sub_.subscribe(nh, "/vins_fusion/odometry", 100, ros::TransportHints().tcpNoDelay());
  apriltag_sub_front_.subscribe(nh, "/april_tag_front/tag_detections", 100, ros::TransportHints().tcpNoDelay());
  apriltag_sub_down_.subscribe(nh, "/april_tag_down/tag_detections", 100, ros::TransportHints().tcpNoDelay());
  apriltag_sub_realsense.subscribe(nh, "/realsense/tag_detections", 100, ros::TransportHints().tcpNoDelay());

  // 同步
  apriltag_odom_sync_Ptr_front = std::make_shared<ApriltagOdomSynchronizer>(ApriltagOdomSyncPolicy(200), apriltag_sub_front_, odom_sub_);
  apriltag_odom_sync_Ptr_front->registerCallback(boost::bind(&update_state_front_callback, _1, _2));

  apriltag_odom_sync_Ptr_down = std::make_shared<ApriltagOdomSynchronizer>(ApriltagOdomSyncPolicy(200), apriltag_sub_down_, odom_sub_);
  apriltag_odom_sync_Ptr_down->registerCallback(boost::bind(&update_state_down_callback, _1, _2));

  apriltag_odom_sync_Ptr_realsense = std::make_shared<ApriltagOdomSynchronizer>(ApriltagOdomSyncPolicy(200), apriltag_sub_realsense, odom_sub_);
  apriltag_odom_sync_Ptr_realsense->registerCallback(boost::bind(&update_state_realsense_callback, _1, _2));


  // ekf
  ekf_predict_timer_ = nh.createTimer(ros::Duration(1.0 / ekf_rate), &predict_state_callback);

  ros::spin();
  return 0;
}
