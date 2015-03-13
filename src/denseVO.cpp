#define _CRT_SECURE_NO_WARNINGS

//for ROS
#include "ros/ros.h"
#include "tf/transform_broadcaster.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Path.h"
#include "visualization_msgs/Marker.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/PointCloud.h"

//for c++ std library
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <omp.h>
//#include <boost/thread.hpp>

//for SLAM
#include "kMeansClustering.h"
#include "planeExtraction.h"
#include "stateEstimation.h"
#include "variableDefinition.h"
#include "testDataGeneration.h"

//For Eigen
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"

//for openCV
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"

//for sensor driver
#include "PrimeSenseCam.h"

using namespace std;
using namespace cv;
using namespace Eigen;

CAMER_PARAMETERS cameraParameters(535.4, 539.2, 320.1, 247.6);//TUM Freiburg 3 sequences
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6,	255.3,	0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6, 255.3);//TUM Freiburg 1 sequences
STATEESTIMATION slidingWindows(IMAGE_HEIGHT, IMAGE_WIDTH, &cameraParameters);
Matrix3d firstFrameRtoVICON;
Vector3d firstFrameTtoVICON;

//PrimeSenseCam cam;

ros::Publisher pub_path ;
ros::Publisher pub_odometry ;
ros::Publisher pub_pose ;
ros::Publisher pub_cloud ;
image_transport::Publisher pub_grayImage ;
image_transport::Publisher pub_depthImage ;
image_transport::Subscriber sub_image;
image_transport::Subscriber sub_depth;
visualization_msgs::Marker path_line;

Mat rgbImage, depthImage;
Mat grayImage[maxPyramidLevel];
STATE tmpState;
STATE* lastFrame;

bool vst = false;
Matrix3d R_k_c;//R_k^(k+1)
Matrix3d R_c_0;
Vector3d T_k_c;//T_k^(k+1)
Vector3d T_c_0;

Vector3d R_to_ypr(const Matrix3d& R)
{
  Vector3d n = R.col(0);
  Vector3d o = R.col(1);
  Vector3d a = R.col(2);
  Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;
  return 180.0 / PI * ypr;
}

void frameToFrameDenseTracking(Matrix3d& R_k_c, Vector3d& T_k_c)
{
    Matrix3d nextR = Matrix3d::Identity();
    Vector3d nextT = Vector3d::Zero();
    slidingWindows.denseTrackingWithoutSuperpixel(lastFrame, grayImage, nextR, nextT);

    T_k_c = nextR * T_k_c + nextT;
    R_k_c = nextR * R_k_c;
}

void keyframeToFrameDenseTracking(Matrix3d& R_k_c, Vector3d& T_k_c )
{
    STATE* keyframe = &slidingWindows.states[slidingWindows.tail];
    slidingWindows.denseTrackingWithoutSuperpixel(keyframe, grayImage, R_k_c, T_k_c);
}

void RtoEulerAngles(Matrix3d R, double a[3])
{
    double theta = acos(0.5*(R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
    a[0] = (R(2, 1) - R(1, 2)) / (2.0* sin(theta));
    a[1] = (R(0, 2) - R(2, 0)) / (2.0* sin(theta));
    a[2] = (R(1, 0) - R(0, 1)) / (2.0* sin(theta));
}

void initCalibrationParameters()
{
    FileStorage fs("/home/nova/calibration/camera_rgbd_sensor.yml", FileStorage::READ);

    Mat cameraMatrix ;
    Mat distortionCoff ;

    if (fs.isOpened() == false ){
        puts("Can not open") ;
        return ;
    }

    fs["camera_matrix"] >> cameraMatrix ;
    fs["distortion_coefficients"] >> distortionCoff ;

    cameraParameters.cameraMatrix = cameraMatrix ;
    cameraParameters.setParameters(
                cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1),
                cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));
    cameraParameters.distCoeffs = distortionCoff ;

    fs.release();
}

void pubOdometry(const Vector3d& p, const Matrix3d& R )
{
    nav_msgs::Odometry odometry;
    Quaterniond q(R) ;

    odometry.header.stamp = ros::Time::now();
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = p(0);
    odometry.pose.pose.position.y = p(1);
    odometry.pose.pose.position.z = p(2);
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    pub_pose.publish(pose_stamped);
}

void pubPath(const Vector3d& p)
{
    geometry_msgs::Point pose_p;
    pose_p.x = p(0);
    pose_p.y = p(1);
    pose_p.z = p(2);

    path_line.points.push_back(pose_p);
    path_line.scale.x = 0.01 ;
    pub_path.publish(path_line);
}
/*
bool fun()
{
    ros::Rate loop_rate(0.1) ;

    Mat rgbImage, depthImage;
    //ofstream fileOutput("result.txt");

    for ( int i = 1; ros::ok() ; i += 1 )
    {
        printf("id : %d\n", i);

        cam.retriveFrame( rgbImage, depthImage) ;

        cvtColor(rgbImage, grayImage[0], CV_BGR2GRAY);

#ifdef DOWNSAMPLING
        pyrDown(grayImage[0], grayImage[0]);//down-sampling
#endif
        for (int kk = 1; kk < maxPyramidLevel; kk++){
            pyrDown(grayImage[kk-1], grayImage[kk]);//down-sampling
        }

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grayImage[0]).toImageMsg() ;
        pub_grayImage.publish( msg ) ;

        sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono16", depthImage).toImageMsg() ;
        pub_depthImage.publish( msg2 ) ;

        depthImage.convertTo(depthImage, CV_32F );

#ifdef DOWNSAMPLING
        pyrDown(depthImage, depthImage ) ;
#endif

        if (vst == false )//the first frame
        {
            vst = true;

            slidingWindows.insertKeyFrame(grayImage, depthImage, Matrix3d::Identity(), Vector3d::Zero() );
            slidingWindows.planeDection();

            R_k_c = Matrix3d::Identity();
            T_k_c = Vector3d::Zero();

            lastFrame = &slidingWindows.states[slidingWindows.tail];

            continue;
        }

        //double t = (double)cvGetTickCount();

        frameToFrameDenseTracking(R_k_c, T_k_c );
        //keyframeToFrameDenseTracking(R_k_c, T_k_c );

        //t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
        //printf("cal time: %f\n", t);

        R_c_0 = slidingWindows.states[slidingWindows.tail].R_k0*R_k_c.transpose();
        T_c_0 = R_c_0*(
            R_k_c*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - T_k_c);

        pubOdometry(T_c_0, R_c_0);
        pubPath(T_c_0);

        cout << "currentPosition:\n" << T_c_0.transpose() << endl;

        if ((i % 10) == 1)
        {
            slidingWindows.insertKeyFrame(grayImage, depthImage, R_c_0, T_c_0 );

//            cout << "estimate position[before BA]:\n"
//                << slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;

//            double t = (double)cvGetTickCount();

//            slidingWindows.PhotometricBA();

//            t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
//            printf("BA cal time: %f\n", t);

//            slidingWindows.planeDection();

            R_k_c = Matrix3d::Identity();
            T_k_c = Vector3d::Zero();

            lastFrame = &slidingWindows.states[slidingWindows.tail];

//            cout << "estimate position[after BA]:\n"
//                << slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;

//            Vector3d groundTruthT;
//            groundTruthT << groundTruth[timeID].tx, groundTruth[timeID].ty, groundTruth[timeID].tz;
//            groundTruthT = firstFrameRtoVICON.transpose()*(groundTruthT - firstFrameTtoVICON);

//            cout << "ground truth position:\n"
//                << groundTruthT.transpose() << endl;

//            fileOutput << slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;
//            fileOutput << groundTruthT.transpose() << endl;

//            Quaterniond q;
//            Matrix3d truthR;
//            q.x() = groundTruth[timeID].qx;
//            q.y() = groundTruth[timeID].qy;
//            q.z() = groundTruth[timeID].qz;
//            q.w() = groundTruth[timeID].qw;
//            truthR = q.toRotationMatrix();

//            double estimateEularAngels[3];
//            double groundEularAngels[3];

//            RtoEulerAngles(firstFrameRtoVICON*slidingWindows.states[slidingWindows.tail].R_k0, estimateEularAngels);
//            RtoEulerAngles(truthR, groundEularAngels);
//            //RtoEulerAngles(slidingWindows.states[slidingWindows.tail].R_k0, estimateEularAngels);
//            //RtoEulerAngles(firstFrameRtoVICON.transpose()*truthR, groundEularAngels);

//            cout << "estimate angels:\n" << estimateEularAngels[0] << " " << estimateEularAngels[1] << " " << estimateEularAngels[2] << endl;
//            cout << "ground truth angels:\n" << groundEularAngels[0] << " " << groundEularAngels[1] << " " << groundEularAngels[2] << endl;

//            fileOutput << estimateEularAngels[0] << " " << estimateEularAngels[1] << " " << estimateEularAngels[2]  << endl;
//            fileOutput << groundEularAngels[0] << " " << groundEularAngels[1] << " " << groundEularAngels[2] << endl;

        }
        else
        {
            lastFrame = &tmpState;
            tmpState.insertFrame(grayImage, depthImage, R_c_0, T_c_0, slidingWindows.para );
        }

//        ros::Time ros_t = ros::Time::now() ;

//        vector<Vector3d> pointCloud ;
//        vector<unsigned short>R, G, B ;
//        vector<Vector3d> ps ;
//        vector<Matrix3d> Rs ;
//        slidingWindows.prepareDateForVisualization(pointCloud, R, G, B, ps, Rs );
//        publish_all(pointCloud, R, G, B, ps, Rs, ros_t, currentR, currentT ) ;
//        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grayImage[0]).toImageMsg() ;
//        pubImage.publish( msg ) ;



        //ros::spinOnce() ;
        //loop_rate.sleep() ;
        //imshow("image",  grayImage[0]) ;
        //waitKey(30) ;
    }

    return true ;
}
*/
void init()
{
    //initCalibrationParameters() ;
    //cam.start();
}

int cnt = 0;

void estimateCurrentState()
{
    cnt++ ;
    if (vst == false )//the first frame
    {
        vst = true;

        slidingWindows.insertKeyFrame(grayImage, depthImage, Matrix3d::Identity(), Vector3d::Zero() );
        //slidingWindows.planeDection();

        R_k_c = Matrix3d::Identity();
        T_k_c = Vector3d::Zero();

        lastFrame = &slidingWindows.states[slidingWindows.tail];
        cnt = 1 ;
        return ;
    }

    //double t = (double)cvGetTickCount();

    frameToFrameDenseTracking(R_k_c, T_k_c );
    //keyframeToFrameDenseTracking(R_k_c, T_k_c );

    //t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
    //printf("cal time: %f\n", t);

    R_c_0 = slidingWindows.states[slidingWindows.tail].R_k0*R_k_c.transpose();
    T_c_0 = R_c_0*(
                R_k_c*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - T_k_c);

    if ((cnt % 10) == 1)
    {
        slidingWindows.insertKeyFrame(grayImage, depthImage, R_c_0, T_c_0 );

        pubOdometry(T_c_0, R_c_0);
        pubPath(T_c_0);
        cout << cnt/10 << "-" << "currentPosition:\n" << T_c_0.transpose() << endl;

        R_k_c = Matrix3d::Identity();
        T_k_c = Vector3d::Zero();
        lastFrame = &slidingWindows.states[slidingWindows.tail];
    }
    else
    {
        lastFrame = &tmpState;
        tmpState.insertFrame(grayImage, depthImage, R_c_0, T_c_0, slidingWindows.para );
    }
}

int rgbImageNum = 0 ;

void rgbImageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    //cout << "rgb: " << msg->header.stamp << endl ;
    //cout << "rgbImageNum = " << rgbImageNum++ << endl ;
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    cvtColor(cv_ptr->image, grayImage[0], CV_BGR2GRAY);

#ifdef DOWNSAMPLING
    pyrDown(grayImage[0], grayImage[0]);//down-sampling
#endif

    for (int kk = 1; kk < maxPyramidLevel; kk++){
        pyrDown(grayImage[kk-1], grayImage[kk]);//down-sampling
    }
    if ( depthImage.rows != grayImage[0].rows || depthImage.cols != grayImage[0].cols ){
        return ;
    }

    depthImage.convertTo(depthImage, CV_32F );
    //depthImage /= depthFactor;

#ifdef DOWNSAMPLING
    pyrDown(depthImage, depthImage ) ;
#endif

    estimateCurrentState() ;
}


void grayImageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    grayImage[0] = cv_ptr->image.clone() ;

#ifdef DOWNSAMPLING
    pyrDown(grayImage[0], grayImage[0]);//down-sampling
#endif

    for (int kk = 1; kk < maxPyramidLevel; kk++){
        pyrDown(grayImage[kk-1], grayImage[kk]);//down-sampling
    }
    if ( depthImage.rows != grayImage[0].rows || depthImage.cols != grayImage[0].cols ){
        return ;
    }

    depthImage.convertTo(depthImage, CV_32F );

#ifdef DOWNSAMPLING
    pyrDown(depthImage, depthImage ) ;
#endif

    estimateCurrentState() ;
}

void depthImageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    //cout << "depth: " << msg->header.stamp << endl ;
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1 );
    depthImage = cv_ptr->image.clone() ;
}

int main(int argc, char** argv )
{
    //init() ;
    ros::init(argc, argv, "denseVO") ;
    ros::NodeHandle n ;
    //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    image_transport::ImageTransport it(n) ;

    //sub_image = it.subscribe("camera/grayImage", 1, &grayImageCallBack);
    sub_image = it.subscribe("/camera/rgb/image_color", 1, &rgbImageCallBack);
    sub_depth = it.subscribe("/camera/depth/image", 1, &depthImageCallBack);

    pub_path = n.advertise<visualization_msgs::Marker>("/denseVO/path", 1000);;
    pub_odometry = n.advertise<nav_msgs::Odometry>("/denseVO/odometry", 1000);;
    pub_pose = n.advertise<geometry_msgs::PoseStamped>("/denseVO/pose", 1000);
    pub_cloud = n.advertise<sensor_msgs::PointCloud>("/denseVO/cloud", 1000);

    //pub_grayImage = it.advertise("camera/grayImage", 1 );
    //pub_depthImage = it.advertise("camera/depthImage", 1 ) ;

    path_line.header.frame_id    = "world";
    path_line.header.stamp       = ros::Time::now();
    path_line.ns                 = "dense_vo";
    path_line.action             = visualization_msgs::Marker::ADD;
    path_line.pose.orientation.w = 1.0;
    path_line.type               = visualization_msgs::Marker::LINE_STRIP;
    path_line.scale.x            = 0.01 ;
    path_line.color.a            = 1.0;
    path_line.color.r            = 1.0;
    path_line.id                 = 1;
    path_line.points.push_back( geometry_msgs::Point());
    pub_path.publish(path_line);

    //fun() ;

    ros::spin();

    return 0;
}
