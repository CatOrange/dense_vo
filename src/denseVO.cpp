#define _CRT_SECURE_NO_WARNINGS

//unroll loops, utility, unroller

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
#include "sensor_msgs/fill_image.h"
#include "opencv2/gpu/gpu.hpp"

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
//#include "testDataGeneration.h"

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
#include "opencv2/features2d/features2d.hpp"

//for sensor driver
#include "PrimeSenseCam.h"

//for multithread
//#include <boost/thread.hpp>
//boost::mutex mtx;

using namespace std;
using namespace cv;
using namespace Eigen;

CAMER_PARAMETERS cameraParameters;
//CAMER_PARAMETERS cameraParameters(535.4, 539.2, 320.1, 247.6);//TUM Freiburg 3 sequences
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6,	255.3,	0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6, 255.3);//TUM Freiburg 1 sequences
STATEESTIMATION slidingWindows(IMAGE_HEIGHT, IMAGE_WIDTH, &cameraParameters);
Matrix3f firstFrameRtoVICON;
Vector3f firstFrameTtoVICON;

//PrimeSenseCam cam;

ros::Publisher pub_path ;
ros::Publisher pub_odometry ;
ros::Publisher pub_pose ;
ros::Publisher pub_cloud ;
ros::Publisher pub_grayImage ;
ros::Publisher pub_resudualMap ;
ros::Publisher pub_gradientMapForDebug ;
ros::Subscriber sub_image;
visualization_msgs::Marker path_line;

Mat depthImage[maxPyramidLevel*bufferSize];
Mat grayImage[maxPyramidLevel*bufferSize];
STATE tmpState;
STATE* lastFrame;

bool vst = false;
int rgbImageNum = 0 ;
int bufferHead = 0;
int bufferTail = 0 ;
Matrix3f R_k_c;//R_k^(k+1)
Matrix3f R_c_0;
Vector3f T_k_c;//T_k^(k+1)
Vector3f T_c_0;

Vector3f R_to_ypr(const Matrix3f& R)
{
  Vector3f n = R.col(0);
  Vector3f o = R.col(1);
  Vector3f a = R.col(2);
  Vector3f ypr(3);
  float y = atan2(n(1), n(0));
  float p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  float r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;
  return 180.0 / PI * ypr;
}

void RtoEulerAngles(Matrix3f R, float a[3])
{
    float theta = acos(0.5*(R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
    a[0] = (R(2, 1) - R(1, 2)) / (2.0* sin(theta));
    a[1] = (R(0, 2) - R(2, 0)) / (2.0* sin(theta));
    a[2] = (R(1, 0) - R(0, 1)) / (2.0* sin(theta));
}

void initCalibrationParameters()
{
    //FileStorage fs("/home/nova/calibration/camera_rgbd_sensor.yml", FileStorage::READ);

    Mat cameraMatrix ;
    Mat distortionCoff ;

//    if (fs.isOpened() == false ){
//        puts("Can not open") ;
//        return ;
//    }


    cameraMatrix = cv::Mat::zeros(3, 3, CV_32F ) ;
    cameraMatrix.at<float>(0, 0) = 270.310139 ;
    cameraMatrix.at<float>(0, 2) = 157.085025 ;
    cameraMatrix.at<float>(1, 1) = 269.501236 ;
    cameraMatrix.at<float>(1, 2) = 127.390471 ;
    cameraMatrix.at<float>(2, 2) = 1.0 ;

    distortionCoff = cv::Mat::zeros(1, 4, CV_32F ) ;
    distortionCoff.at<float>(0, 0) = 0.030448 ;
    distortionCoff.at<float>(0, 1) = -0.095764 ;
    distortionCoff.at<float>(0, 2) = 0.007353999999999999 ;
    distortionCoff.at<float>(0, 3) = 0.001485 ;
    distortionCoff.at<float>(0, 4) = 0 ;

//    fs["camera_matrix"] >> cameraMatrix ;
//    fs["distortion_coefficients"] >> distortionCoff ;

    cameraParameters.cameraMatrix = cameraMatrix ;
    cameraParameters.setParameters(
                cameraMatrix.at<float>(0, 0), cameraMatrix.at<float>(1, 1),
                cameraMatrix.at<float>(0, 2), cameraMatrix.at<float>(1, 2));
    cameraParameters.distCoeffs = distortionCoff ;

//    fs.release();


}

void pubOdometry(const Vector3f& p, const Matrix3f& R )
{
    nav_msgs::Odometry odometry;
    Quaternionf q(R) ;

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

void pubPath(const Vector3f& p)
{
    geometry_msgs::Point pose_p;
    pose_p.x = p(0);
    pose_p.y = p(1);
    pose_p.z = p(2);

    path_line.points.push_back(pose_p);
    path_line.scale.x = 0.01 ;
    pub_path.publish(path_line);
}

int cnt = 0;
Mat inputImage[maxPyramidLevel] ;
Mat inputDepth[maxPyramidLevel] ;
Mat gradientMapForDebug, residualMap, display ;
sensor_msgs::Image msg;

void init()
{
    for ( int j = 0 ; j < bufferSize ; j++ )
    {
        int k = j*maxPyramidLevel ;
        for ( int i = 0 ; i < maxPyramidLevel ; i++ )
        {
            int height = IMAGE_HEIGHT >> i ;
            int width = IMAGE_WIDTH >> i ;
            grayImage[k+i] = Mat::zeros(height, width, CV_8U ) ;
            depthImage[k+i] = Mat::zeros(height, width, CV_32F ) ;
        }
    }
    initCalibrationParameters() ;
    residualMap.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U );
    //cam.start();
}

void estimateCurrentState()
{
    for ( int bufferLength = ( bufferTail - bufferHead + bufferSize ) % bufferSize ; bufferLength > 0 ; bufferLength-- )
    {
        //printf("head=%d tail=%d\n", bufferHead, bufferTail ) ;
        //cnt++ ;
        float t = (float)cvGetTickCount();
        for ( int i = 0 ; i < maxPyramidLevel ; i++ ){
            inputImage[i] = grayImage[bufferHead*maxPyramidLevel+i] ;
            inputDepth[i] = depthImage[bufferHead*maxPyramidLevel+i] ;
        }
        bufferHead++ ;
        if ( bufferHead >= bufferSize ){
            bufferHead -= bufferSize ;
        }
//        imshow("image_gray", inputImage[0] ) ;
//        waitKey(1) ;
        if (vst == false )//the first frame
        {
            vst = true;
            slidingWindows.insertFrame(grayImage, Matrix3f::Identity(), Vector3f::Zero(), Vector3f::Zero() );
            float ttt = (float)cvGetTickCount() ;
            slidingWindows.prepareKeyFrame(&slidingWindows.states[slidingWindows.tail], inputDepth, gradientMapForDebug );

            msg.header.stamp = ros::Time() ;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_WIDTH*3,
                                   gradientMapForDebug.data );
            pub_gradientMapForDebug.publish(msg) ;


            printf("prepare keyframe time: %lf\n", ((float)cvGetTickCount() - ttt) / (cvGetTickFrequency() * 1000) ) ;

            slidingWindows.R_k_2_c = Matrix3f::Identity();
            slidingWindows.T_k_2_c = Vector3f::Zero();
            continue ;
        }
        //dense tracking
        MatrixXf lastestATA ;
        float averageResidial ;
        STATE* keyFrame = slidingWindows.getKeyFrame_ptr();
        float validPercent = slidingWindows.denseTrackingHuber(keyFrame, inputImage, slidingWindows.R_k_2_c, slidingWindows.T_k_2_c, lastestATA, averageResidial);
        //slidingWindows.badPixelFiltering(keyFrame, denseTrackingImage, slidingWindows.R_k_2_c, slidingWindows.T_k_2_c) ;
        slidingWindows.visualizeResidualMap(keyFrame, inputImage, slidingWindows.R_k_2_c, slidingWindows.T_k_2_c, residualMap, display);

        msg.header.stamp = ros::Time() ;
        sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_WIDTH*3,
                               display.data );
        pub_resudualMap.publish(msg) ;

        bool insertKeyFrameFlag = false ;
        //printf("validPercent=%f averageResidial=%f\n", validPercent, averageResidial ) ;
        if ( validPercent < 0.05 || lastestATA.rows() < 6
             || slidingWindows.T_k_2_c.norm() > 0.5
           //  || averageResidial > 10.0
             ){
            insertKeyFrameFlag = true ;
        }

//    #ifdef FRAME_TO_FRAME
//        frameToFrameDenseTracking(R_k_c, T_k_c);
//    #else
//        keyframeToFrameDenseTracking( bufferHead, R_k_c, T_k_c );
//    #endif

        Matrix3f R_bk1_2_b0 = slidingWindows.states[slidingWindows.tail].R_bk_2_b0*slidingWindows.R_k_2_c.transpose();
        Vector3f T_bk1_2_b0 = slidingWindows.states[slidingWindows.tail].T_bk_2_b0 + R_bk1_2_b0*slidingWindows.T_k_2_c ;

        pubOdometry(T_bk1_2_b0, R_bk1_2_b0);
        pubPath(T_bk1_2_b0);

        //cv::Mat falseColorsMap;
        //applyColorMap(residualImage, falseColorsMap, cv::COLORMAP_RAINBOW );

        //cv::imshow("Resid", falseColorsMap);
        //cv::waitKey(10) ;

        if ( insertKeyFrameFlag )
        {
            slidingWindows.insertFrame(inputImage, R_bk1_2_b0, T_bk1_2_b0, Vector3f::Zero() );
            float ttt = (float)cvGetTickCount() ;
            slidingWindows.prepareKeyFrame(&slidingWindows.states[slidingWindows.tail], inputDepth, gradientMapForDebug );

            msg.header.stamp = ros::Time() ;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_WIDTH*3,
                                   gradientMapForDebug.data );
            pub_gradientMapForDebug.publish(msg) ;

            printf("prepare keyframe time: %lf\n", ((float)cvGetTickCount() - ttt) / (cvGetTickFrequency() * 1000) ) ;

            slidingWindows.R_k_2_c = Matrix3f::Identity();
            slidingWindows.T_k_2_c = Vector3f::Zero();
        }
        printf("dense tracking time: %f\n", ((float)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );
//        bufferHead++ ;
//        if ( bufferHead >= bufferSize ){
//            bufferHead -= bufferSize ;
//        }
    }
}

void imageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    //printf("%d\n", rgbImageNum++ ) ;
    cv::Mat currentImage = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8)->image;

    int startIndex = bufferTail * maxPyramidLevel ;
    const int h2 =  IMAGE_HEIGHT* 2 ;
    for ( int i = 0 ; i < IMAGE_HEIGHT ; i++ )
    {
        for ( int j = 0 ; j < IMAGE_WIDTH ; j++ )
        {
            grayImage[startIndex+0].at<unsigned char>(i, j) = currentImage.at<unsigned char>(i, j) ;
            unsigned short high8 = currentImage.at<unsigned char>(i+IMAGE_HEIGHT, j) ;
            unsigned short low8 = currentImage.at<unsigned char>(i+h2, j) ;
            unsigned short tmp = ( high8 << 8 ) | low8 ;
            depthImage[startIndex+0].at<float>(i, j) = (float)tmp / 1000.0 ;
        }
    }

//    imshow("grayImage", grayImage[0]) ;
//    waitKey(1) ;

//    if ( depthImage[0].rows != grayImage[0].rows || depthImage[0].cols != grayImage[0].cols ){
//        return ;
//    }
//    depthImage[0] /= 1000;

//    printf("%d %d %d %d %d\n", grayImage[0].at<uchar>(160, 120), grayImage[0].at<uchar>(10, 10),
//            grayImage[0].at<uchar>(310, 230), grayImage[0].at<uchar>(0, 230), grayImage[0].at<uchar>(310, 10) ) ;

    //printf("depth = %f\n", depthImage[0].at<float>(120, 160) ) ;

    //grayImage
//#ifdef DOWNSAMPLING
//    pyrDownMeanSmooth<uchar>(grayImage[0], grayImage[0]);
//#endif

    //depthImage[0].convertTo(depthImage[0], CV_32F );

//#ifdef DOWNSAMPLING
//    pyrDownMedianSmooth<float>(depthImage[0], depthImage[0]);
//#endif


    for (int kk = 1; kk < maxPyramidLevel; kk++){
        pyrDownMeanSmooth<uchar>(grayImage[startIndex + kk - 1], grayImage[startIndex + kk]);
        pyrDownMedianSmooth<float>(depthImage[startIndex + kk - 1], depthImage[startIndex + kk ]);
    }

    bufferTail++ ;
    if ( bufferTail >= bufferSize ){
        bufferTail -= bufferSize ;
    }
}

/*
void fun()
{
    cv::Mat rgbImage ;
    ros::Rate loop_rate(30) ;

    PrimeSenseCam cam;
    cam.start();
   // while( ros::ok() )
    while( true )
    {
        cam.retriveFrame( rgbImage, depthImage[0]) ;
        cvtColor(rgbImage, grayImage[0], CV_BGR2GRAY);
        depthImage[0] /= 1000;

        for (int kk = 1; kk < maxPyramidLevel; kk++){
            pyrDownMeanSmooth<uchar>(grayImage[kk - 1], grayImage[kk]);
        }

        for (int kk = 1; kk < maxPyramidLevel; kk++){
            pyrDownMedianSmooth<float>(depthImage[kk - 1], depthImage[kk]);
        }

        estimateCurrentState() ;
    }
}
*/

int main(int argc, char** argv )
{
    init() ;
    ros::init(argc, argv, "denseVO") ;
    ros::NodeHandle n ;
    //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);

    sub_image = n.subscribe("camera/grayAndDepthImage", 10, &imageCallBack);
    //sub_depth = it.subscribe("/camera/depthImage", 20, &depthImageCallBack);

    //sub_image = it.subscribe("/camera/rgb/image_color", 100, &rgbImageCallBack);
    //sub_depth = it.subscribe("/camera/depth/image", 100, &depthImageCallBack);

    //pub_grayImage = n.advertise<sensor_msgs::Image>("camera/grayImage", 10 ) ;
    pub_path = n.advertise<visualization_msgs::Marker>("/denseVO/path", 1000);;
    pub_odometry = n.advertise<nav_msgs::Odometry>("/denseVO/odometry", 1000);;
    pub_pose = n.advertise<geometry_msgs::PoseStamped>("/denseVO/pose", 1000);
    pub_cloud = n.advertise<sensor_msgs::PointCloud>("/denseVO/cloud", 1000);
    pub_resudualMap = n.advertise<sensor_msgs::Image>("denseVO/residualMap", 100 );
    pub_gradientMapForDebug = n.advertise<sensor_msgs::Image>("denseVO/debugMap", 100 );

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

    ros::Rate loop_rate(100.0);
    bufferHead = bufferTail = 0 ;
    while( n.ok() )
    {
        loop_rate.sleep() ;
        ros::spinOnce() ;
        //TODO
        estimateCurrentState() ;
    }

    return 0;
}
