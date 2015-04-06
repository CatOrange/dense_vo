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

//for c++ std library
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <list>
#include <omp.h>
#include <boost/thread.hpp>

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

#include "visensor_node/visensor_imu.h"
#include "visensor_node/visensor_calibration.h"

//for sensor driver
//#include "PrimeSenseCam.h"

//for multithread
//#include <boost/thread.hpp>
//boost::mutex mtx;

using namespace std;
using namespace cv;
using namespace Eigen;

//CAMER_PARAMETERS cameraParameters(535.4, 539.2, 320.1, 247.6);//TUM Freiburg 3 sequences
CAMER_PARAMETERS cameraParameters ;
CAMER_PARAMETERS stereoPara[2] ;
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
ros::Subscriber sub_image[2];
ros::Subscriber sub_imu;
ros::Subscriber sub_cali ;
visualization_msgs::Marker path_line;

list<visensor_node::visensor_imu> imuQueue ;
list<sensor_msgs::Image> imageQueue[2] ;
//list<ros::Time> imageTimeQueue[2] ;

Mat depthImage[maxPyramidLevel];
Mat grayImage[maxPyramidLevel];
Mat gradientMapForDebug ;
//Mat grayImage[maxPyramidLevel*bufferSize];
STATE tmpState;
STATE* lastFrame;

bool vst = false;
int rgbImageNum = 0 ;
Matrix3d R_k_c;//R_k^(k+1)
Matrix3d R_c_0;
Vector3d T_k_c;//T_k^(k+1)
Vector3d T_c_0;

cv::StereoBM bm_( cv::StereoBM::BASIC_PRESET, maxDisparity, 21 );
double lastTime = -1 ;

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

bool keyframeToFrameDenseTracking(Matrix3d& R_k_c, Vector3d& T_k_c )
{
    STATE* keyframe = &slidingWindows.states[slidingWindows.tail];
    return slidingWindows.denseTrackingWithoutSuperpixel(keyframe, grayImage, R_k_c, T_k_c);
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
    FileStorage fs("/home/nova/calibration/visensor.yml", FileStorage::READ);

    if (fs.isOpened() == false ){
        ROS_WARN("Can not open") ;
        return ;
    }

    char tmp[128] ;
    cv::Mat cvRic ;
    cv::Mat cvTic ;
    cv::Mat dist_coeff ;
    cv::Mat K ;
    Eigen::Matrix3d Ric ;
    Eigen::Vector3d Tic ;
    for( int camera_id = 0 ; camera_id < 2 ; camera_id++ )
    {
        std::sprintf(tmp, "Ric_%d", camera_id ) ;
        fs[tmp] >> cvRic ;
        for( int i = 0 ; i < 3 ; i++ ){
            for ( int j = 0 ; j < 3 ; j++ ){
                Ric(i, j) = cvRic.at<double>(i, j) ;
            }
        }

        std::sprintf(tmp, "Tic_%d", camera_id ) ;
        fs[tmp] >> cvTic ;
        for( int i = 0 ; i < 3 ; i++ ){
            Tic(i) = cvTic.at<double>(i, 0) ;
        }

        stereoPara[camera_id].setExtrinsics( Ric, Tic );

        std::sprintf(tmp, "dist_coeff_%d", camera_id ) ;
        fs[tmp] >>  dist_coeff ;

        stereoPara[camera_id].setDistortionCoff( dist_coeff.at<double>(0, 0), dist_coeff.at<double>(1, 0),
                                                 dist_coeff.at<double>(2, 0), dist_coeff.at<double>(3, 0), dist_coeff.at<double>(4, 0) );

        std::sprintf(tmp, "K_%d", camera_id ) ;
        fs[tmp] >> K ;

        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        fx /= 2.0;
        fy /= 2.0;
        cx = (cx + 0.5) / 2.0 - 0.5;
        cy = (cy + 0.5) / 2.0 - 0.5;

        stereoPara[camera_id].setParameters(fx, fy, cx, cy);
    }
    cameraParameters = stereoPara[1] ;
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

void init()
{
    for ( int i = 0 ; i < maxPyramidLevel ; i++ )
    {
        int height = IMAGE_HEIGHT >> i ;
        int width = IMAGE_WIDTH >> i ;
        grayImage[i] = Mat::zeros(height, width, CV_8U ) ;
        depthImage[i] = Mat::zeros(height, width, CV_32F ) ;
    }
    //gradientMapForDebug.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    initCalibrationParameters() ;
    //cam.start();
}

int cnt = 0 ;


void estimateCurrentState()
{

    if ( imageQueue[0].empty() || imageQueue[1].empty() ){
        return;
    }
    list<sensor_msgs::Image>::iterator iter0 = imageQueue[0].begin();
    list<sensor_msgs::Image>::iterator iter1 = imageQueue[1].begin();
    while ( iter1 != imageQueue[1].end() && iter0->header.stamp > iter1->header.stamp )
    {
        iter1 =  imageQueue[1].erase( iter1 ) ;
    }
    while ( iter0 != imageQueue[0].end() && iter0->header.stamp < iter1->header.stamp )
    {
        iter0 =  imageQueue[0].erase( iter0 ) ;
    }
    if ( imuQueue.empty() ){
        return ;
    }
    ros::Time imageTimeStamp = iter0->header.stamp ;
    list<visensor_node::visensor_imu>::reverse_iterator reverse_iterImu = imuQueue.rbegin() ;
    if ( reverse_iterImu->header.stamp < imageTimeStamp ){
        return ;
    }

    double t = (double)cvGetTickCount();

    //begin integrate imu data
    list<visensor_node::visensor_imu>::iterator iterImu = imuQueue.begin() ;

    Quaterniond q, dq ;
    q.setIdentity() ;
    while ( iterImu->header.stamp < imageTimeStamp )
    {
        double t = iterImu->header.stamp.toSec() ;
        if ( lastTime < 0 ){
            lastTime = t ;
        }
        double dt = t - lastTime ;
        lastTime = t ;
        dq.x() = iterImu->angular_velocity.x*dt*0.5 ;
        dq.y() = iterImu->angular_velocity.y*dt*0.5 ;
        dq.z() = iterImu->angular_velocity.z*dt*0.5 ;
        dq.w() =  sqrt( 1 - SQ(dq.x()) * SQ(dq.y()) * SQ(dq.z()) ) ;
        q = (q * dq).normalized();
        //cout << "[Pop out] " <<  iterImu->header << endl ;
        iterImu = imuQueue.erase( iterImu ) ;
    }
    //printf("x=%f y=%f z=%f w=%f\n", q.x(), q.y(), q.z(), q.w() ) ;

    Mat img0, img1;
    Mat disparity, depth1 ;
    bool insertKeyFrameFlag = false ;

    img1.create(iter1->height, iter1->width, CV_8UC1);
    memcpy(&img1.data[0], &iter1->data[0], iter1->height*iter1->width ) ;

    grayImage[0] = img1 ;
    for (int kk = 1; kk < maxPyramidLevel; kk++){
        pyrDownMeanSmooth<uchar>(grayImage[kk - 1], grayImage[kk]);
    }

    if ( vst == false )
    {
        vst = true;
        insertKeyFrameFlag = true ;

        img0.create(iter0->height, iter0->width, CV_8UC1);
        memcpy(&img0.data[0], &iter0->data[0], iter0->height*iter0->width ) ;

        bm_(img1, img0, disparity, CV_32F);
        int height = iter0->height ;
        int width = iter0->width ;
        depth1.create(iter0->height, iter0->width, CV_32F );
        double baseline = ( stereoPara[0].Tic - stereoPara[1].Tic ).norm() ;
        double f = stereoPara[1].fx[0] ;

        //printf("baseline=%f f=%f\n", baseline, f ) ;
        //float testSum = 0 ;
        //int cnt = 0 ;
        for ( int i = 0 ; i < height ; i++ )
        {
            for ( int j = 0 ; j < width ; j++ )
            {
                float d = disparity.at<float>(i, j) ;
                if (  d < 1.5 ) {
                    depth1.at<float>(i, j) = 0 ;
                }
                else {
                    depth1.at<float>(i, j) = baseline * f / d ;
                    //                testSum += depthImage.at<float>(i, j) ;
                    //                cnt++ ;
                }
            }
        }
        //ROS_INFO("testNum = %f\n",  testSum/cnt ) ;
        depthImage[0] = depth1 ;
        for (int kk = 1; kk < maxPyramidLevel; kk++){
            pyrDownMedianSmooth<float>(depthImage[kk - 1], depthImage[kk]);
        }

        slidingWindows.insertKeyFrame(grayImage, depthImage, gradientMapForDebug, Matrix3d::Identity(), Vector3d::Zero() );
        //slidingWindows.planeDection();

        R_k_c = Matrix3d::Identity();
        T_k_c = Vector3d::Zero();

        lastFrame = &slidingWindows.states[slidingWindows.tail];
    }
    else
    {
        //imu prior for rotation
        Matrix3d deltaR(q) ;

        Matrix3d deltaR_c = stereoPara[1].Ric.transpose() * deltaR * stereoPara[1].Ric ;

        //R_k_c = deltaR.transpose() * R_k_c ;
        R_k_c = deltaR_c.transpose() * R_k_c ;

        //puts("before dense tracking") ;

#ifdef FRAME_TO_FRAME
        frameToFrameDenseTracking(R_k_c, T_k_c);
#else
        insertKeyFrameFlag = keyframeToFrameDenseTracking( R_k_c, T_k_c );
#endif

        R_c_0 = slidingWindows.states[slidingWindows.tail].R_k0*R_k_c.transpose();
        T_c_0 = R_c_0*(
                    R_k_c*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - T_k_c);

        pubOdometry(T_c_0, R_c_0);
        pubPath(T_c_0);

        if ( insertKeyFrameFlag == true )

        {
            img0.create(iter0->height, iter0->width, CV_8UC1);
            memcpy(&img0.data[0], &iter0->data[0], iter0->height*iter0->width ) ;

            bm_(img1, img0, disparity, CV_32F);
            int height = iter0->height ;
            int width = iter0->width ;
            depth1.create(iter0->height, iter0->width, CV_32F );
            double baseline = ( stereoPara[0].Tic - stereoPara[1].Tic ).norm() ;
            double f = stereoPara[1].fx[0] ;

            //printf("baseline=%f f=%f\n", baseline, f ) ;
            //float testSum = 0 ;
            //int cnt = 0 ;
            for ( int i = 0 ; i < height ; i++ )
            {
                for ( int j = 0 ; j < width ; j++ )
                {
                    float d = disparity.at<float>(i, j) ;
                    if (  d < 1.5 ) {
                        depth1.at<float>(i, j) = 0 ;
                    }
                    else {
                        depth1.at<float>(i, j) = baseline * f / d ;
                        //                testSum += depthImage.at<float>(i, j) ;
                        //                cnt++ ;
                    }
                }
            }
            //ROS_INFO("testNum = %f\n",  testSum/cnt ) ;
            depthImage[0] = depth1 ;
            for (int kk = 1; kk < maxPyramidLevel; kk++){
                pyrDownMedianSmooth<float>(depthImage[kk - 1], depthImage[kk]);
            }

            slidingWindows.insertKeyFrame(grayImage, depthImage, gradientMapForDebug, R_c_0, T_c_0 );
            cout << "currentPosition:\n" << T_c_0.transpose() << endl;

            R_k_c = Matrix3d::Identity();
            T_k_c = Vector3d::Zero();
            lastFrame = &slidingWindows.states[slidingWindows.tail];
        }
        else
        {
#ifdef FRAME_TO_FRAME
            lastFrame = &tmpState;
            tmpState.insertFrame(grayImage, depthImage, R_c_0, T_c_0, slidingWindows.para );
#endif
        }
    }

    //printf("estimation time: %f\n", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );

    imshow("image1", img1 ) ;
    cv::moveWindow("image1", 0, 0 );
    imshow("dense key points", gradientMapForDebug ) ;
    cv::moveWindow("dense key points", 500, 0 );

    if ( insertKeyFrameFlag ){
        imshow("image0", img0 ) ;
        cv::moveWindow("image0", 500, 500 );
        imshow("diparity", disparity/maxDisparity ) ;
        cv::moveWindow("diparity", 0, 500 );
    }

    imageQueue[0].pop_front();
    imageQueue[1].pop_front();

    waitKey(1) ;
}

void imuCallBack(const visensor_node::visensor_imu& imu_msg )
{
    imuQueue.push_back( imu_msg );
    //cout << "imu" << endl << imu_msg.header << endl ;
}

void image0CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    imageQueue[0].push_back( *msg );
}

void image1CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    imageQueue[1].push_back( *msg );
}

/*
void caliCallBack(const visensor_node::visensor_calibration & msg)
{
    readyCaliNum++ ;
    cout << msg << endl ;

    int index = 0 ;
    if ( msg.cam_name == "cam1" ) {
        index = 1 ;
    }

    Eigen::Quaterniond q_IC;
    q_IC.x() = msg.T_IC.orientation.x ;
    q_IC.y() = msg.T_IC.orientation.y ;
    q_IC.z() = msg.T_IC.orientation.z ;
    q_IC.w() = msg.T_IC.orientation.w ;

    Eigen::Matrix3d Ric(q_IC) ;
    Eigen::Vector3d Tic ;
    Tic(0) = msg.T_IC.position.x ;
    Tic(1) = msg.T_IC.position.y ;
    Tic(2) = msg.T_IC.position.z ;

    double fx = msg.focal_length[0] ;
    double fy = msg.focal_length[1] ;
    double cx = msg.principal_point[0] ;
    double cy = msg.principal_point[1] ;

    fx /= 2.0;
    fy /= 2.0;
    cx = (cx + 0.5) / 2.0 - 0.5;
    cy = (cy + 0.5) / 2.0 - 0.5;

    stereoPara[index].setExtrinsics(Ric, Tic);
    stereoPara[index].setParameters( fx, fy, cx, cy );
    stereoPara[index].setDistortionCoff( msg.dist_coeff[0], msg.dist_coeff[1], msg.dist_coeff[2], msg.dist_coeff[3], msg.dist_coeff[4] );

    if ( index == 1 ){
        cameraParameters = stereoPara[1] ;
    }
}
*/

int main(int argc, char** argv )
{
    init() ;
    ros::init(argc, argv, "denseVO") ;
    ros::NodeHandle n ;
    //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);

    sub_image[0] = n.subscribe("/cam0", 20, &image0CallBack );
    sub_image[1] = n.subscribe("/cam1", 20, &image1CallBack );
    sub_imu = n.subscribe("/cust_imu0", 1000, &imuCallBack ) ;
    //sub_cali = n.subscribe("/calibration", 10, &caliCallBack ) ;
    //sub_depth = it.subscribe("/camera/depthImage", 20, &depthImageCallBack);

    //sub_image = it.subscribe("/camera/rgb/image_color", 100, &rgbImageCallBack);
    //sub_depth = it.subscribe("/camera/depth/image", 100, &depthImageCallBack);

    //pub_grayImage = n.advertise<sensor_msgs::Image>("camera/grayImage", 10 ) ;
    pub_path = n.advertise<visualization_msgs::Marker>("/denseVO/path", 1000);;
    pub_odometry = n.advertise<nav_msgs::Odometry>("/denseVO/odometry", 1000);;
    pub_pose = n.advertise<geometry_msgs::PoseStamped>("/denseVO/pose", 1000);
    pub_cloud = n.advertise<sensor_msgs::PointCloud>("/denseVO/cloud", 1000);

    //pub_grayImage = it.advertise("camera/grayImage", 1 );200
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

    imuQueue.clear();
    imageQueue[0].clear();
    imageQueue[1].clear();
    //    ros::spinOnce() ;
    //    imuQueue.clear();
    //    imageQueue[0].clear();
    //    imageQueue[1].clear();
    ros::Rate loop_rate(500.0);
    while( ros::ok() )
    {
        loop_rate.sleep() ;
        ros::spinOnce() ;
        //TODO
        estimateCurrentState() ;
    }

    return 0;
}
