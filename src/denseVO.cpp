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

PrimeSenseCam cam;

geometry_msgs::PoseStamped poseROS;
nav_msgs::Path             pathROS;

ros::Publisher pubVOdom ;
ros::Publisher pubCloud ;
ros::Publisher pubPoses ;
ros::Publisher marker_pub ;

Mat rgbImage, depthImage;
Mat grayImage[maxPyramidLevel];
STATE tmpState;
STATE* lastFrame;

/*
// Current VINS odometry
void publish_current(const Vector3d& p,
                     const Vector3d& v,
                     const Matrix3d& R,
                     const ros::Time t )
{
  nav_msgs::Odometry odom;
  Quaterniond q = Quaterniond(R);
  odom.header.stamp    = t;
  odom.header.frame_id = string("/map");
  odom.pose.pose.position.x = p(0);
  odom.pose.pose.position.y = p(1);
  odom.pose.pose.position.z = p(2);
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.twist.twist.linear.x = v(0);
  odom.twist.twist.linear.y = v(1);
  odom.twist.twist.linear.z = v(2);
//  odom.twist.twist.angular.x = imuImage.w(0);
//  odom.twist.twist.angular.y = imuImage.w(1);
//  odom.twist.twist.angular.z = imuImage.w(2);
//  odom.twist.covariance[0]  = stats[0];
//  odom.twist.covariance[1]  = stats[1];
//  odom.twist.covariance[2]  = stats[2];
//  odom.twist.covariance[3]  = stats[3];
//  odom.twist.covariance[33] = imuImage.a(0);
//  odom.twist.covariance[34] = imuImage.a(1);
//  odom.twist.covariance[35] = imuImage.a(2);
  pubVOdom.publish(odom);
}
*/

// Current pointcloud and all poses in the window
void publish_all(const vector<Vector3d>& pointcloud,
                 const vector<unsigned short>& R,
                 const vector<unsigned short>& G,
                 const vector<unsigned short>& B,
                 const vector<Vector3d>& ps,
                 const vector<Matrix3d>& Rs,
                 const ros::Time         t,
                 const Matrix3d& currentR,
                 const Vector3d& currentT )
{
    sensor_msgs::PointCloud cloud;
    cloud.header.stamp    = t;
    cloud.header.frame_id = string("/map");
    cloud.points.resize(pointcloud.size());
    cloud.channels.resize(1) ;
    cloud.channels[0].name = "rgb" ;
    cloud.channels[0].values.resize( pointcloud.size() ) ;
    for (unsigned int k = 0; k < pointcloud.size(); k++)
    {
        cloud.points[k].x = pointcloud[k](0);
        cloud.points[k].y = pointcloud[k](1);
        cloud.points[k].z = pointcloud[k](2);
        int rgb = (R[k]<<16) | (G[k]<<8) | B[k] ;
        cloud.channels[0].values[k] = *reinterpret_cast<float*>(&rgb) ;
    }

    geometry_msgs::PoseArray poses;
    poses.header.stamp    = t;
    poses.header.frame_id = string("/map");
    poses.poses.resize(ps.size()+1);
    unsigned int k ;
    for ( k = 0; k < ps.size(); k++)
    {
        poses.poses[k].position.x = ps[k](0);
        poses.poses[k].position.y = ps[k](1);
        poses.poses[k].position.z = ps[k](2);
        Quaterniond q = Quaterniond(Rs[k]);
        poses.poses[k].orientation.w = q.w();
        poses.poses[k].orientation.x = q.x();
        poses.poses[k].orientation.y = q.y();
        poses.poses[k].orientation.z = q.z();
    }
    poses.poses[k].position.x = currentT(0);
    poses.poses[k].position.y = currentT(1);
    poses.poses[k].position.z = currentT(2);
    Quaterniond q = Quaterniond(currentR.transpose());
    poses.poses[k].orientation.w = q.w();
    poses.poses[k].orientation.x = q.x();
    poses.poses[k].orientation.y = q.y();
    poses.poses[k].orientation.z = q.z();

    printf("[pub] pointCloud Size:%d Pose Size:%d\n", pointcloud.size(), k+1 ) ;
    for ( int i = 0 ; i <= k ; i++ ){
        printf("Pose: %d x:%f y:%f z:%f\n", i, poses.poses[i].position.x, poses.poses[i].position.y, poses.poses[i].position.z ) ;
    }

    pubCloud.publish(cloud);
    pubPoses.publish(poses);
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

int main(int argc, char** argv )
{
    ros::init(argc, argv, "densevo") ;
    ros::NodeHandle n ;
    image_transport::ImageTransport it(n) ;
    image_transport::Publisher pubImage = it.advertise("camera/image", 1 ) ;

    pubVOdom = n.advertise<nav_msgs::Odometry>(      "vodom",           10);
    pubCloud = n.advertise<sensor_msgs::PointCloud>( "cloud",           10, true);
    pubPoses = n.advertise<geometry_msgs::PoseArray>( "poses",           10, true);
    marker_pub = n.advertise<visualization_msgs::Marker>( "visualization_marker", 1 ) ;

    ros::Rate loop_rate(0.1) ;
    bool vst = false;
    Matrix3d R_k_c;//R_k^(k+1)
    Matrix3d R_c_0;
    Vector3d T_k_c;//T_k^(k+1)
    Vector3d T_c_0;
    Mat rgbImage, depthImage;
    //ofstream fileOutput("result.txt");

    cam.start();
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

        depthImage.convertTo(depthImage, CV_32F );

#ifdef DOWNSAMPLING
        pyrDown(depthImage, depthImage ) ;
#endif

        if (vst == false )//the first frame
        {
            vst = true;
//            Quaterniond q;
//            q.x() = groundTruth[timeID].qx;
//            q.y() = groundTruth[timeID].qy;
//            q.z() = groundTruth[timeID].qz;
//            q.w() = groundTruth[timeID].qw;
//            firstFrameRtoVICON = q.toRotationMatrix();
//            firstFrameTtoVICON << groundTruth[timeID].tx, groundTruth[timeID].ty, groundTruth[timeID].tz;

            //cout << firstFrameTtoVICON << endl;

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

            cout << "estimate position[after BA]:\n"
                << slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;

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

    return 0;
}
