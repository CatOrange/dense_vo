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
//for VO
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
using namespace std;
using namespace cv;
using namespace Eigen;
const int numImage = 800;
const int groundTruthDataNum = 5000;
char filePath[256] = "D:\\Dataset\\rgbd_dataset_freiburg3_structure_texture_near\\" ;
char depthDataPath[256] ;
char rgbDataPath[256] ;
char rgbListPath[256] ;
char depthListPath[256] ;
char groundTruthDataPath[256];
char depthFileNameList[numImage][128];
char rgbFileNameList[numImage][128];
unsigned long long rgbImageTimeStamp[numImage];
unsigned long long depthImageTimeStamp[numImage];
CAMER_PARAMETERS cameraParameters(535.4, 539.2, 320.1, 247.6);//TUM Freiburg 3 sequences
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6,	255.3,	0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6, 255.3);//TUM Freiburg 1 sequences
STATEESTIMATION slidingWindows(IMAGE_HEIGHT, IMAGE_WIDTH, &cameraParameters);
Matrix3d firstFrameRtoVICON;
Vector3d firstFrameTtoVICON;

struct VICONDATA{
    unsigned long long timeStamp;
    float tx, ty, tz;
    float qx, qy, qz, qw;
    bool operator < (const VICONDATA& a)const{
        return timeStamp < a.timeStamp;
    }
};
VICONDATA groundTruth[groundTruthDataNum];
map<unsigned long long, int> groundTruthMap;

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

void InitFIleList()
{
    char tmp[256];
    FILE *fp;

    strcpy(depthDataPath, filePath);
    strcat(depthDataPath, "depth/");

    strcpy(rgbDataPath, filePath);
    strcat(rgbDataPath, "rgb/");

    strcpy(rgbListPath, filePath);
    strcat(rgbListPath, "rgb.txt");

    strcpy(depthListPath, filePath);
    strcat(depthListPath, "depth.txt");

    strcpy(groundTruthDataPath, filePath);
    strcat(groundTruthDataPath, "groundtruth.txt");

    //read rgb image name list
    fp = fopen(rgbListPath, "r");
    cout << rgbListPath << endl ;
    if (fp == NULL){
        puts("rgbList Path error");
        return ;
    }
    while (fgets(tmp, 256, fp) != NULL){
        if (tmp[0] != '#') break;
    }
    for (int i = 0, j; i < numImage; i++)
    {
        if (fgets(tmp, 256, fp) == NULL) break;

        char tt[128];
        int n, ns;
        sscanf(tmp, "%d.%d %s", &n, &ns, tt);
        rgbImageTimeStamp[i] = n;
        rgbImageTimeStamp[i] *= 1000000;
        rgbImageTimeStamp[i] += ns;
        //if ( i < 10 )
        //printf("%d %lld\n", i, rgbImageTimeStamp[i]);

        for (j = 0; tmp[j] != '\0'; j++){
            if (tmp[j] == '/') break;
        }
        strcpy(rgbFileNameList[i], &tmp[j + 1]);
        rgbFileNameList[i][strlen(rgbFileNameList[i]) - 1] = '\0';
        //printf("%s\n", rgbFileNameList[i] );
    }
    fclose(fp);

    //read depth image name list
    fp = fopen(depthListPath, "r");
    if (fp == NULL){
        puts("depthList Path error");
    }
    while (fgets(tmp, 256, fp) != NULL){
        if (tmp[0] != '#') break;
    }
    for (int i = 0, j; i < numImage; i++)
    {
        if (fgets(tmp, 256, fp) == NULL) break;

        char tt[128];
        int n, ns;
        sscanf(tmp, "%d.%d %s", &n, &ns, tt);
        depthImageTimeStamp[i] = n;
        depthImageTimeStamp[i] *= 1000000;
        depthImageTimeStamp[i] += ns;

        for (j = 0; tmp[j] != '\0'; j++){
            if (tmp[j] == '/') break;
        }
        strcpy(depthFileNameList[i], &tmp[j + 1]);
        depthFileNameList[i][strlen(depthFileNameList[i]) - 1] = '\0';
        //printf("%s\n", depthFileNameList[i]);
    }
    fclose(fp);

    //read the ground truth data
    groundTruthMap.clear();
    fp = fopen(groundTruthDataPath, "r");
    if (fp == NULL){
        puts("groundTruthData Path error");
    }
    while (fgets(tmp, 256, fp) != NULL){
        if (tmp[0] != '#') break;
    }
    for (int i = 0, j; i < groundTruthDataNum; i++)
    {
        if (fgets(tmp, 256, fp) == NULL) break;
        int len = strlen(tmp);
        for (j = 0; tmp[j] != '\0'; j++){
            if (tmp[j] == '/') break;
        }
        //tx ty tz qx qy qz qw
        int s, ns;
        sscanf(tmp, "%d.%d %f %f %f %f %f %f %f", &s, &ns, &groundTruth[i].tx, &groundTruth[i].ty, &groundTruth[i].tz,
               &groundTruth[i].qx, &groundTruth[i].qy, &groundTruth[i].qz, &groundTruth[i].qw);
        groundTruth[i].timeStamp = s;
        groundTruth[i].timeStamp *= 1000000;
        groundTruth[i].timeStamp += ns*100;
        //printf("%s\n", depthFileNameList[i]);
        groundTruthMap.insert( pair<unsigned long long, int>(groundTruth[i].timeStamp, i) );
    }
    fclose(fp);
}

inline unsigned long long absUnsignedLongLong(unsigned long long a, unsigned long long b){
    if (a > b) return a - b;
    else return b - a;
}


void init()
{
    InitFIleList();
}

void updateR_T(Vector3d& w, Vector3d& v)
{
    Matrix3d skewW(3, 3);
    skewW(0, 0) = skewW(1, 1) = skewW(2, 2) = 0;
    skewW(0, 1) = -w(2);
    skewW(1, 0) = w(2);
    skewW(0, 2) = w(1);
    skewW(2, 0) = -w(1);
    skewW(1, 2) = -w(0);
    skewW(2, 1) = w(0);

    double theta = sqrt(w.squaredNorm());

    Matrix3d deltaR = Matrix3d::Identity() + (sin(theta) / theta)*skewW + ((1 - cos(theta)) / (theta*theta))*skewW*skewW;
    Vector3d deltaT = (Matrix3d::Identity() + ((1 - cos(theta)) / (theta*theta)) *skewW + ((theta - sin(theta)) / (theta*theta*theta)*skewW*skewW)) * v;

    Matrix3d R1 = deltaR.transpose();
    Vector3d T1 = -deltaR.transpose()*deltaT;

    w = -w;
    v = -v;

    skewW(0, 0) = skewW(1, 1) = skewW(2, 2) = 0;
    skewW(0, 1) = -w(2);
    skewW(1, 0) = w(2);
    skewW(0, 2) = w(1);
    skewW(2, 0) = -w(1);
    skewW(1, 2) = -w(0);
    skewW(2, 1) = w(0);

    Matrix3d R2 = Matrix3d::Identity() + (sin(theta) / theta)*skewW + ((1 - cos(theta)) / (theta*theta))*skewW*skewW;
    Vector3d T2 = (Matrix3d::Identity() + ((1 - cos(theta)) / (theta*theta)) *skewW + ((theta - sin(theta)) / (theta*theta*theta)*skewW*skewW)) * v;


    cout << R1 << endl;
    cout << R2 << endl;
    cout << T1 << endl;
    cout << T2 << endl;

    //Matrix3d newR = R*deltaR.transpose();
    //Vector3d newT = -R*deltaR.transpose()*deltaT + T;

    //R = newR;
    //T = newT;
}

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
    pubPoses = n.advertise<geometry_msgs::PoseArray>("poses",           10, true);
    marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1 ) ;

    ros::Rate loop_rate(0.1) ;

    bool vst = false;
    map<unsigned long long, int>::iterator iter;
    Matrix3d nextR;//R_k^(k+1)
    Matrix3d currentR;
    Vector3d nextT;//T_k^(k+1)
    Vector3d currentT;
    Mat rgbImage, depthImage;
    Mat grayImage[maxPyramidLevel];
    //ofstream fileOutput("result.txt");

    init();
    //for ( int i = 1; ros::ok() && i < 750; i += 1 )
    for ( int i = 1; i < 750; i += 1 )
    {
        printf("id : %d\n", i);
        char tmp[256];

        //read rgb image
        strcpy(tmp, rgbDataPath);
        strcat(tmp, rgbFileNameList[i]);
        rgbImage = imread(tmp, CV_LOAD_IMAGE_COLOR);
        cvtColor(rgbImage, grayImage[0], CV_BGR2GRAY);

#ifdef DOWNSAMPLING
        pyrDown(grayImage[0], grayImage[0]);//down-sampling
#endif

        for (int kk = 1; kk < maxPyramidLevel; kk++){
            pyrDown(grayImage[kk-1], grayImage[kk]);//down-sampling
        }

        //read depth image
        strcpy(tmp, depthDataPath);
        int k = i - 1;
        if (k < 0){
            k++;
        }
        unsigned long long minS = absUnsignedLongLong(depthImageTimeStamp[k], rgbImageTimeStamp[i]);
        if (absUnsignedLongLong(depthImageTimeStamp[i], rgbImageTimeStamp[i]) < minS) {
            k = i;
            minS = absUnsignedLongLong(depthImageTimeStamp[i], rgbImageTimeStamp[i]);
        }
        if (i + 1 < numImage && absUnsignedLongLong(depthImageTimeStamp[i+1], rgbImageTimeStamp[i]) < minS ){
            k = i + 1;
        }
        strcat(tmp, depthFileNameList[k]);
        depthImage = imread(tmp, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

        depthImage.convertTo(depthImage, CV_32F  );
        depthImage /= depthFactor;

#ifdef DOWNSAMPLING
        pyrDown(depthImage, depthImage ) ;
#endif


        //find groundTruth time stamp
        int timeID = 0;
        minS = 10000000;

        iter = groundTruthMap.lower_bound(rgbImageTimeStamp[i]);
        if (iter != groundTruthMap.end())
        {
            if (absUnsignedLongLong(iter->first, rgbImageTimeStamp[i]) < minS){
                timeID = iter->second;
                minS = absUnsignedLongLong(iter->first, rgbImageTimeStamp[i]);
            }
        }
        if (iter != groundTruthMap.begin())
        {
            iter--;
            if (absUnsignedLongLong(iter->first, rgbImageTimeStamp[i]) < minS){
                timeID = iter->second;
            }
        }
        if (minS == 10000000){
            break;
        }

        if (vst == false )//the first frame
        {
            vst = true;
            Quaterniond q;
            q.x() = groundTruth[timeID].qx;
            q.y() = groundTruth[timeID].qy;
            q.z() = groundTruth[timeID].qz;
            q.w() = groundTruth[timeID].qw;
            firstFrameRtoVICON = q.toRotationMatrix();
            //firstFrameRtoVICON.transposeInPlace();
            firstFrameTtoVICON << groundTruth[timeID].tx, groundTruth[timeID].ty, groundTruth[timeID].tz;
            cout << firstFrameTtoVICON << endl;
            slidingWindows.insertKeyFrame(grayImage, depthImage, Matrix3d::Identity(), Vector3d::Zero() );
            currentR = Matrix3d::Identity();
            currentT = Vector3d::Zero();

            nextR = Matrix3d::Identity();
            nextT = Vector3d::Zero();
            continue;
        }

        double t = (double)cvGetTickCount();
        slidingWindows.denseTrackingWithoutSuperpixel(grayImage, nextR, nextT);
        //cout << nextR << endl;
        //cout << nextT << endl;
        t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
        printf("cal time: %f\n", t);

        //update current  calculation
        currentR = slidingWindows.states[slidingWindows.tail].R_k0*nextR.transpose();
        currentT = currentR*(
                    nextR*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - nextT);
        //fileOutput << (firstFrameRtoVICON*currentT).transpose() << endl;

        //insert key frame
        if ((i % 10) == 1)
        {
            slidingWindows.insertKeyFrame(grayImage, depthImage, currentR, currentT );
            slidingWindows.PhotometricBA();

            nextR = Matrix3d::Identity();
            nextT = Vector3d::Zero();

            cout << "estimate position:\n" << firstFrameRtoVICON.transpose()*slidingWindows.states[slidingWindows.tail].T_k0 + firstFrameTtoVICON << endl;
            cout << "ground truth position:\n" << groundTruth[timeID].tx << endl << groundTruth[timeID].ty << endl << groundTruth[timeID].tz << endl;
        }

        ros::Time ros_t = ros::Time::now() ;

        vector<Vector3d> pointCloud ;
        vector<unsigned short>R, G, B ;
        vector<Vector3d> ps ;
        vector<Matrix3d> Rs ;
        slidingWindows.prepareDateForVisualization(pointCloud, R, G, B, ps, Rs );
        publish_all(pointCloud, R, G, B, ps, Rs, ros_t, currentR, currentT ) ;
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", grayImage[0]).toImageMsg() ;
        pubImage.publish( msg ) ;
        //ros::spinOnce() ;
        //loop_rate.sleep() ;
        //imshow("image",  grayImage[0]) ;
        //waitKey(30) ;
    }
    //	fileOutput.close();

    return 0;
}
