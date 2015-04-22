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
#include "tic_toc.h"
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
ros::Publisher pub_debugGradientMap ;
ros::Publisher pub_disparityMap ;
ros::Subscriber sub_image[2];
ros::Subscriber sub_imu;
ros::Subscriber sub_cali ;
visualization_msgs::Marker path_line;

list<visensor_node::visensor_imu> imuQueue ;
list<sensor_msgs::Image> imageQueue[2] ;
//list<ros::Time> imageTimeQueue[2] ;


void imuCallBack(const visensor_node::visensor_imu& imu_msg ){
    imuQueue.push_back( imu_msg );
}

vector<double>imgTimeSeq;

void image0CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    Mat img ;
    img.create( msg->height, msg->width, CV_8UC1 );
    memcpy(&img.data[0], &msg->data[0], msg->height*msg->width ) ;

    double t = msg->header.stamp.toSec() ;
    char c[128] ;

    std::sprintf(c,"/home/ygling2008/dataSet/img0/%.6f_0.png\n", t ) ;
    imwrite(c, img) ;
    imgTimeSeq.push_back(t);

    imshow("img0", img ) ;
}

void image1CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    Mat img ;
    img.create( msg->height, msg->width, CV_8UC1 );
    memcpy(&img.data[0], &msg->data[0], msg->height*msg->width ) ;

    double t = msg->header.stamp.toSec() ;
    char c[128] ;

    std::sprintf(c,"/home/ygling2008/dataSet/img1/%.6f_1.png\n", t ) ;
    imwrite(c, img) ;

    imshow("img1", img ) ;
}

int main(int argc, char** argv )
{

    ros::init(argc, argv, "denseVO") ;
    ros::NodeHandle n ;
    //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);

    sub_image[0] = n.subscribe("/cam0", 20, &image0CallBack );
    sub_image[1] = n.subscribe("/cam1", 20, &image1CallBack );
    sub_imu = n.subscribe("/cust_imu0", 1000, &imuCallBack ) ;

    imuQueue.clear();
    imgTimeSeq.clear();
    imageQueue[0].clear();
    imageQueue[1].clear();

    ros::Rate loop_rate(500.0);
    while( ros::ok() )
    {
        loop_rate.sleep() ;
        ros::spinOnce() ;
        char key = cv::waitKey(1) ;
        if ( key == 'e' ){
            break ;
        }
    }

    FILE *fp = NULL;

    fp = fopen("/home/ygling2008/dataSet/camera.txt", "w");
    int img_sz = imgTimeSeq.size() ;
    printf("img_sz = %d\n", img_sz ) ;
    for ( int i = 0 ; i < img_sz; i++ ){
         fprintf( fp, "%.6f\n", imgTimeSeq[i] ) ;
    }
    fclose(fp) ;

    fp = fopen("/home/ygling2008/dataSet/imu.txt", "w");
    int sz = imuQueue.size() ;
    list<visensor_node::visensor_imu>::iterator iterImu = imuQueue.begin() ;
    for ( ; iterImu != imuQueue.end(); iterImu++ )
    {
         double t = iterImu->header.stamp.toSec() ;
         fprintf( fp, "%.6f %f %f %f %f %f %f\n", t
                  , iterImu->linear_acceleration.x
                  , iterImu->linear_acceleration.y
                  , iterImu->linear_acceleration.z
                  , iterImu->angular_velocity.x
                  , iterImu->angular_velocity.y
                  , iterImu->angular_velocity.z ) ;
    }
    fclose(fp) ;

    return 0;
}
