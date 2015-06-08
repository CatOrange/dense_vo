#ifndef __VARIABLEDEFINIATION_H
#define __VARIABLEDEFINIATION_H
#include <cmath>
#include "Eigen/Dense"

#define DOWNSAMPLING

//#define SAVEVIDEO

#ifdef DOWNSAMPLING
const int IMAGE_WIDTH = 320;
//const int IMAGE_WIDTH = 376;
const int IMAGE_HEIGHT = 240;
#else
const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 480;
#endif

const int variablesNumInState = 9;

//maths
const float PI = acos(-1.0);
const float PI_2 = PI / 2;
const float PI2 = PI*2;
const float SQR2 = sqrt(2.0);
const float zeroThreshold = 0.0000001;

//data structure
const int queueSize = IMAGE_HEIGHT*IMAGE_WIDTH + 10;
const int unisetSize = IMAGE_HEIGHT*IMAGE_WIDTH + 10;

//for integral normal estimation
const float Gamma = 2;

//for plane extraction - simple clustering
const float cosThreshold = cos(10.0 / 180 * PI);

//for insert Superpixel
//#define DEBUG_INSERT_SUPERPIXEL


//for BA
//#define DEBUG_BA
const Eigen::Matrix3d acc_cov = 1e-2 * Eigen::Matrix3d::Identity();
const Eigen::Matrix3d gra_cov = 1e-6 * Eigen::Matrix3d::Identity();
const Eigen::Matrix3d gyr_cov = 1e-4 * Eigen::Matrix3d::Identity();
const int slidingWindowSize = 30 ;
const float AverageIntensityError = 80.0;
const float invIntensityCov = 1.0 / (3.0*3.0);
const float dxOptProportion = 0.1;
const int maxIterationBA = 4;
const int maxPyramidLevelBA = 1;
const float normalizedParallaxThreshold = 10.0/268 ;
const float angularThreshold = 3.0 / 180 * PI;
const float huber_r_v = 0.05 ;
const float huber_r_w = 1.0/180.0*PI ;

//const float normalizedParallaxThreshold = 1.0;

//for kMeansClustering
#define REGULARITY
const int maxKMeansNum = 500;

//for huber kernel
#define WEIGHTEDCOST
const float huberKernelThreshold = 20.0;
const float validPixelThreshold = 0.5;
const float validPixelPercentageThreshold = 0.5;

//for dense tracking
//#define DEBUG_DENSETRACKING
//#define FRAME_TO_FRAME
//#define ADD_VELOCITY_PRIOR
const int keyFrameInterval = 10 ;
const float graidientThreshold = 5.0;
const float minimumUpdateTranslationThreshold = 0.005;
const float minimumUpdateAngularThreshold = 0.3/180.0*PI ;
const Eigen::Matrix3d v_cov_inv = 100*1.0/(minimumUpdateTranslationThreshold*minimumUpdateTranslationThreshold) * Eigen::Matrix3d::Identity();
const Eigen::Matrix3d w_cov_inv = 100*1.0/(minimumUpdateAngularThreshold*minimumUpdateAngularThreshold) * Eigen::Matrix3d::Identity();
const int maxIteration[5] = {5, 20, 50, 100, 100} ;
const int maxPyramidLevel = 5 ;
const int beginPyramidLevel = 0 ;
const int minDenseTrackingNum = 16000 ;
const float covRotationVel = 1.0 / 180.0 * PI ;
const float weightRotationVel = 1.0 / covRotationVel*10 ;
const float covLinearVel = 0.01 ;

const float weightLinearVel = 1.0 / covLinearVel*10 ;

//For reprojection
const int expectedClusteringNum = 80;
const int minOptNum = 800 * (1 << (maxPyramidLevelBA - 1) * 2);
//const int minOptNum = T_NUM;

/*
for plane extraction - AHC
*/
//#define DEBUG_CLUSTRING
const int winH = 2;
const int winW = 2;
const float depthFactor = 5000.0;
const float alpha = 0.015;
const float sigma = 1.6;
const float eps = 5;
const int T_NUM = 5000 ;
const int minimumPixelsInASuperpixel = 100;
const int DISPLAY_NUM = 50;

/*
for openmp
*/
const int ompNumThreads = 4;

//#define DEBUG_INFO

/*
for ros
*/
const int frameInfoListSize = 200 ;
const int bufferSize = 100 ;

const int maxDisparity[3] = {64, 32, 16} ;
const int SADwindowSize[3] = {15, 11, 5} ;

const float denseTrackingDiv = 1000.0 ;
const float trustResidualT = 8.0 ;

const short stereoThreshold = 10 ;

#define MODEL_ORDER 3



#endif
