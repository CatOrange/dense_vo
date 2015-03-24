#ifndef __VARIABLEDEFINIATION_H
#define __VARIABLEDEFINIATION_H
#include <cmath>

#define DOWNSAMPLING

//#define SAVEVIDEO

#ifdef DOWNSAMPLING
const int IMAGE_WIDTH = 320;
const int IMAGE_HEIGHT = 240;
#else
const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 480;
#endif


//maths
const double PI = acos(-1.0);
const double PI_2 = PI / 2;
const double PI2 = PI*2;
const double SQR2 = sqrt(2.0);
const double zeroThreshold = 0.0000001;

//data structure
const int queueSize = IMAGE_HEIGHT*IMAGE_WIDTH + 10;
const int unisetSize = IMAGE_HEIGHT*IMAGE_WIDTH + 10;

//for integral normal estimation
const double Gamma = 2;

//for plane extraction - simple clustering
const double cosThreshold = cos(10.0 / 180 * PI);

//for insert Superpixel
//#define DEBUG_INSERT_SUPERPIXEL

//for BA
//#define DEBUG_BA
const int slidingWindowSize = 10;
const double AverageIntensityError = 80.0;
const double invIntensityCov = 1.0 / (3.0*3.0);
const double dxOptProportion = 0.1;
const int maxIterationBA = 10;
const int maxPyramidLevelBA = 1;
const double normalizedParallaxThreshold = 10.0 / 535.0;
const double angularThreshold = 3.0 / 180 * PI;

//const double normalizedParallaxThreshold = 1.0;

//for kMeansClustering
#define REGULARITY 
const int maxKMeansNum = 500;

//for huber kernel
#define WEIGHTEDCOST
const double huberKernelThreshold = 16.0;
const double validPixelThreshold = 0.5;
const double validPixelPercentageThreshold = 0.5;

//for dense tracking
//#define DEBUG_DENSETRACKING
//#define FRAME_TO_FRAME
//#define ADD_VELOCITY_PRIOR
const double graidientThreshold = 10.0;
const double minimumUpdateTranslationThreshold = 0.005;
const double minimumUpdateAngularThreshold = 0.5/180.0*PI ;
const int maxIteration = 5 ;
const int maxPyramidLevel =  3;
const int minDenseTrackingNum = 4800;
const double covRotationVel = 1.0 / 180.0 * PI ;
const double weightRotationVel = 1.0 / covRotationVel*10 ;
const double covLinearVel = 0.01 ;
const double weightLinearVel = 1.0 / covLinearVel*10 ;

//For reprojection
const int expectedClusteringNum = 200;
const int minOptNum = 100 * (1 << (maxPyramidLevelBA - 1) * 2);
//const int minOptNum = T_NUM;

/*
for plane extraction - AHC
*/
//#define DEBUG_CLUSTRING
const int winH = 4;
const int winW = 4;
const double depthFactor = 5000.0;
const double alpha = 0.015;
const double sigma = 1.6;
const double eps = 5;
const int T_NUM = 20000 ;
const int minimumPixelsInASuperpixel = minOptNum;
const int DISPLAY_NUM = 50;

/*
for openmp
*/
const int ompNumThreads = 2;
#endif
