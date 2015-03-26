#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <omp.h>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "kMeansClustering.h"
//#include "planeExtraction.h"
#include "stateEstimation.h"
#include "variableDefinition.h"
//#include "testDataGeneration.h"
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
char filePath[256] = "/home/nova/dataSet/mydataSet/" ;

CAMER_PARAMETERS cameraParameters(533.750640, 533.578597, 315.564012, 246.066623);//my rgbd sensor
//CAMER_PARAMETERS cameraParameters(535.4, 539.2, 320.1, 247.6);//TUM Freiburg 3 sequences
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6,	255.3,	0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6, 255.3);//TUM Freiburg 1 sequences
STATEESTIMATION slidingWindows(IMAGE_HEIGHT, IMAGE_WIDTH, &cameraParameters);

//testDataGenerator TDG;
Mat rgbImage;
Mat depthImage[maxPyramidLevel];
Mat grayImage[maxPyramidLevel];
STATE tmpState;
STATE* lastFrame;

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

	Matrix3d tmpR = R_k_c;
	Vector3d tmpT = T_k_c;

	slidingWindows.denseTrackingWithoutSuperpixel(keyframe, grayImage, R_k_c, T_k_c );

	slidingWindows.last_delta_R = R_k_c * tmpR.transpose();
	slidingWindows.last_delta_T = T_k_c - slidingWindows.last_delta_R* tmpT;

	//cout << slidingWindows.last_delta_R << endl;
	//cout << slidingWindows.last_delta_T.transpose() << endl;
}

void RtoEulerAngles(Matrix3d R, double a[3])
{
	double theta = acos(0.5*(R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
	a[0] = (R(2, 1) - R(1, 2)) / (2.0* sin(theta));
	a[1] = (R(0, 2) - R(2, 0)) / (2.0* sin(theta));
	a[2] = (R(1, 0) - R(0, 1)) / (2.0* sin(theta));
}

int tmpGradientThreshold = 0;

void on_trackbar(int, void*)
{
	STATE* current = &slidingWindows.states[slidingWindows.tail];

	for (int level = 0; level >= 0; level--)
	{
		float* pDepth = current->depthImage[level];
		int n = IMAGE_HEIGHT >> level;
		int m = IMAGE_WIDTH >> level;
		unsigned char* pIntensity = current->intensity[level];
		unsigned char *nextIntensity = (unsigned char*)grayImage[level].data;
		double* pGradientX = current->gradientX[level];
		double* pGradientY = current->gradientY[level];
		double proportion = 0.3;

		Mat now(n, m, CV_8UC3);
		Mat next;
		cv::cvtColor(grayImage[level], next, CV_GRAY2BGR);
		int num = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				if (SQ(pGradientX[INDEX(i, j, n, m)]) + SQ(pGradientY[INDEX(i, j, n, m)]) < tmpGradientThreshold )
				{
					now.at<cv::Vec3b>(i, j)[0] = 0;
					now.at<cv::Vec3b>(i, j)[1] = 255;
					now.at<cv::Vec3b>(i, j)[2] = 0;
				}
				else
				{
					num++;
					now.at<cv::Vec3b>(i, j)[0] = pIntensity[INDEX(i, j, n, m)];
					now.at<cv::Vec3b>(i, j)[1] = pIntensity[INDEX(i, j, n, m)];
					now.at<cv::Vec3b>(i, j)[2] = pIntensity[INDEX(i, j, n, m)];
				}
			}
		}
		imshow("effect", now);
		printf("num=%d\n", num);
	}
}

int main()
{
	//init();
	//TDG.generate();
	//TDG.testProgram(&slidingWindows);
	//return 0 ;

	//Eigen::setNbThreads(4);
	//vector<int> a(5);
	//for (int i = 0; i < 5; i++){
	//	a[i] = i * 2 + 1;
	//}
	//vector<int>::iterator iter2;
	//int idx = lower_bound(a.begin(), a.begin()+5, 20 ) - a.begin() ;
	//cout << idx << endl;
	//return 0;
	//testEigen();
	//return 0;

	//SparseMatrix<double> H(3, 3);

	//H.coeffRef(1, 2) += 1;
	//cout << H << endl;
	//return 0;

	//Vector3d v = Vector3d::Random();
	//Vector3d w = Vector3d::Random();

	//updateR_T(w, v);
	//return 0;

	//vector<int>a(10);
	//for (int i= 0; i < 10; i++){
	//	a[i] = 10 - i;
	//}
	//sort(a.begin(), a.begin()+5);
	//for (int i = 0; i < 10; i++){
	//	printf("%d ", a[i]);
	//}
	//puts("");

	//vector<int>a;
	//a.reserve(10);
	//for (int i = 0; i < 10; i++)
	//{
	//	a.push_back(i);
	//	//a[i] = i;
	//	cout << a[i] << " " << a.size() << " ";
	//}
	//return 0;

	bool vst = false;

	Matrix3d R_k_c;//R_k^(k+1)
	Matrix3d R_c_0;
	Vector3d T_k_c;//T_k^(k+1)
	Vector3d T_c_0;

  char filePath[256] ;

  for ( int i = 0; i < numImage; i += 1 )
	{
		printf("id : %d\n", i);
    sprintf( filePath, "/home/nova/dataSet/mydataSet/image_%05d.png", i );

    grayImage[0] = imread( filePath, CV_LOAD_IMAGE_GRAYSCALE ) ;

    sprintf( filePath, "/home/nova/dataSet/mydataSet/depth_%05d.png", i );
    depthImage[0] = imread( filePath, CV_LOAD_IMAGE_ANYDEPTH ) ;

    //cout << filePath << endl ;

    //Mat rgbRectImage;

		//cv::undistort(rgbImage, rgbRectImage, cameraParameters.cameraMatrix, cameraParameters.distCoeffs);

		//imshow("rgbImge", rgbImage ) ;
		//imshow("rgbRectImage", rgbRectImage);
		//waitKey(0);

    depthImage[0].convertTo(depthImage[0], CV_32F);
    depthImage[0] /= 1000.0;

		for (int kk = 1; kk < maxPyramidLevel; kk++){
			pyrDownMeanSmooth<uchar>(grayImage[kk - 1], grayImage[kk]);
		}

		for (int kk = 1; kk < maxPyramidLevel; kk++){
			pyrDownMedianSmooth<float>(depthImage[kk - 1], depthImage[kk]);
		}


		if (vst == false )//the first frame
		{
      vst = true ;
      //cout << firstFrameTtoVICON << endl;
			slidingWindows.insertKeyFrame(grayImage, depthImage, Matrix3d::Identity(), Vector3d::Zero() );

			//slidingWindows.planeDection();

			R_k_c = Matrix3d::Identity();
			T_k_c = Vector3d::Zero();

			lastFrame = &slidingWindows.states[slidingWindows.tail];

			continue;
		}

		//tmpGradientThreshold = 0;
		//namedWindow("effect", 1);
		//createTrackbar("graidenThreshold", "effect", &tmpGradientThreshold, 200, on_trackbar);
		//on_trackbar(tmpGradientThreshold, 0);
		//cv::waitKey(0);

		//double t = (double)cvGetTickCount();

#ifdef FRAME_TO_FRAME
		frameToFrameDenseTracking(R_k_c, T_k_c);
#else
		keyframeToFrameDenseTracking(R_k_c, T_k_c );
#endif

		//t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
		//printf("cal time: %f\n", t);

		R_c_0 = slidingWindows.states[slidingWindows.tail].R_k0*R_k_c.transpose();
		T_c_0 = R_c_0*(
			R_k_c*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - T_k_c);

    cout << "currentPosition:\n" << T_c_0.transpose() << endl;

    if ((i % 3) == 1)
		{
			//double t = (double)cvGetTickCount();

			slidingWindows.insertKeyFrame(grayImage, depthImage, R_c_0, T_c_0 );


			//t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
			//printf("BA cal time: %f\n", t);

			//slidingWindows.planeDection();

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
		//if (i >= 580)
		//{
		//	imshow("currentImage", residualImage );
		//	imshow("grayImage", grayImage[0]);
		//	char c = waitKey(0);
		//	//printf("%c\n", c);
		//}
	}

	return 0;
}
