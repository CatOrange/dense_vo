#ifndef __STATEESTIMATION_H
#define __STATEESTIMATION_H

#include "dataStructure.h"
#include "variableDefinition.h"
#include "normalMapGeneration.h"
#include "AHClustering.h"
#include "utility.h"
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <list>
#include <omp.h>
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "sophus/se3.hpp"
using namespace std;
using namespace cv;
using namespace Eigen;


/*

for Bundle Adjustment

*/
//
//struct FT_INPUT
//{
//	int frame_id, video_id; // frame no use
//	int ft_id;
//	double x, y;
//};
//
//// ==== nonlinear ====
//struct CALIBRATION
//{
//	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//	Matrix3d R;
//	Vector3d T;
//};
//
//enum FT_LINK_MODE {
//	FT_MONO,
//	FT_STEREO_SYNC,
//	FT_STEREO_ASYNC
//};

struct SUPERPIXEL_INFO
{
public:
	std::vector<int> listOfU;
	std::vector<double> listOfU_;
	std::vector<int> listOfV;
	std::vector<double> listOfV_;
	std::vector<unsigned char>intensity;
	double averageGradient;
	SUPERPIXEL_INFO()
	{
		listOfU.clear();
		listOfU_.clear();
		listOfV.clear();
		listOfV_.clear();
		intensity.clear();
		averageGradient = 0;
	}
	~SUPERPIXEL_INFO(){
	}
};

struct PIXEL_INFO_IN_A_FRAME
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	vector<MatrixXd> Aij;
	vector<MatrixXd> AijTAij;
	MatrixXd piList;
	vector<double> intensity;
	PIXEL_INFO_IN_A_FRAME()
	{
		Aij.clear();
		AijTAij.clear();
		intensity.clear();
	}
	~PIXEL_INFO_IN_A_FRAME(){
	}
};

struct SUPERPIXEL_IN_3D_SPACE
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	int stateID;

	double normal[3];
	double depth;
	bool valid;

	bool ready;

	double lambda[3];
	double prior_lambda[3];
	Vector3d pk[3];
	MatrixXd u1xu2_T;
	MatrixXd u1xu2_T_u0;
	MatrixXd u0xu2_T;
	MatrixXd u0xu2_T_u0;
	MatrixXd u0xu1_T;
	MatrixXd u0xu1_T_u0;

	SUPERPIXEL_INFO superpixelsInPyramid[maxPyramidLevel];
	vector<short> reprojectList;

	SUPERPIXEL_IN_3D_SPACE(){
		reprojectList.clear();
		reprojectList.reserve(slidingWindowSize);
	}
	~SUPERPIXEL_IN_3D_SPACE(){
	}

	char used;
	// -1 == init
	// 0 == in sw_ft[] (in lamda)
	// 1 == parallax < threshold, not in lamda, need to be marginalized
	// 2 == outliers
};

class STATE // each frame information of sliding window
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Matrix3d R_k0;//R_k^0
	Vector3d T_k0;//T_k^0
	//Vector3d alpha, beta;
	//MatrixXd P_cov;
	unsigned long long timestamp;
	//int id; // frame ID
	int totalNumOfValidPixels[maxPyramidLevel];

	class STATE *next; // next frame
	// === nonlinear ===
	//Matrix3d R_nl;
	//Quaterniond dq;
	//MatrixXd P_nl; // P_cov_nl
	unsigned char* intensity[maxPyramidLevel];
	//[maxPyramidLevel][IMAGE_HEIGHT][IMAGE_WIDTH];
	float* depthImage[maxPyramidLevel];
	//[maxPyramidLevel][IMAGE_HEIGHT][IMAGE_WIDTH];
	double* gradientX[maxPyramidLevel];
	//[maxPyramidLevel][IMAGE_HEIGHT][IMAGE_WIDTH];
	double* gradientY[maxPyramidLevel];
	//[maxPyramidLevel][IMAGE_HEIGHT][IMAGE_WIDTH];
	PIXEL_INFO_IN_A_FRAME pixelInfo[maxPyramidLevel];

	/*
	vector<MatrixXd> Aij(height*width);
	vector<MatrixXd> AijTAij(height*width);
	vector<Vector3d> pi(height*width);
	*/
	STATE()
	{
		int height = IMAGE_HEIGHT;
		int width = IMAGE_WIDTH;
		for (int i = 0; i < maxPyramidLevel; i++)
		{
			intensity[i] = new unsigned char[height*width];
			depthImage[i] = new float[height*width];
			gradientX[i] = new double[height*width];
			gradientY[i] = new double[height*width];
			height >>= 1;
			width >>= 1;
		}
		R_k0 = Matrix3d::Identity();
		T_k0 = Vector3d::Zero();
	}

	~STATE()
	{
		for (int i = 0; i < maxPyramidLevel; i++)
		{
			if (intensity[i] != NULL){
				delete[] intensity[i];
			}
			if (depthImage[i] != NULL){
				delete[] depthImage[i];
			}
			if (gradientX[i] != NULL){
				delete[] gradientX[i];
			}
			if (gradientY[i] != NULL){
				delete[] gradientY[i];
			}
		}
	}

	void insertFrame(const Mat grayImage[maxPyramidLevel], const Mat depthImage[maxPyramidLevel], const Matrix3d& R, const Vector3d& T, CAMER_PARAMETERS* para)
	{
		//		STATE *current = this;
		//		Mat currentDepthImage;
		//
		//		//init the intensity and the depth value
		//		int n = IMAGE_HEIGHT;
		//		int m = IMAGE_WIDTH;
		//		for (int i = 0; i < maxPyramidLevel; i++)
		//		{
		//			memcpy(current->intensity[i], (unsigned char*)grayImage[i].data, n*m*sizeof(unsigned char));
		//			memcpy(current->depthImage[i], (float*)depthImage[i].data, n*m*sizeof(float));
		//			n >>= 1;
		//			m >>= 1;
		//		}
		//
		//		//init the graident map
		//		current->computeGradientMap( grayImage ) ;
		//
		//		//init the pixel info in a frame
		//		for (int level = maxPyramidLevel - 1; level >= 0; level--)
		//		{
		//			int n = IMAGE_HEIGHT >> level;
		//			int m = IMAGE_WIDTH >> level;
		//			float* pDepth = current->depthImage[level];
		//			double* pGradientX = current->gradientX[level];
		//			double* pGradientY = current->gradientY[level];
		//
		//			current->pixelInfo[level].valid.clear();
		//			current->pixelInfo[level].valid.resize(n*m);
		//			current->pixelInfo[level].pi.clear();
		//			current->pixelInfo[level].pi.resize(n*m);
		//			current->pixelInfo[level].Aij.clear();
		//			current->pixelInfo[level].Aij.resize(n*m);
		//			current->pixelInfo[level].AijTAij.clear();
		//			current->pixelInfo[level].AijTAij.resize(n*m);
		//			PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];
		//
		////			omp_set_num_threads(ompNumThreads);
		////#pragma omp parallel for 
		//			for (int u = 0; u < n; u++)
		//			{
		//				for (int v = 0; v < m; v++)
		//				{
		//					int k = INDEX(u, v, n, m);
		//					double Z = pDepth[k];
		//					if (Z < zeroThreshold) {
		//						currentPixelInfo.valid[k] = false;
		//						continue;
		//					}
		//					if (SQ(pGradientX[k]) + SQ(pGradientY[k]) < graidientThreshold){
		//						currentPixelInfo.valid[k] = false;
		//						continue;
		//					}
		//					currentPixelInfo.valid[k] = true;
		//
		//					double X = (v - para->cx[level]) * Z / para->fx[level];
		//					double Y = (u - para->cy[level]) * Z / para->fy[level];
		//
		//					currentPixelInfo.pi[k] << X, Y, Z;
		//
		//					MatrixXd oneBytwo(1, 2);
		//					MatrixXd twoBySix(2, 6);
		//
		//					oneBytwo(0, 0) = pGradientX[k];
		//					oneBytwo(0, 1) = pGradientY[k];
		//
		//					twoBySix(0, 0) = para->fx[level] / Z;
		//					twoBySix(0, 1) = 0;
		//					twoBySix(0, 2) = -X * para->fx[level] / SQ(Z);
		//					twoBySix(1, 0) = 0;
		//					twoBySix(1, 1) = para->fy[level] / Z;
		//					twoBySix(1, 2) = -Y * para->fy[level] / SQ(Z);
		//
		//					twoBySix(0, 3) = twoBySix(0, 2) * Y;
		//					twoBySix(0, 4) = twoBySix(0, 0)*Z - twoBySix(0, 2)*X;
		//					twoBySix(0, 5) = -twoBySix(0, 0)*Y;
		//					twoBySix(1, 3) = -twoBySix(1, 1)*Z + twoBySix(1, 2)*Y;
		//					twoBySix(1, 4) = -twoBySix(1, 2)* X;
		//					twoBySix(1, 5) = twoBySix(1, 1)* X;
		//
		//					//currentPixelInfo.Aij[k] = (oneBytwo*twoByThree*threeBySix).transpose();
		//					currentPixelInfo.Aij[k] = (oneBytwo*twoBySix).transpose();
		//					currentPixelInfo.AijTAij[k] = currentPixelInfo.Aij[k] * currentPixelInfo.Aij[k].transpose();
		//					/*
		//					int k = INDEX(u, v, n, m);
		//					double Z = pDepth[k];
		//					if (Z < zeroThreshold) {
		//						currentPixelInfo.valid[k] = false;
		//						continue;
		//					}
		//					if (SQ(pGradientX[k]) + SQ(pGradientY[k]) < graidientThreshold){
		//						currentPixelInfo.valid[k] = false;
		//						continue;
		//					}
		//					currentPixelInfo.valid[k] = true;
		//
		//					double X = (v - para->cx[level]) * Z / para->fx[level];
		//					double Y = (u - para->cy[level]) * Z / para->fy[level];
		//
		//					currentPixelInfo.pi[k] << X, Y, Z;
		//
		//					MatrixXd oneBytwo(1, 2);
		//					MatrixXd twoByThree(2, 3);
		//					MatrixXd threeBySix(3, 6);
		//					MatrixXd oneBySix(1, 6);
		//
		//					oneBytwo(0, 0) = pGradientX[k];
		//					oneBytwo(0, 1) = pGradientY[k];
		//
		//					twoByThree(0, 0) = para->fx[level] / Z;
		//					twoByThree(0, 1) = 0;
		//					twoByThree(0, 2) = -X * para->fx[level] / SQ(Z);
		//					twoByThree(1, 0) = 0;
		//					twoByThree(1, 1) = para->fy[level] / Z;
		//					twoByThree(1, 2) = -Y * para->fy[level] / SQ(Z);
		//
		//					threeBySix.topLeftCorner(3, 3) = Matrix3d::Identity();
		//					threeBySix(0, 3) = threeBySix(1, 4) = threeBySix(2, 5) = 0;
		//					threeBySix(0, 4) = Z;
		//					threeBySix(1, 3) = -Z;
		//					threeBySix(0, 5) = -Y;
		//					threeBySix(2, 3) = Y;
		//					threeBySix(1, 5) = X;
		//					threeBySix(2, 4) = -X;
		//
		//					currentPixelInfo.Aij[k] = (oneBytwo*twoByThree*threeBySix).transpose();
		//					currentPixelInfo.AijTAij[k] = currentPixelInfo.Aij[k] * currentPixelInfo.Aij[k].transpose();
		//					*/
		//				}
		//			}
		//		}
		//
		//		//init the pose
		//		current->R_k0 = R;
		//		current->T_k0 = T;
		//		
	}

	void computeGradientMap(const Mat grayImage[maxPyramidLevel])
	{
		int height = IMAGE_HEIGHT;
		int width = IMAGE_WIDTH;

		//for eaxh pyramid level
		for (int id = 0; id < maxPyramidLevel; id++)
		{
			totalNumOfValidPixels[id] = 0;
			float* pDepth = depthImage[id];
			double* pGradientX = gradientX[id];
			double* pGradientY = gradientY[id];
			unsigned char* pIntensity = intensity[id];

			//Mat gradientXMap;
			//Mat gradientYMap;

			//Sobel(grayImage[id], gradientXMap, CV_64F, 1, 0 );
			//Sobel(grayImage[id], gradientYMap, CV_64F, 0, 1 );

			//memcpy(pGradientX, gradientXMap.data, height*width*sizeof(double));
			//memcpy(pGradientY, gradientYMap.data, height*width*sizeof(double));

			//calculate gradient map
			//			omp_set_num_threads(ompNumThreads);
			//#pragma omp parallel for 
			for (int i = height - 2; i > 0; i--)
			{
				for (int j = width - 2; j > 0; j--)
				{
					if (pDepth[INDEX(i, j, height, width)] > zeroThreshold){
						totalNumOfValidPixels[id]++;
					}
					//pGradientX[INDEX(i, j, height, width)] = 0.125*(
					//	(int)pIntensity[INDEX(i - 1, j + 1, height, width)] - (int)pIntensity[INDEX(i - 1, j - 1, height, width)] +
					//	((int)pIntensity[INDEX(i, j + 1, height, width)] << 1) - ((int)pIntensity[INDEX(i, j - 1, height, width)] << 1) +
					//	(int)pIntensity[INDEX(i + 1, j + 1, height, width)] - (int)pIntensity[INDEX(i + 1, j - 1, height, width)]
					//	);
					//pGradientY[INDEX(i, j, height, width)] = 0.125*(
					//	(int)pIntensity[INDEX(i + 1, j - 1, height, width)] - (int)pIntensity[INDEX(i - 1, j - 1, height, width)] +
					//	((int)pIntensity[INDEX(i + 1, j, height, width)] << 1) - ((int)pIntensity[INDEX(i - 1, j, height, width)] << 1) +
					//	(int)pIntensity[INDEX(i + 1, j + 1, height, width)] - (int)pIntensity[INDEX(i - 1, j + 1, height, width)]
					//	);
					pGradientX[INDEX(i, j, height, width)] = 0.5*(
						(double)pIntensity[INDEX(i, j + 1, height, width)] - (double)pIntensity[INDEX(i, j - 1, height, width)]
						);
					pGradientY[INDEX(i, j, height, width)] = 0.5*(
						(double)pIntensity[INDEX(i + 1, j, height, width)] - (double)pIntensity[INDEX(i - 1, j, height, width)]
						);
				}
			}

			//			omp_set_num_threads(ompNumThreads);
			//#pragma omp parallel for 
			for (int i = height - 2; i > 0; i--)
			{

				if (pDepth[INDEX(i, 0, height, width)] > zeroThreshold){
					totalNumOfValidPixels[id]++;
				}
				if (pDepth[INDEX(i, width - 1, height, width)] > zeroThreshold){
					totalNumOfValidPixels[id]++;
				}

				pGradientX[INDEX(i, 0, height, width)] = pGradientX[INDEX(i, 1, height, width)];
				pGradientY[INDEX(i, 0, height, width)] = pGradientY[INDEX(i, 1, height, width)];
				pGradientX[INDEX(i, width - 1, height, width)] = pGradientX[INDEX(i, width - 2, height, width)];
				pGradientY[INDEX(i, width, height, width)] = pGradientY[INDEX(i, width - 2, height, width)];
			}

			//			omp_set_num_threads(ompNumThreads);
			//#pragma omp parallel for 
			for (int j = width - 1; j >= 0; j--)
			{

				if (pDepth[INDEX(0, j, height, width)] > zeroThreshold){
					totalNumOfValidPixels[id]++;
				}
				if (pDepth[INDEX(height - 1, j, height, width)] > zeroThreshold){
					totalNumOfValidPixels[id]++;
				}

				pGradientX[INDEX(0, j, height, width)] = pGradientX[INDEX(1, j, height, width)];
				pGradientY[INDEX(0, j, height, width)] = pGradientY[INDEX(1, j, height, width)];
				pGradientX[INDEX(height - 1, j, height, width)] = pGradientX[INDEX(height - 2, j, height, width)];
				pGradientY[INDEX(height - 1, j, height, width)] = pGradientY[INDEX(height - 2, j, height, width)];
			}

			height >>= 1;
			width >>= 1;
		}
	}
};

class STATEESTIMATION
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		//camera parameters
		int height, width;
	CAMER_PARAMETERS* para;
	//kMeansClustering clustering;
	AHClustering clustering;

	//state sequences
	int head, tail;
	STATE states[slidingWindowSize];

	//superpixels in sliding wiindow
	int numOfSuperpixelsInSlidingWindow;
	int numOfState;
	int idCnt;
	std::list<SUPERPIXEL_IN_3D_SPACE> superpixelList;

	//occlusion mask for tomporal use only
	bool mask[IMAGE_HEIGHT][IMAGE_WIDTH];
	int maskStateID[IMAGE_HEIGHT][IMAGE_WIDTH];
	int maskSuperpixelID[IMAGE_HEIGHT][IMAGE_WIDTH];
	double maskDistance[IMAGE_HEIGHT][IMAGE_WIDTH];

	//for dense tracking
	Vector3d last_delta_v, last_delta_w;
	Matrix3d last_delta_R;
	Vector3d last_delta_T;

	STATEESTIMATION(int hh, int ww, CAMER_PARAMETERS* p)
	{
		height = hh;
		width = ww;
		para = p;
		clustering.setCameraParameters(height, width, para->fx[0], para->fy[0], para->cx[0], para->cy[0]);

		initSlidingWindow();
		numOfSuperpixelsInSlidingWindow = 0;
		numOfState = 0;
		idCnt = 0;
		memset(mask, true, sizeof(mask));

		superpixelList.clear();
		last_delta_v.setZero();
		last_delta_w.setZero();
		last_delta_R.setIdentity();
		last_delta_T.setZero();
	}

	~STATEESTIMATION(){
	}

	void initSlidingWindow()
	{
		head = 0;
		tail = -1;
		for (int i = 0; i < slidingWindowSize; i++)
		{
			//states[i].pts = NULL;
			if (i + 1 < slidingWindowSize){
				states[i].next = &states[i + 1];
			}
		}
		states[slidingWindowSize - 1].next = &states[0];
	}

	void insertSuperpixel(int K, int iterNum)
	{
		STATE *current = &states[tail];

		clustering.setInputData(current->depthImage[0]);
		clustering.setMask(&mask[0][0]);
		//clustering.runClustering(K, iterNum);
		clustering.runClustering(K);

		int currentNumOfSuperpixel = clustering.seedsNum;

		//printf("clustering num:%d\n", currentNumOfSuperpixel);

		vector<double>depthEstimation(currentNumOfSuperpixel, 0.0);

#ifdef DEBUG_CLUSTRING
		printf("currentNumOfSuperpixel=%d\n", currentNumOfSuperpixel);
		Mat now(height, width, CV_8UC3);
		unsigned char*pIntensityDebug = current->intensity[0];
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++){
				now.at<cv::Vec3b>(i, j)[0] = pIntensityDebug[INDEX(i, j, height, width)];
				now.at<cv::Vec3b>(i, j)[1] = pIntensityDebug[INDEX(i, j, height, width)];
				now.at<cv::Vec3b>(i, j)[2] = pIntensityDebug[INDEX(i, j, height, width)];
			}
		}
		double proportion = 0.3;
		RNG rng(168);
		vector<short>R(currentNumOfSuperpixel);
		vector<short>G(currentNumOfSuperpixel);
		vector<short>B(currentNumOfSuperpixel);
		vector<double> errorCheck(currentNumOfSuperpixel);
		for (int i = 0; i < currentNumOfSuperpixel; i++){
			R[i] = rng.uniform(0, 255);
			G[i] = rng.uniform(0, 255);
			B[i] = rng.uniform(0, 255);
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (clustering.labels[i][j] < 1){
					continue;
				}
				int id = clustering.labels[i][j] - 1;
				now.at<cv::Vec3b>(i, j)[0] = now.at<cv::Vec3b>(i, j)[0] * proportion + R[id] * (1 - proportion);
				now.at<cv::Vec3b>(i, j)[1] = now.at<cv::Vec3b>(i, j)[1] * proportion + G[id] * (1 - proportion);
				now.at<cv::Vec3b>(i, j)[2] = now.at<cv::Vec3b>(i, j)[2] * proportion + B[id] * (1 - proportion);
			}
		}
		imshow("now", now);
		waitKey(0);
#endif

		//1. init
		vector<std::list<SUPERPIXEL_IN_3D_SPACE>::iterator> iterList(currentNumOfSuperpixel);
		for (int i = 0; i < currentNumOfSuperpixel; i++)//actual ID is [1, currentNumOfSuperpixel], 0 is the label for invalid pixels
		{
			SUPERPIXEL_IN_3D_SPACE tmp;

			tmp.stateID = tail;
			tmp.normal[0] = clustering.kSeedNormal[i + 1][0];
			tmp.normal[1] = clustering.kSeedNormal[i + 1][1];
			tmp.normal[2] = clustering.kSeedNormal[i + 1][2];
			tmp.valid = true;
			numOfSuperpixelsInSlidingWindow++;

			iterList[i] = superpixelList.emplace(superpixelList.end(), tmp);

			//clear reproject list
			iterList[i]->reprojectList.clear();
			//reserver space
			int kk = clustering.numOfEachLabels[i + 1];
			for (int ith = 0; ith < maxPyramidLevelBA; ith++)
			{
				iterList[i]->superpixelsInPyramid[ith].listOfU.clear();
				iterList[i]->superpixelsInPyramid[ith].listOfU.reserve(kk);
				iterList[i]->superpixelsInPyramid[ith].listOfU_.clear();
				iterList[i]->superpixelsInPyramid[ith].listOfU_.reserve(kk);
				iterList[i]->superpixelsInPyramid[ith].listOfV.clear();
				iterList[i]->superpixelsInPyramid[ith].listOfV.reserve(kk);
				iterList[i]->superpixelsInPyramid[ith].listOfV_.clear();
				iterList[i]->superpixelsInPyramid[ith].listOfV_.reserve(kk);
				iterList[i]->superpixelsInPyramid[ith].intensity.clear();
				iterList[i]->superpixelsInPyramid[ith].intensity.reserve(kk);
				iterList[i]->superpixelsInPyramid[ith].averageGradient = 0.0;
			}
		}
		//printf("%d %d\n", currentNumOfSuperpixel, superpixelList.size());
		//for (int i = 0; i < currentNumOfSuperpixel; i++){
		//	printf("%d %f %f %f\n", iterList[i]->stateID, iterList[i]->normal[0], iterList[i]->normal[1], iterList[i]->normal[2] );
		//}


		//2. insert 2D observation
		double* pGradientX = states[tail].gradientX[0];
		double* pGradientY = states[tail].gradientY[0];
		unsigned char* pIntensity = states[tail].intensity[0];
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (clustering.labels[i][j] < 1){
					continue;
				}

				//gridient filtering
				if (SQ(pGradientX[INDEX(i, j, height, width)]) + SQ(pGradientY[INDEX(i, j, height, width)]) < graidientThreshold){
					continue;
				}

				int labelID = clustering.labels[i][j] - 1;
				iterList[labelID]->superpixelsInPyramid[0].listOfU.push_back(i);
				iterList[labelID]->superpixelsInPyramid[0].listOfU_.push_back((i - para->cy[0]) / para->fy[0]);
				iterList[labelID]->superpixelsInPyramid[0].listOfV.push_back(j);
				iterList[labelID]->superpixelsInPyramid[0].listOfV_.push_back((j - para->cx[0]) / para->fx[0]);
				iterList[labelID]->superpixelsInPyramid[0].intensity.push_back(pIntensity[INDEX(i, j, height, width)]);
				iterList[labelID]->superpixelsInPyramid[0].averageGradient += SQ(pGradientX[INDEX(i, j, height, width)]) + SQ(pGradientY[INDEX(i, j, height, width)]);

				double x, y, z;
				z = clustering.depth[INDEX(i, j, height, width)];
				//if (z < zeroThreshold){
				//	puts("@@@@@@@@@@@");
				//}
				y = (i - para->cy[0]) * z / para->fy[0];
				x = (j - para->cx[0]) * z / para->fx[0];
				depthEstimation[labelID] += x * clustering.kSeedNormal[labelID + 1][0] + y * clustering.kSeedNormal[labelID + 1][1] + z * clustering.kSeedNormal[labelID + 1][2];
			}
		}

		//3. calculate the depth of the superpixel & select support points
		for (int i = 0; i < currentNumOfSuperpixel; i++)
		{
			iterList[i]->depth = -depthEstimation[i] / clustering.numOfEachLabels[i + 1];

			//minimum number filtering
			if (iterList[i]->superpixelsInPyramid[0].listOfU_.size() < minOptNum){
				iterList[i]->valid = false;
				continue;
			}

			SUPERPIXEL_INFO& currentSuperPixel = iterList[i]->superpixelsInPyramid[0];
			int sz = currentSuperPixel.listOfU.size();

			vector<HEAPNODE> XYAdd(sz);
			vector<HEAPNODE> XYMinus(sz);
			for (int k = 0; k < sz; k++)
			{
				int u = currentSuperPixel.listOfU[k];
				int v = currentSuperPixel.listOfV[k];
				XYAdd[k].MSE = u + v;
				XYAdd[k].id = k;

				XYMinus[k].MSE = u - v;
				XYMinus[k].id = k;
			}
			sort(XYAdd.begin(), XYAdd.end());
			sort(XYMinus.begin(), XYMinus.end());
			int IDs[4];
			IDs[0] = XYAdd[0].id;
			IDs[1] = XYAdd[sz - 1].id;
			IDs[2] = XYMinus[0].id;
			IDs[3] = XYMinus[sz - 1].id;
			sort(IDs, IDs + 4);
			std::unique(IDs, IDs + 4);

			//select three points
			for (int k = 0; k < 3; k++)
			{
				int u = currentSuperPixel.listOfU[IDs[k]];
				int v = currentSuperPixel.listOfV[IDs[k]];

				double z = current->depthImage[0][INDEX(u, v, height, width)];
				double y = (u - para->cy[0]) / para->fy[0];
				double x = (v - para->cx[0]) / para->fx[0];

				iterList[i]->pk[k] << x, y, 1;
				iterList[i]->lambda[k] = z;
				iterList[i]->prior_lambda[k] = z;
			}
			iterList[i]->u1xu2_T = (iterList[i]->pk[1].cross(iterList[i]->pk[2])).transpose();
			iterList[i]->u1xu2_T_u0 = iterList[i]->u1xu2_T * iterList[i]->pk[0];
			iterList[i]->u0xu2_T = (iterList[i]->pk[0].cross(iterList[i]->pk[2])).transpose();
			iterList[i]->u0xu2_T_u0 = iterList[i]->u0xu2_T * iterList[i]->pk[0];
			iterList[i]->u0xu1_T = (iterList[i]->pk[0].cross(iterList[i]->pk[1])).transpose();
			iterList[i]->u0xu1_T_u0 = iterList[i]->u0xu1_T * iterList[i]->pk[0];

#ifdef  DEBUG_INSERT_SUPERPIXEL
			printf("Superpixel=%d\n", i);
			Mat now(height, width, CV_8UC3);
			unsigned char*pIntensity = current->intensity[0];
			for (int iDebug = 0; iDebug < height; iDebug++)
			{
				for (int jDebug = 0; jDebug < width; jDebug++){
					now.at<cv::Vec3b>(iDebug, jDebug)[0] = pIntensity[INDEX(iDebug, jDebug, height, width)];
					now.at<cv::Vec3b>(iDebug, jDebug)[1] = pIntensity[INDEX(iDebug, jDebug, height, width)];
					now.at<cv::Vec3b>(iDebug, jDebug)[2] = pIntensity[INDEX(iDebug, jDebug, height, width)];
				}
			}
			double proportion = 0.3;
#endif

#ifdef  DEBUG_INSERT_SUPERPIXEL
			imshow("now", now);
			waitKey(0);
#endif

		}

		//4. calculate the pyramid observation of superpixel
		for (int level = 1; level < maxPyramidLevelBA; level++)
		{
			int n = height >> level;
			int m = width >> level;
			double* pGradientX = states[tail].gradientX[level];
			double* pGradientY = states[tail].gradientY[level];
			int totalNum = 1 << (level * 2);
			int halfNum = totalNum / 2;
			vector<int> numList(totalNum);
			unsigned char* pIntensity = states[tail].intensity[level];
			for (int y = 0; y < n; y++)
			{
				for (int x = 0; x < m; x++)
				{
					int cnt = 0;
					int leftTopY = y << level;
					int leftTopX = x << level;
					int rightDownY = (y + 1) << level;
					int rightDownX = (x + 1) << level;
					for (int i = leftTopY; i < rightDownY; i++)
					{
						for (int j = leftTopX; j < rightDownX; j++)
						{
							if (clustering.labels[i][j] < 1){
								continue;
							}
							int labelID = clustering.labels[i][j] - 1;
							if (iterList[labelID]->valid == false){
								continue;
							}
							numList[cnt++] = labelID;
						}
					}
					if (cnt == 0){
						continue;
					}
					sort(numList.begin(), numList.begin() + cnt);
					int num = 0;
					int labelID = findMaxContinousLength(numList, cnt, num);
					if (num < halfNum){
						continue;
					}

					iterList[labelID]->superpixelsInPyramid[level].listOfU.push_back(y);
					iterList[labelID]->superpixelsInPyramid[level].listOfU_.push_back((y - para->cy[level]) / para->fy[level]);
					iterList[labelID]->superpixelsInPyramid[level].listOfV.push_back(x);
					iterList[labelID]->superpixelsInPyramid[level].listOfV_.push_back((x - para->cx[level]) / para->fx[level]);
					iterList[labelID]->superpixelsInPyramid[level].intensity.push_back(pIntensity[INDEX(y, x, n, m)]);
					iterList[labelID]->superpixelsInPyramid[level].averageGradient += SQ(pGradientX[INDEX(y, x, n, m)]) + SQ(pGradientY[INDEX(y, x, n, m)]);
				}
			}
		}

		//5. check the validness of superpixel
		for (int i = 0; i < currentNumOfSuperpixel; i++)
		{
			for (int level = maxPyramidLevelBA - 1; level >= 0; level--)
			{
				SUPERPIXEL_INFO& currentSuperpixel = iterList[i]->superpixelsInPyramid[level];
				currentSuperpixel.averageGradient /= currentSuperpixel.listOfU.size();
				//if (currentSuperpixel.averageGradient < graidientThreshold){
				//	iterList[i]->valid = false;
				//}
			}
		}

	}

	void checkNormalProjectionOnNewKeyFrame()
	{
		memset(mask, true, sizeof(mask));//clustering will be banned if false

		int n = height;
		int m = width;

		int level = 0;
		unsigned char *nextIntensity = states[tail].intensity[level];
		int linkStateID = tail;

		std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter;
		for (iter = superpixelList.begin(); iter != superpixelList.end(); iter++)
		{
			int currentStateID = iter->stateID;
			const SUPERPIXEL_INFO& currentSuperpixel = iter->superpixelsInPyramid[level];

			//double totalError = 0.0;
			int sz = currentSuperpixel.listOfU.size();
			//unsigned char* pIntensity = states[currentStateID].intensity[level];

			Vector3d p0 = iter->pk[0] * iter->lambda[0];
			Vector3d p1 = iter->pk[1] * iter->lambda[1];
			Vector3d p2 = iter->pk[2] * iter->lambda[2];
			Vector3d p02 = p2 - p0;
			Vector3d p01 = p1 - p0;
			MatrixXd normalT = p02.cross(p01).transpose();
			MatrixXd numerotor = normalT*p0;
			//bool flag = false ;
			for (int i = 0; i < sz; i++)//for every pixel within a superpixel
			{
				Vector3d u_e;
				u_e << currentSuperpixel.listOfV_[i], currentSuperpixel.listOfU_[i], 1.0;

				MatrixXd denorminator = normalT*u_e;
				double lambda = numerotor(0, 0) / denorminator(0, 0);
				Vector3d pi = lambda * u_e;
				Vector3d pj = states[linkStateID].R_k0.transpose()*(states[currentStateID].T_k0 - states[linkStateID].T_k0 + states[currentStateID].R_k0*pi);

				if (pj(2) < zeroThreshold){
					continue;
				}

				int u2 = pj(1)*para->fy[level] / pj(2) + para->cy[level] + 0.5;
				int v2 = pj(0)*para->fx[level] / pj(2) + para->cx[level] + 0.5;

				if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
					continue;
				}

				//double w = 1.0;
				//double r = nextIntensity[INDEX(u2, v2, n, m)] - currentSuperpixel.intensity[i];
				//double r_fabs = fabs(r);
				//if (r_fabs > huberKernelThreshold){
				//	w = huberKernelThreshold / r_fabs;
				//}
				//if (w > validPixelThreshold){
				//	validNum++;
				//}
				double w = 1.0;
				double r = currentSuperpixel.intensity[i] - nextIntensity[INDEX(u2, v2, n, m)];
				double r_fabs = fabs(r);
				if (r_fabs > huberKernelThreshold){
					w = huberKernelThreshold / r_fabs;
				}
				if (w > validPixelThreshold){
					//validNum++;
					mask[u2][v2] = false;
				}
			}
		}
	}

	void insertKeyFrame(const Mat grayImage[maxPyramidLevel], const Mat depthImage[maxPyramidLevel], const Matrix3d& R, const Vector3d& T)
	{
		if (numOfState == slidingWindowSize){
			//puts("pop state");
			popOldestState();
		}
		tail++;
		numOfState++;
		if (tail >= slidingWindowSize){
			tail -= slidingWindowSize;
		}
		STATE *current = &states[tail];
    //Mat currentDepthImage;

		//init the intensity and the depth value
		int n = height;
		int m = width;
		for (int i = 0; i < maxPyramidLevel; i++)
		{
			memcpy(current->intensity[i], (unsigned char*)grayImage[i].data, n*m*sizeof(unsigned char));
			memcpy(current->depthImage[i], (float*)depthImage[i].data, n*m*sizeof(float));
			n >>= 1;
			m >>= 1;
		}

		//init the graident map
		current->computeGradientMap(grayImage);

		//init the pixel info in a frame
		for (int level = maxPyramidLevel - 1; level >= 0; level--)
		{
			int n = height >> level;
			int m = width >> level;
			float* pDepth = current->depthImage[level];
			unsigned char*pIntensity = current->intensity[level];
			double* pGradientX = current->gradientX[level];
			double* pGradientY = current->gradientY[level];

			int validNum = 0;
			vector<GRADIENTNODE> gradientList(n*m);
			for (int u = 0; u < n; u++)
			{
				for (int v = 0; v < m; v++)
				{
					int k = INDEX(u, v, n, m);
					double Z = pDepth[k];
					if (Z < zeroThreshold) {
						continue;
					}
					if (SQ(pGradientX[k]) + SQ(pGradientY[k]) < graidientThreshold){
						continue;
					}
					gradientList[validNum].cost = SQ(pGradientX[k]) + SQ(pGradientY[k]);
					gradientList[validNum].u = u;
					gradientList[validNum].v = v;
					validNum++;
				}
			}
      printf("dense tracking validNum: %d\n", validNum ) ;

			sort(&gradientList[0], &gradientList[validNum]);

			int bin[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
			int tmpCnt = 0;
			int numThrehold = (minDenseTrackingNum >> level) / 8;
			//int numThrehold = 1000000000 ;
			for (int i = 0; i < validNum; i++)
			{
				int u = gradientList[i].u;
				int v = gradientList[i].v;
				int k = INDEX(u, v, n, m);
				int index = angelSpace(pGradientX[k], pGradientY[k]);
				if (bin[index] < numThrehold){
					bin[index]++;
					gradientList[tmpCnt++] = gradientList[i];
				}
			}

			//validNum = std::min( validNum, minDenseTrackingNum >> level ) ;
			validNum = tmpCnt;

			current->pixelInfo[level].Aij.clear();
			current->pixelInfo[level].Aij.resize(validNum);
			current->pixelInfo[level].AijTAij.clear();
			current->pixelInfo[level].AijTAij.resize(validNum);
			current->pixelInfo[level].intensity.clear();
			current->pixelInfo[level].intensity.resize(validNum);
			PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];
			currentPixelInfo.piList.resize(3, validNum);

			//			omp_set_num_threads(ompNumThreads);
			//#pragma omp parallel for 
			for (int cnt = 0; cnt < validNum; cnt++)
			{
				int u = gradientList[cnt].u;
				int v = gradientList[cnt].v;
				int k = INDEX(u, v, n, m);
				double Z = pDepth[k];

				double X = (v - para->cx[level]) * Z / para->fx[level];
				double Y = (u - para->cy[level]) * Z / para->fy[level];

				currentPixelInfo.piList(0, cnt) = X;
				currentPixelInfo.piList(1, cnt) = Y;
				currentPixelInfo.piList(2, cnt) = Z;
				currentPixelInfo.intensity[cnt] = pIntensity[k];

				MatrixXd oneBytwo(1, 2);
				MatrixXd twoBySix(2, 6);

				oneBytwo(0, 0) = pGradientX[k];
				oneBytwo(0, 1) = pGradientY[k];

				//twoByThree(0, 0) = para->fx[level] / Z;
				//twoByThree(0, 1) = 0;
				//twoByThree(0, 2) = -X * para->fx[level] / SQ(Z);
				//twoByThree(1, 0) = 0;
				//twoByThree(1, 1) = para->fy[level] / Z;
				//twoByThree(1, 2) = -Y * para->fy[level] / SQ(Z);

				//threeBySix.topLeftCorner(3, 3) = Matrix3d::Identity();
				//threeBySix(0, 3) = threeBySix(1, 4) = threeBySix(2, 5) = 0;
				//threeBySix(0, 4) = Z;
				//threeBySix(1, 3) = -Z;
				//threeBySix(0, 5) = -Y;

				//threeBySix(2, 3) = Y;
				//threeBySix(1, 5) = X;
				//threeBySix(2, 4) = -X;

				twoBySix(0, 0) = para->fx[level] / Z;
				twoBySix(0, 1) = 0;
				twoBySix(0, 2) = -X * para->fx[level] / SQ(Z);
				twoBySix(1, 0) = 0;
				twoBySix(1, 1) = para->fy[level] / Z;
				twoBySix(1, 2) = -Y * para->fy[level] / SQ(Z);

				twoBySix(0, 3) = twoBySix(0, 2) * Y;
				twoBySix(0, 4) = twoBySix(0, 0)*Z - twoBySix(0, 2)*X;
				twoBySix(0, 5) = -twoBySix(0, 0)*Y;
				twoBySix(1, 3) = -twoBySix(1, 1)*Z + twoBySix(1, 2)*Y;
				twoBySix(1, 4) = -twoBySix(1, 2)* X;
				twoBySix(1, 5) = twoBySix(1, 1)* X;

				//currentPixelInfo.Aij[k] = (oneBytwo*twoByThree*threeBySix).transpose();
				currentPixelInfo.Aij[cnt] = (oneBytwo*twoBySix).transpose();
				currentPixelInfo.AijTAij[cnt] = currentPixelInfo.Aij[cnt] * currentPixelInfo.Aij[cnt].transpose();
			}
		}

		//init the pose
		current->R_k0 = R;
		current->T_k0 = T;
	}

	void planeDection()
	{
		//init the reprojection list
		if (numOfState > 1){
			checkNormalProjectionOnNewKeyFrame();
		}

		//init the superpixel
		int iterNum = 3;
		int K = expectedClusteringNum;
		insertSuperpixel(K, iterNum);
	}

	double maxAbsValueOfVector(const VectorXd&a)
	{
		double maxValue = fabs(a(0));
		for (int i = 1; i < 6; i++)
		{
			double tmp = fabs(a(i));
			if (tmp > maxValue){
				maxValue = tmp;
			}
		}
		return maxValue;
	}

	Matrix3d vectorToSkewMatrix(const Vector3d& w)
	{
		Matrix3d skewW(3, 3);
		skewW(0, 0) = skewW(1, 1) = skewW(2, 2) = 0;
		skewW(0, 1) = -w(2);
		skewW(1, 0) = w(2);
		skewW(0, 2) = w(1);
		skewW(2, 0) = -w(1);
		skewW(1, 2) = -w(0);
		skewW(2, 1) = w(0);

		return skewW;
	}

	void updateR_T(Matrix3d& R, Vector3d& T, const Vector3d& v, const Vector3d& w, Matrix3d& incR, Vector3d& incT)
	{
		Matrix3d skewW = vectorToSkewMatrix(w);

		
		double theta = sqrt(w.squaredNorm());
		Matrix3d deltaR = Matrix3d::Identity() + (sin(theta) / theta)*skewW + ((1 - cos(theta)) / (theta*theta))*skewW*skewW;
		Vector3d deltaT = (Matrix3d::Identity() + ((1 - cos(theta)) / (theta*theta)) *skewW + ((theta - sin(theta)) / (theta*theta*theta)*skewW*skewW)) * v;
		

		//Matrix3d deltaR = Matrix3d::Identity() + skewW ;
		//Vector3d deltaT = v;

		//Matrix3d newR = R*deltaR.transpose();
		//Vector3d newT = -R*deltaR.transpose()*deltaT + T;

		Matrix3d newR = R*deltaR;
		Vector3d newT = R*deltaT + T;

		incT = incR*deltaT + incT;
		incR = incR*deltaR;

		R = newR;
		T = newT;
	}

	void updateR_T_withoutScaling(Matrix3d& R, Vector3d& T, const Vector3d& v, const Vector3d& w, Matrix3d& incR, Vector3d& incT)
	{		
		incR *= (Matrix3d::Identity() + vectorToSkewMatrix(w));
		incT += v;

		R *= (Matrix3d::Identity() + vectorToSkewMatrix(w)) ;
		T += v;
	}

	void denseTrackingWithoutSuperpixel(STATE* current, const Mat grayImage[maxPyramidLevel], Matrix3d& R, Vector3d& T)
	{
		//no assumption on angular and linear velocity
		Matrix3d tmpR = R;
		Vector3d tmpT = T;

		//linear assumption on angular and linear velocity
		//Matrix3d tmpR = last_delta_R * R ;
		//Vector3d tmpT = last_delta_R * T + last_delta_T;

		Matrix3d incR = Matrix3d::Identity();
		Vector3d incT = Vector3d::Zero();
		for (int level = maxPyramidLevel - 1; level >= 0; level--)
		{
			int n = height >> level;
			int m = width >> level;
			unsigned char *nextIntensity = (unsigned char*)grayImage[level].data;
			PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];
			double lastError = 100000000000.0;
			last_delta_v.Zero();
			last_delta_w.Zero();

#ifdef DEBUG_DENSETRACKING
			//if ( level == 0 )
			//float* pDepth = current->depthImage[level];
			//double* pGradientX = current->gradientX[level];
			//double* pGradientY = current->gradientY[level];
			unsigned char* pIntensity = current->intensity[level];
			//double proportion = 0.3;

			//Mat now(n, m, CV_8UC3);
			Mat gradientMap(n, m, CV_8UC3) ;
			Mat next;
			Mat residualImage(n, m, CV_8U);
			//cv::cvtColor(grayImage[level], next, CV_GRAY2BGR);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < m; j++)
				{
					gradientMap.at<cv::Vec3b>(i, j)[0] = pIntensity[INDEX(i, j, n, m)];
					gradientMap.at<cv::Vec3b>(i, j)[1] = pIntensity[INDEX(i, j, n, m)];
					gradientMap.at<cv::Vec3b>(i, j)[2] = pIntensity[INDEX(i, j, n, m)];
				}
			}

#endif

			for (int ith = 0; ith < maxIteration; ith++)
			{
#ifdef DEBUG_DENSETRACKING
				//cout << tmpR << endl;
				//cout << tmpT << endl;
				cv::cvtColor(grayImage[level], next, CV_GRAY2BGR);

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < m; j++)
					{
						residualImage.at<uchar>(i, j) = 0;
					}
				}
#endif
				double currentError = 0;

				int actualNum = 0;
				MatrixXd ATA = MatrixXd::Zero(6, 6);
				VectorXd ATb = VectorXd::Zero(6);

				MatrixXd pi2List = tmpR * currentPixelInfo.piList;

				int validNum = currentPixelInfo.Aij.size();
				for (int i = 0; i < validNum; i++)
				{
					Vector3d p2 = pi2List.block(0, i, 3, 1) + tmpT;
					//Vector3d p2 = tmpR* currentPixelInfo.piList.block(0, i, 3, 1) + tmpT;

#ifdef DEBUG_DENSETRACKING
					Vector3d p1 = currentPixelInfo.piList.block(0, i, 3, 1) ;
					int u = int(p1(1)*para->fy[level] / p1(2) + para->cy[level] + 0.5);
					int v = int(p1(0)*para->fx[level] / p1(2) + para->cx[level] + 0.5);

					gradientMap.at<cv::Vec3b>(u, v)[0] = 0;
					gradientMap.at<cv::Vec3b>(u, v)[1] = 255;
					gradientMap.at<cv::Vec3b>(u, v)[2] = 0;
#endif

					int u2 = int(p2(1)*para->fy[level] / p2(2) + para->cy[level] + 0.5);
					int v2 = int(p2(0)*para->fx[level] / p2(2) + para->cx[level] + 0.5);

					double reprojectIntensity;
					if (linearIntepolation(u2, v2, nextIntensity, n, m, reprojectIntensity) == false){
						continue;
					}
					//if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
					//	continue;
					//}

					//#ifdef DEBUG_DENSETRACKING
					//						next.at<cv::Vec3b>(u2, v2)[0] = proportion*next.at<cv::Vec3b>(u2, v2)[0] + (1 - proportion) * R[pointsLabel[i]];
					//						next.at<cv::Vec3b>(u2, v2)[1] = proportion*next.at<cv::Vec3b>(u2, v2)[1] + (1 - proportion) * G[pointsLabel[i]];
					//						next.at<cv::Vec3b>(u2, v2)[2] = proportion*next.at<cv::Vec3b>(u2, v2)[2] + (1 - proportion) * B[pointsLabel[i]];
					//#endif
					double w = 1.0;
					//double r = pIntensity[k] - nextIntensity[INDEX(u2, v2, n, m)];
					double r = currentPixelInfo.intensity[i] - reprojectIntensity;
					double r_fabs = fabs(r);

#ifdef WEIGHTEDCOST
					if (r_fabs > huberKernelThreshold){
						w = huberKernelThreshold / (r_fabs);
					}
#endif

#ifdef DEBUG_DENSETRACKING
					residualImage.at<uchar>(u2, v2) = (uchar)r_fabs*10;
#endif
					currentError += w*r_fabs;
					actualNum++;
					ATA += w*currentPixelInfo.AijTAij[i];
					ATb -= w*r*currentPixelInfo.Aij[i];
				}

#ifdef DEBUG_DENSETRACKING
				cv::imshow("gradientMap", gradientMap);
				cv::imshow("next", next);

				cv::Mat falsecolorsmap;
				cv::applyColorMap(residualImage, falsecolorsmap, cv::COLORMAP_RAINBOW);

				char tmp[256] ;
				sprintf(tmp, "Resid" ) ;

				cv::imshow(tmp, falsecolorsmap);

				cv::waitKey(500);
#endif

				if (actualNum < 6){
					puts("Dense Tracking: lack of rank!break!");
					break;
				}

				//        if (actualNum < (minDenseTrackingNum>>level) ){
				//					puts("Dense Tracking: gradients are not rich!");
				//					break;
				//				}

				if (currentError > lastError){
					//revert
					updateR_T(tmpR, tmpT, -last_delta_v, -last_delta_w, incR, incT);
					break;
				}
				else{
					lastError = currentError;
				}

#ifdef ADD_VELOCITY_PRIOR
				Sophus::SE3 inc(incR, incT);
				VectorXd current_xi = inc.log().cast<double>();

				ATA.block(0, 0, 3, 3) += weightLinearVel * Matrix3d::Identity();
				ATA.block(3, 3, 3, 3) += weightRotationVel * Matrix3d::Identity();
				ATb.segment(0, 3) -= weightLinearVel *current_xi.segment(0, 3);
				ATb.segment(3, 3) -= weightRotationVel *current_xi.segment(3, 3);
#endif

				LLT<MatrixXd> lltOfA = ATA.llt();
				ComputationInfo info = lltOfA.info();

				//printf("info:%d ith:%d actualNum:%d cnt:%d level:%d\n", info, ith, actualNum, cnt, level);
				//cout << "ATA" << endl << ATA << endl;
				//cout << "-ATb" << endl << ATb << endl;
				if (info == Success)
				{
					VectorXd x = lltOfA.solve(ATb);

#ifdef DEBUG_DENSETRACKING
					MatrixXd L = lltOfA.matrixL();
					cout << "currntError: " << currentError/actualNum << endl ;
					// cout << "lltofATA.L() " << ith << ":\n" <<  L << endl ;
					// cout << "ATb " << ith << ":\n" << ATb << endl ;
					// cout << "dx " << ith << ":\n" << x.transpose() << endl;
#endif
					//printf("x.norm()=%f\n", x.norm() );
					Vector3d w, v;
					v(0) = -x(0);
					v(1) = -x(1);
					v(2) = -x(2);
					w(0) = -x(3);
					w(1) = -x(4);
					w(2) = -x(5);
					updateR_T(tmpR, tmpT, v, w, incR, incT);
					last_delta_v = v;
					last_delta_w = w;

					//#ifdef DEBUG_DENSETRACKING
					//					printf("ith=%d num=%d norm=%f error=%f\n", ith, actualNum, x.norm(), currentError);
					//#endif
					//if ( maxAbsValueOfVector(x) < updateThreshold){
					if (fabs(x(0)) < minimumUpdateTranslationThreshold
						&& fabs(x(1)) < minimumUpdateTranslationThreshold
						&& fabs(x(2)) < minimumUpdateTranslationThreshold
						&& fabs(x(3)) < minimumUpdateAngularThreshold
						&& fabs(x(4)) < minimumUpdateAngularThreshold
						&& fabs(x(5)) < minimumUpdateAngularThreshold
						){
						break;
					}
				}
				else {
					puts("can not solve Ax = b");
					break;
				}
			}//end of interation
		}//end of pyramid level
		R = tmpR;
		T = tmpT;
	}

	void popOldestState()
	{
		numOfState--;

		int nextStateID = head + 1;
		if (nextStateID >= slidingWindowSize){
			nextStateID -= slidingWindowSize;
		}
		std::list<SUPERPIXEL_IN_3D_SPACE>::iterator insertPos = superpixelList.begin();
		while (insertPos != superpixelList.end())
		{
			if (insertPos->stateID != nextStateID){
				insertPos++;
			}
			else {
				break;
			}
		}

		std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
		while (iter != insertPos)
		{
			bool flag = true;
			//if (iter->reprojectList.size() <= 2){
			//	iter = superpixelList.erase(iter);
			//	continue;
			//}

			//reproject points within a local plane
			Vector3d p0 = iter->pk[0] * iter->lambda[0];
			Vector3d p1 = iter->pk[1] * iter->lambda[1];
			Vector3d p2 = iter->pk[2] * iter->lambda[2];
			Vector3d p02 = p2 - p0;
			Vector3d p01 = p1 - p0;
			MatrixXd normalT = p02.cross(p01).transpose();
			MatrixXd numerotor = normalT*p0;
			for (int level = 0; level < maxPyramidLevelBA; level++)
			{
				int n = height >> level;
				int m = width >> level;
				SUPERPIXEL_INFO& currentSuperpixel = iter->superpixelsInPyramid[level];
				int sz = currentSuperpixel.listOfU_.size();
				int totalNum = 0;
				for (int i = 0; i < sz; i++)
				{
					Vector3d u_e;
					u_e << currentSuperpixel.listOfV_[i], currentSuperpixel.listOfU_[i], 1.0;

					MatrixXd denorminator = normalT*u_e;
					double lambda = numerotor(0, 0) / denorminator(0, 0);
					Vector3d pi = lambda * u_e;
					Vector3d pj = states[nextStateID].R_k0.transpose()*(states[head].T_k0 - states[nextStateID].T_k0 + states[head].R_k0*pi);

					int u2 = int(pj(1)*para->fy[level] / pj(2) + para->cy[level] + 0.5);
					int v2 = int(pj(0)*para->fx[level] / pj(2) + para->cx[level] + 0.5);

					if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
						continue;
					}
					totalNum++;
					currentSuperpixel.listOfV_[i] = pj(0) / pj(2);
					currentSuperpixel.listOfU_[i] = pj(1) / pj(2);
				}

				if (level == 0 && totalNum < minOptNum)
				{
					flag = false;
					printf("[Pop out] StateID:%d Num:%d\n", iter->stateID, totalNum);
					iter = superpixelList.erase(iter);
					break;
				}
			}
			if (flag == false){
				continue;
			}

			//reproject support points
			Vector3d pkReproject[3];
			for (int i = 0; i < 3; i++)
			{
				int currentStateID = iter->stateID;
				pkReproject[i] = states[nextStateID].R_k0.transpose()*(states[currentStateID].T_k0 - states[nextStateID].T_k0 + states[currentStateID].R_k0*
					iter->lambda[i] * iter->pk[i]);

				iter->lambda[i] = pkReproject[i](2);
				//if (iter->lambda[i] < zeroThreshold){
				//	flag = false;
				//	break;
				//}
				iter->pk[i] = pkReproject[i] / iter->lambda[i];
			}
			//if (flag == false){
			//	iter = superpixelList.erase(iter);
			//	continue;
			//}

			//change stateID
			iter->stateID = nextStateID;

			////reduce reproject list
			//iter->reprojectList.pop_front();

			iter++;
		}

		//pop the oldest state
		head++;
		if (head >= slidingWindowSize){
			head -= slidingWindowSize;
		}
	}

	inline void insertMatrixToSparseMatrix(SparseMatrix<double>& to, const MatrixXd& from, int y, int x, int n, int m)
	{
		for (int i = 0; i < n; i++){
			for (int j = 0; j < m; j++){
				to.insert(y + i, x + j) = from(i, j);
			}
		}
	}

	void PhotometricBA()
	{
		vector<Matrix3d>R(slidingWindowSize);
		vector<Vector3d>T(slidingWindowSize);
		for (int i = 0; i < slidingWindowSize; i++){
			R[i] = states[i].R_k0;
			T[i] = states[i].T_k0;
		}
		//SparseMatrix<double> priorMatrix(sizeOfH, sizeOfH);
		//insertMatrixToSparseMatrix(priorMatrix, 100000000.0*MatrixXd::Identity(6, 6), 6*head, 6*head, 6, 6);
		//VectorXd priorVector = VectorXd::Zero(sizeOfH);


		//Precheck the bundle adjustment reproject list
		int level = maxPyramidLevelBA - 1;
		int n = height >> level;
		int m = width >> level;
		int readyNum = 0;
		printf("before precheck SP_NUM:%d\n", (int)superpixelList.size());
		for (std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
			iter != superpixelList.end();)
		{
			if (iter->valid == false){
				iter = superpixelList.erase(iter);
				continue;
			}
			iter->ready = false;

			int currentStateID = iter->stateID;
			const SUPERPIXEL_INFO& currentSuperpixel = iter->superpixelsInPyramid[level];
			int sz = currentSuperpixel.listOfU_.size();
			Vector3d p0 = iter->pk[0] * iter->lambda[0];
			Vector3d p1 = iter->pk[1] * iter->lambda[1];
			Vector3d p2 = iter->pk[2] * iter->lambda[2];
			Vector3d p02 = p2 - p0;
			Vector3d p01 = p1 - p0;
			MatrixXd normalT = p02.cross(p01).transpose();
			MatrixXd numerotor = normalT*p0;

			//already satisfy the dispartity citeria and valid pixel citeria
			if (iter->reprojectList.size() > 0)
			{
				iter->ready = true;
				readyNum++;

				//check the valid pixel number within a superpixel
				MatrixXd u_e_List(3, sz);
				MatrixXd piList(3, sz);
				MatrixXd denorminatorList;
				for (int i = 0; i < sz; i++)//for every pixel within a superpixel
				{
					u_e_List(0, i) = currentSuperpixel.listOfV_[i];
					u_e_List(1, i) = currentSuperpixel.listOfU_[i];
					u_e_List(2, i) = 1.0;
				}
				denorminatorList = normalT * u_e_List;
				for (int i = 0; i < sz; i++)//for every pixel within a superpixel
				{
					double lambda = numerotor(0, 0) / denorminatorList(0, i);
					piList.block(0, i, 3, 1) = lambda * u_e_List.block(0, i, 3, 1);
				}

				int linkStateID = tail;
				unsigned char *nextIntensity = states[linkStateID].intensity[level];
				Vector3d deltaT = R[linkStateID].transpose()*(T[currentStateID] - T[linkStateID]);
				MatrixXd pjList = R[linkStateID].transpose()*R[currentStateID] * piList;

				int validNum = 0;
				for (int i = 0; i < sz; i++)//for every pixel within a superpixel
				{
					Vector3d pj = pjList.block(0, i, 3, 1) + deltaT;
					if (pj(2) < zeroThreshold){
						continue;
					}

					int u2 = int(pj(1)*para->fy[level] / pj(2) + para->cy[level] + 0.5);
					int v2 = int(pj(0)*para->fx[level] / pj(2) + para->cx[level] + 0.5);

					if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
						continue;
					}
					//if ( level == 0 && SQ(nextGradientX[INDEX(u2, v2, n, m)]) + SQ(nextGradientY[INDEX(u2, v2, n, m)]) < graidientThreshold){
					//	continue;
					//}
					double w = 1.0;
					double r = nextIntensity[INDEX(u2, v2, n, m)] - currentSuperpixel.intensity[i];
					double r_fabs = fabs(r);
					if (r_fabs > huberKernelThreshold){
						w = huberKernelThreshold / r_fabs;
					}
					if (w > validPixelThreshold){
						validNum++;
					}
				}
				double currentValidPixelPercentage = double(validNum) / sz;
				if (currentValidPixelPercentage > validPixelPercentageThreshold){
					iter->reprojectList.push_back(linkStateID);
				}
				else
				{
					//iter = superpixelList.erase(iter);
					//puts("[Pop Out in Data association!]");
				}

				iter++;
				continue;
			}

			//check and see if the lastest state satisfy the dispartity citeria and valid pixel citeria 

			//1. check the disparity of support points
			bool flag = true;
			int linkStateID = tail;
			for (int i = 0; i < 3; i++)
			{
				//Vector3d reprojectNormalized_p = R[linkStateID].transpose()*(T[currentStateID] - T[linkStateID] + R[currentStateID] * iter->pk[i]);
				Vector3d reprojectNormalized_p = R[linkStateID].transpose()*(R[currentStateID] * iter->pk[i]);
				reprojectNormalized_p /= reprojectNormalized_p(2);
				//printf("normalized parallex: %f\n", (reprojectNormalized_p - iter->pk[i]).norm());
				if ((reprojectNormalized_p - iter->pk[i]).norm() < normalizedParallaxThreshold){
					flag = false;
				}
			}
			if (flag == false){
				iter->ready = false;
				iter++;
				continue;
			}

			//2. check the valid pixel number within a superpixel
			MatrixXd u_e_List(3, sz);
			MatrixXd piList(3, sz);
			MatrixXd denorminatorList;
			for (int i = 0; i < sz; i++)//for every pixel within a superpixel
			{
				u_e_List(0, i) = currentSuperpixel.listOfV_[i];
				u_e_List(1, i) = currentSuperpixel.listOfU_[i];
				u_e_List(2, i) = 1.0;
				//u_e_List(0, i) = currentSuperpixel.listOfV_[i];
				//u_e_List(1, i) = currentSuperpixel.listOfU_[i];
				//u_e_List(2, i) = 1.0;
				//Vector3d u_e;
				//u_e << currentSuperpixel.listOfV_[i], currentSuperpixel.listOfU_[i], 1.0;
				//MatrixXd denorminator = normalT*u_e;
				//double lambda = numerotor(0, 0) / denorminator(0, 0);
				//Vector3d pi = lambda * u_e;
			}
			denorminatorList = normalT * u_e_List;
			for (int i = 0; i < sz; i++)//for every pixel within a superpixel
			{
				double lambda = numerotor(0, 0) / denorminatorList(0, i);
				piList.block(0, i, 3, 1) = lambda * u_e_List.block(0, i, 3, 1);
			}

			unsigned char *nextIntensity = states[linkStateID].intensity[level];
			Vector3d deltaT = R[linkStateID].transpose()*(T[currentStateID] - T[linkStateID]);
			MatrixXd pjList = R[linkStateID].transpose()*R[currentStateID] * piList;

			int validNum = 0;
			for (int i = 0; i < sz; i++)//for every pixel within a superpixel
			{
				Vector3d pj = pjList.block(0, i, 3, 1) + deltaT;
				if (pj(2) < zeroThreshold){
					continue;
				}

				int u2 = int(pj(1)*para->fy[level] / pj(2) + para->cy[level] + 0.5);
				int v2 = int(pj(0)*para->fx[level] / pj(2) + para->cx[level] + 0.5);

				if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
					continue;
				}
				//if ( level == 0 && SQ(nextGradientX[INDEX(u2, v2, n, m)]) + SQ(nextGradientY[INDEX(u2, v2, n, m)]) < graidientThreshold){
				//	continue;
				//}
				double w = 1.0;
				double r = nextIntensity[INDEX(u2, v2, n, m)] - currentSuperpixel.intensity[i];
				double r_fabs = fabs(r);
				if (r_fabs > huberKernelThreshold){
					w = huberKernelThreshold / r_fabs;
				}
				if (w > validPixelThreshold){
					validNum++;
				}
			}
			double currentValidPixelPercentage = double(validNum) / sz;
			if (currentValidPixelPercentage > validPixelPercentageThreshold)
			{
				iter->ready = true;
				readyNum++;

				int j = (tail - currentStateID);
				if ( j <= 0){
					j += slidingWindowSize;
				}

				for (int k = j; k >= 1; k--)
				{
					int tmpStateID = currentStateID + k;
					if (tmpStateID >= slidingWindowSize){
						tmpStateID -= slidingWindowSize;
					}
					iter->reprojectList.push_back(tmpStateID);
				}
			}
			else
			{
				//iter = superpixelList.erase(iter);
				//puts("[Pop Out in Data association!]");
			}
			iter++;
		}
		printf("after precheck ready SP_NUM:%d\n", readyNum);

		if (readyNum < 1){
			puts("No ready Superpixels! Exit from BA!");
			return;
		}

		//Begin Pyramid Bundle Adjustment
		for (level = maxPyramidLevelBA - 1; level >= 0; level--)
		{
			int n = height >> level;
			int m = width >> level;
			for (int iterNum = 0; iterNum < maxIterationBA; iterNum++)
			{
				//pre check valid superpixel
				int numOfValidSuperpixelsInSlidingWindow = 0;
				double minDist = DBL_MAX;
				int fixedIndexSP = -1;
				for (std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
					iter != superpixelList.end();)
				{
					if (iter->valid == false){
						iter = superpixelList.erase(iter);
						continue;
					}
					else if (iter->ready == false){
						iter++;
						continue;
					}
					else
					{
						if (iter->stateID == head)//only head frame should be considered
						{
							double tmpDist = (iter->lambda[0] + iter->lambda[1] + iter->lambda[2]) / 3.0;
							if (tmpDist < minDist){
								minDist = tmpDist;
								fixedIndexSP = numOfValidSuperpixelsInSlidingWindow;
							}
						}
						numOfValidSuperpixelsInSlidingWindow++;
						iter++;
					}
				}

#ifdef DEBUG_BA
				printf("numOfValidSuperpixels:%d\n", numOfValidSuperpixelsInSlidingWindow);
				RNG rng(666);
#endif
				//Begin Bundle Adjustment
				int sizeOfH = 6 * numOfState + 3 * numOfValidSuperpixelsInSlidingWindow;
				MatrixXd HTH = MatrixXd::Zero(sizeOfH, sizeOfH);
				VectorXd HTb = VectorXd::Zero(sizeOfH);

				int SuperpixelID = 0;
				double totalError = 0;
				for (std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
					iter != superpixelList.end(); iter++)
				{
					if (iter->ready == false){
						continue;
					}
					//printf("SuperpixelID=%d reProjectListSz=%d\n", SuperpixelID, iter->reprojectList.size() );

					int currentStateID = iter->stateID;
					const SUPERPIXEL_INFO& currentSuperpixel = iter->superpixelsInPyramid[level];
					//					unsigned char* pIntensity = states[currentStateID].intensity[level];
					int sz = currentSuperpixel.listOfU_.size();

#ifdef DEBUG_BA
					printf("SPID=%d\n", SuperpixelID );

					int colorR = rng.uniform(0, 255);
					int colorG = rng.uniform(0, 255);
					int colorB = rng.uniform(0, 255);

					Mat currentImage(n, m, CV_8UC3);
					Mat weightImage(n, m, CV_32F);
					for (int iDebug = 0; iDebug < n; iDebug++)
					{
						for (int jDebug = 0; jDebug < m; jDebug++)
						{
							currentImage.at<cv::Vec3b>(iDebug, jDebug)[0] = pIntensity[INDEX(iDebug, jDebug, n, m)];
							currentImage.at<cv::Vec3b>(iDebug, jDebug)[1] = pIntensity[INDEX(iDebug, jDebug, n, m)];
							currentImage.at<cv::Vec3b>(iDebug, jDebug)[2] = pIntensity[INDEX(iDebug, jDebug, n, m)];
							weightImage.at<float>(iDebug, jDebug) = 0;
						}
					}

					int reProjectListSz = iter->reprojectList.size();
					printf("reProjectListSz=%d\n", reProjectListSz);
					vector<Mat> reprojectImageSet(reProjectListSz );
					for (int j = 0; j < reProjectListSz; j++)//reporject to frame j
					{
						int linkStateID = iter->reprojectList[j];
						//printf("%d\n", linkStateID);
						unsigned char *nextIntensity = states[linkStateID].intensity[level];
						double* nextGradientX = states[linkStateID].gradientX[level];
						double* nextGradientY = states[linkStateID].gradientY[level];
						Mat nextImage(n, m, CV_8UC3);
						for (int iDebug = 0; iDebug < n; iDebug++)
						{
							for (int jDebug = 0; jDebug < m; jDebug++)
							{
								nextImage.at<cv::Vec3b>(iDebug, jDebug)[0] = nextIntensity[INDEX(iDebug, jDebug, n, m)];
								nextImage.at<cv::Vec3b>(iDebug, jDebug)[1] = nextIntensity[INDEX(iDebug, jDebug, n, m)];
								nextImage.at<cv::Vec3b>(iDebug, jDebug)[2] = nextIntensity[INDEX(iDebug, jDebug, n, m)];
							}
						}
						reprojectImageSet[j] = nextImage;
					}
#endif

					Vector3d p0 = iter->pk[0] * iter->lambda[0];
					Vector3d p1 = iter->pk[1] * iter->lambda[1];
					Vector3d p2 = iter->pk[2] * iter->lambda[2];
					Vector3d p02 = p2 - p0;
					Vector3d p01 = p1 - p0;
					MatrixXd normalT = p02.cross(p01).transpose();
					MatrixXd numerotor = normalT*p0;

					double up = iter->lambda[0] * iter->lambda[1] * iter->lambda[2] * iter->u1xu2_T_u0(0, 0)
						+ SQ(iter->lambda[0])*  iter->lambda[2] * iter->u0xu2_T_u0(0, 0)
						+ SQ(iter->lambda[0])* iter->lambda[1] * iter->u0xu1_T_u0(0, 0);

					MatrixXd u_e_List(3, sz);
					MatrixXd piList(3, sz);
					VectorXd down(sz);

					for (int i = 0; i < sz; i++)
					{
						u_e_List(0, i) = currentSuperpixel.listOfV_[i];
						u_e_List(1, i) = currentSuperpixel.listOfU_[i];
						u_e_List(2, i) = 1.0;
					}
					MatrixXd denorminatorList_12 = iter->u1xu2_T*u_e_List;
					MatrixXd denorminatorList_02 = iter->u0xu2_T*u_e_List;
					MatrixXd denorminatorList_01 = iter->u0xu1_T*u_e_List;

					for (int i = 0; i < sz; i++)//for every pixel within a superpixel
					{
						down(i) = iter->lambda[1] * iter->lambda[2] * denorminatorList_12(0, i)
							+ iter->lambda[0] * iter->lambda[2] * denorminatorList_02(0, i)
							+ iter->lambda[0] * iter->lambda[1] * denorminatorList_01(0, i);

						piList.block(0, i, 3, 1) = up / down(i) * u_e_List.block(0, i, 3, 1);
					}

#ifdef DEBUG_BA
					double proportion = 0.3;
					int u = int(pi(1)*para->fy[level] / pi(2) + para->cy[level] + 0.5);
					int v = int(pi(0)*para->fx[level] / pi(2) + para->cx[level] + 0.5);

					if ( !(u < 0 || u >= n || v < 0 || v >= m) ){
						currentImage.at<Vec3b>(u, v)[0] = currentImage.at<Vec3b>(u, v)[0] * proportion + colorR * (1 - proportion);
						currentImage.at<Vec3b>(u, v)[1] = currentImage.at<Vec3b>(u, v)[1] * proportion + colorG * (1 - proportion);
						currentImage.at<Vec3b>(u, v)[2] = currentImage.at<Vec3b>(u, v)[2] * proportion + colorB * (1 - proportion);
					}
#endif
					int actualNum = 0;
					int reProjectListSz = iter->reprojectList.size();
					//						omp_set_num_threads(ompNumThreads);
					//#pragma omp parallel for
					for (int j = 0; j < reProjectListSz; j++)
					{
						int linkStateID = iter->reprojectList[j];
						unsigned char *nextIntensity = states[linkStateID].intensity[level];
						double* nextGradientX = states[linkStateID].gradientX[level];
						double* nextGradientY = states[linkStateID].gradientY[level];

						MatrixXd Rji = R[linkStateID].transpose()*R[currentStateID];
						Vector3d deltaT = R[linkStateID].transpose()*(T[currentStateID] - T[linkStateID]);
						MatrixXd pjList = Rji * piList;
						for (int i = 0; i < sz; i++)
						{
							Vector3d pj = pjList.block(0, i, 3, 1) + deltaT;

							if (pj(2) < zeroThreshold){
								continue;
							}

							int u2 = int(pj(1)*para->fy[level] / pj(2) + para->cy[level] + 0.5);
							int v2 = int(pj(0)*para->fx[level] / pj(2) + para->cx[level] + 0.5);

							//TODO: Intepolation
							if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
								continue;
							}

							//if (SQ(nextGradientX[INDEX(u2, v2, n, m)]) + SQ(nextGradientY[INDEX(u2, v2, n, m)]) < graidientThreshold){
							//	continue;
							//}

#ifdef DEBUG_BA
							reprojectImageSet[j].at<Vec3b>(u2, v2)[0] = reprojectImageSet[j].at<Vec3b>(u2, v2)[0] * proportion + colorR * (1 - proportion);
							reprojectImageSet[j].at<Vec3b>(u2, v2)[1] = reprojectImageSet[j].at<Vec3b>(u2, v2)[1] * proportion + colorG * (1 - proportion);
							reprojectImageSet[j].at<Vec3b>(u2, v2)[2] = reprojectImageSet[j].at<Vec3b>(u2, v2)[2] * proportion + colorB * (1 - proportion);
#endif

							actualNum++;

							double w = 1.0;
							double r = nextIntensity[INDEX(u2, v2, n, m)] - currentSuperpixel.intensity[i];
							double r_fabs = fabs(r);
							totalError += r_fabs;

#ifdef WEIGHTEDCOST
							if (r_fabs > huberKernelThreshold){
								w = huberKernelThreshold / (r_fabs);
							}
#endif

							MatrixXd oneByTwo(1, 2);
							oneByTwo << nextGradientX[INDEX(u2, v2, n, m)], nextGradientY[INDEX(u2, v2, n, m)];

							MatrixXd twoByThree(2, 3);
							twoByThree << para->fx[level] / pj(2), 0, -pj(0)*para->fx[level] / SQ(pj(2)), 0, para->fy[level] / pj(2), -pj(1)*para->fy[level] / SQ(pj(2));

							MatrixXd threeByFifteen(3, 15);
							//1. xi
							threeByFifteen.block(0, 0, 3, 3) = R[linkStateID].transpose();
							threeByFifteen.block(0, 3, 3, 3) = -R[linkStateID].transpose()*R[currentStateID] * vectorToSkewMatrix( piList.block(0, i, 3, 1) );

							//2. xj
							threeByFifteen.block(0, 6, 3, 3) = -R[linkStateID].transpose();
							threeByFifteen.block(0, 9, 3, 3) = vectorToSkewMatrix( pj );

							//3. yk
							double tmpNumerator;
							Vector3d u_e_j = 1.0 / SQ(down(i)) * (Rji *u_e_List.block(0, i, 3, 1));
							tmpNumerator = (
								iter->lambda[1] * iter->lambda[2] * iter->u1xu2_T_u0(0, 0)
								+ 2.0*iter->lambda[0] * iter->lambda[2] * iter->u0xu2_T_u0(0, 0)
								+ 2.0*iter->lambda[0] * iter->lambda[1] * iter->u0xu1_T_u0(0, 0)
								) * down(i)
								- ( iter->lambda[1] * (iter->u0xu1_T*u_e_List.block(0, i, 3, 1))(0, 0)
								 - iter->lambda[2] * (iter->u0xu2_T*u_e_List.block(0, i, 3, 1) )(0, 0)
								)*up;
							threeByFifteen.block(0, 12, 3, 1) = tmpNumerator * u_e_j ;

							tmpNumerator = (
								iter->lambda[0] * iter->lambda[2] * iter->u1xu2_T_u0(0, 0)
								+ SQ(iter->lambda[0]) * iter->u0xu2_T_u0(0, 0)
								) * down(i)
								- (
								iter->lambda[2] * (iter->u1xu2_T*u_e_List.block(0, i, 3, 1))(0, 0)
								+ iter->lambda[0] * (iter->u0xu2_T*u_e_List.block(0, i, 3, 1))(0, 0)
								)*up;
							threeByFifteen.block(0, 13, 3, 1) = tmpNumerator  * u_e_j ;

							tmpNumerator = (
								iter->lambda[0] * iter->lambda[1] * iter->u1xu2_T_u0(0, 0)
								- SQ(iter->lambda[0])* iter->u0xu2_T_u0(0, 0)
								) * down(i)
								- (
								iter->lambda[1] * (iter->u1xu2_T*u_e_List.block(0, i, 3, 1))(0, 0)
								- iter->lambda[0] * (iter->u0xu2_T*u_e_List.block(0, i, 3, 1))(0, 0)
								)*up;
							threeByFifteen.block(0, 14, 3, 1) = tmpNumerator  * u_e_j ;

							VectorXd oneByFifteen = oneByTwo*twoByThree*threeByFifteen;
							VectorXd oneByFifteenT = oneByFifteen.transpose();
							MatrixXd fifteenByFifteen = oneByFifteenT*oneByFifteen;
						
							int updatePos = sizeOfH - 3 * (SuperpixelID + 1);
							
#pragma omp critical
							{
								//1. related to xi
								HTH.block(currentStateID * 6, currentStateID * 6, 6, 6) += w* fifteenByFifteen.block(0, 0, 6, 6);
								HTH.block(currentStateID * 6, linkStateID * 6, 6, 6) += w* fifteenByFifteen.block(0, 6, 6, 6);
								HTH.block(currentStateID * 6, updatePos, 6, 3) += w* fifteenByFifteen.block(0, 12, 6, 3);

								//2. related to xj
								HTH.block(linkStateID * 6, currentStateID * 6, 6, 6) += w* fifteenByFifteen.block(6, 0, 6, 6);
								HTH.block(linkStateID * 6, linkStateID * 6, 6, 6) += w* fifteenByFifteen.block(6, 6, 6, 6);
								HTH.block(linkStateID * 6, updatePos, 6, 3) += w* fifteenByFifteen.block(6, 12, 6, 3);

								//3. related to yi
								HTH.block(updatePos, currentStateID * 6, 3, 6) += w* fifteenByFifteen.block(12, 0, 3, 6);
								HTH.block(updatePos, linkStateID * 6, 3, 6) += w* fifteenByFifteen.block(12, 6, 3, 6);
								HTH.block(updatePos, updatePos, 3, 3) += w* fifteenByFifteen.block(12, 12, 3, 3);

								//4. HTb
								HTb.segment(currentStateID * 6, 6) -= w*r* oneByFifteenT.segment(0, 6);
								HTb.segment(linkStateID * 6, 6) -= w*r* oneByFifteenT.segment(6, 6);
								HTb.segment(updatePos, 3) -= w*r* oneByFifteenT.segment(12, 3);
							}

						}//end of pixel within a superpixel
					}//end of reprojection

					//cout << SuperpixelID << endl;
					//cout << HTH << endl;
					if (SuperpixelID == fixedIndexSP)//prior for the first frame depth
					{
#ifdef DEBUG_BA
						printf("fixedIndexSP:%d\n", fixedIndexSP);
#endif
						int updatePos = sizeOfH - 3 * (SuperpixelID + 1);
						for (int i = 0; i < 3; i++){
							HTH(updatePos, updatePos) += 10000000000.0*10000000000.0;
						}
					}

#ifdef DEBUG_BA
					imshow("current", currentImage);
					char ttt[128];
					for (int j = 0; j < reProjectListSz; j++)
					{
						sprintf(ttt, "%d", j);
						imshow(ttt, reprojectImageSet[j]);
					}
					waitKey(0);
					//std::printf("level:%d iterNum:%d stateNum:%d SPNum:%d computeNum:%d\n", level, iterNum, numOfState, SuperpixelID, actualNum);
#endif

					SuperpixelID++;
				}//end of superpixel

				//prior for the oldest state
				int updatePos = head * 6;
				for (int i = 0; i < 6; i++)
				{
					HTH(updatePos, updatePos) += 10000000000.0*10000000000.0;
					updatePos++;
				}

				std::printf("level:%d iterNum:%d stateNum:%d SPNum:%d\n", level, iterNum, numOfState, numOfValidSuperpixelsInSlidingWindow);
				//cout << HTH << endl;
				//cout << HTb << endl;

				LLT<MatrixXd> lltHTH = HTH.llt();
				ComputationInfo info = lltHTH.info();
				//printf("actual num = %d total errors=%f\n", actualNum, totalError)
				if (info == Success)
				{
					VectorXd dx = lltHTH.solve(HTb);
					//cout << dx << endl;

					int updatePos;
					bool flag = true;

					//pre check
					updatePos = sizeOfH - 3;
					for (std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
						iter != superpixelList.end(); iter++)
					{
						if (iter->ready == false){
							continue;
						}
						if (dx(updatePos) < iter->lambda[0] * dxOptProportion){
							;
						}
						else{
							iter->valid = false;
							flag = false;
						}

						if (dx(updatePos + 1) < iter->lambda[1] * dxOptProportion){
							;
						}
						else{
							iter->valid = false;
							flag = false;
						}

						if (dx(updatePos + 2) < iter->lambda[2] * dxOptProportion){
							;
						}
						else{
							iter->valid = false;
							flag = false;
						}
						updatePos -= 3;
					}

					//pass the pre check
					if (flag == true)
					{
						//1. update state
						updatePos = 0;
						for (int i = 0; i < numOfState; i++)
						{
							T[i](0) += dx(updatePos);
							T[i](1) += dx(updatePos + 1);
							T[i](2) += dx(updatePos + 2);
							Vector3d theta;
							theta << dx(updatePos + 3), dx(updatePos + 4), dx(updatePos + 5);
							R[i] = R[i] * (Matrix3d::Identity() + vectorToSkewMatrix(theta));
							updatePos += 6;
						}

						//2. update superpixel
						updatePos = sizeOfH - 3;
						for (std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter = superpixelList.begin();
							iter != superpixelList.end(); iter++)
						{
							if (iter->ready == false){
								continue;
							}
							if (dx(updatePos) < iter->lambda[0] * dxOptProportion){
								iter->lambda[0] += dx(updatePos);
							}
							else{
								iter->valid = false;
								flag = false;
							}

							if (dx(updatePos + 1) < iter->lambda[1] * dxOptProportion){
								iter->lambda[1] += dx(updatePos + 1);
							}
							else{
								iter->valid = false;
								flag = false;
							}

							if (dx(updatePos + 2) < iter->lambda[2] * dxOptProportion){
								iter->lambda[2] += dx(updatePos + 2);
							}
							else{
								iter->valid = false;
								flag = false;
							}
							updatePos -= 3;
						}
					}
					else
					{
						iterNum--;
					}
				}
				else
				{
					puts("LLT error!!!");
					iterNum = maxIteration;
					//cout << HTH << endl;
				}
			}
		}

		//after all the iterations done
		for (int i = 0; i < slidingWindowSize; i++){
			states[i].T_k0 = T[i];
			states[i].R_k0 = R[i];
		}
	}

	void prepareDateForVisualization(vector<Vector3d>& pointcloud,
		vector<unsigned short>& R,
		vector<unsigned short>& G,
		vector<unsigned short>& B,
		vector<Vector3d>& ps,
		vector<Matrix3d>& Rs)
	{
		ps.clear();
		Rs.clear();
		for (int i = 0; i < numOfState; i++)
		{
			int currentState = head + i;
			if (currentState >= slidingWindowSize){
				currentState -= slidingWindowSize;
			}
			ps.push_back(states[currentState].T_k0);
			Rs.push_back(states[currentState].R_k0.transpose());
		}

		pointcloud.clear();
		R.clear();
		G.clear();
		B.clear();
		//		int n = height;
		//		int m = width;
		int level = 0;
		std::list<SUPERPIXEL_IN_3D_SPACE>::iterator iter;
		for (iter = superpixelList.begin(); iter != superpixelList.end(); iter++)
		{
			//			int currentStateID = iter->stateID;
			const SUPERPIXEL_INFO& currentSuperpixel = iter->superpixelsInPyramid[level];

			Vector3d p0 = iter->pk[0] * iter->lambda[0];
			Vector3d p1 = iter->pk[1] * iter->lambda[1];
			Vector3d p2 = iter->pk[2] * iter->lambda[2];
			Vector3d p02 = p2 - p0;
			Vector3d p01 = p1 - p0;
			MatrixXd normalT = p02.cross(p01).transpose();
			MatrixXd numerotor = normalT*p0;

			//unsigned char* pIntensity = states[currentStateID].intensity[level];
			int sz = currentSuperpixel.listOfU_.size();
			for (int i = 0; i < sz; i++)//for every pixel within a superpixel
			{
				//int v = currentSuperpixel.listOfV[i];
				//int u = currentSuperpixel.listOfU[i];
				//double v_ = (coordinateX[level][v] - para->cx) / para->fx;
				//double u_ = (coordinateY[level][u] - para->cy) / para->fy;
				Vector3d u_e;
				u_e << currentSuperpixel.listOfV_[i], currentSuperpixel.listOfU_[i], 1.0;

				MatrixXd denorminator = normalT*u_e;
				double lambda = numerotor(0, 0) / denorminator(0, 0);
				Vector3d pi = lambda * u_e;

				pointcloud.push_back(pi);
				R.push_back(currentSuperpixel.intensity[i]);
				G.push_back(currentSuperpixel.intensity[i]);
				B.push_back(currentSuperpixel.intensity[i]);
			}
		}
	}

};

#endif
