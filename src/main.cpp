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
CAMER_PARAMETERS cameraParameters;
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

void InitFIleList()
{
	char tmp[256];
	FILE *fp = NULL;

	strcpy(depthDataPath, filePath);
	strcat(depthDataPath, "depth\\");

	strcpy(rgbDataPath, filePath);
	strcat(rgbDataPath, "rgb\\");

	strcpy(rgbListPath, filePath);
	strcat(rgbListPath, "rgb.txt");

	strcpy(depthListPath, filePath);
	strcat(depthListPath, "depth.txt");

	strcpy(groundTruthDataPath, filePath);
	strcat(groundTruthDataPath, "groundtruth.txt");
	
	//read rgb image name list
	fp = fopen(rgbListPath, "r");
	if (fp == NULL){
		puts("rgbList Path error");
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
	fp = NULL;
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
	fp = NULL;
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

#ifdef SAVEVIDEO
VideoWriter output_dst("demo.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, Size(640, 480), true);
#endif

void showLabels(Mat rgbImage, int totalLabels, int labels[IMAGE_HEIGHT][IMAGE_WIDTH], bool valid[IMAGE_HEIGHT][IMAGE_WIDTH], Mat depthImage )
{
	RNG rng(123);
	int R[maxKMeansNum];
	int G[maxKMeansNum];
	int B[maxKMeansNum];

	printf("total num of clusters: %d\n", totalLabels);
	Mat showImage = rgbImage.clone();
	int n = rgbImage.rows;
	int m = rgbImage.cols;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (valid[i][j] == false){
				continue;
			}
			bool flag = false;
			for (int k = 0; k < 8; k++)
			{
				int ty = i + dy8[k];
				int tx = j + dx8[k];
				if (ty < 0 || ty >= n || tx < 0 || tx >= m){
					continue;
				}
				if (valid[ty][tx] == false){
					flag = true;
					break;
					continue;
				}
				if (labels[ty][tx] != labels[i][j]){
					flag = true;
					break;
				}
			}
			if (flag == true){
				showImage.at<cv::Vec3b>(i, j)[0] = 0;
				showImage.at<cv::Vec3b>(i, j)[1] = 0;
				showImage.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	cv::imshow("label image", showImage);

	//Mat showImage2(n, m, CV_8U );
	//for (int i = 0; i < n; i++){
	//	for (int j = 0; j < m; j++){
	//		showImage2.at<uchar>(i, j) = valid[i][j] * 255;
	//	}
	//}
	//imshow("debug2", showImage2);

	for (int i = 0; i < totalLabels; i++){
		R[i] = rng.uniform(0, 255);
		G[i] = rng.uniform(0, 255);
		B[i] = rng.uniform(0, 255);
	}

	Mat debugImage(n, m, CV_8UC3);
	for (int i = 0; i < n; i++ )
	{
		for (int j = 0; j < m; j++ )
		{
			if (valid[i][j] == false){
				debugImage.at<cv::Vec3b>(i, j)[0] = 0;
				debugImage.at<cv::Vec3b>(i, j)[1] = 0;
				debugImage.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else{
				int k = labels[i][j];
				debugImage.at<cv::Vec3b>(i, j)[0] = R[k];
				debugImage.at<cv::Vec3b>(i, j)[1] = G[k];
				debugImage.at<cv::Vec3b>(i, j)[2] = B[k];
			}
		}
	}
	cv::imshow("debug", debugImage);

	Mat combine(n * 2, m * 2, CV_8UC3);
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			combine.at<cv::Vec3b>(i, j) = rgbImage.at<cv::Vec3b>(i, j);
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			combine.at<cv::Vec3b>(i + n, j) = showImage.at<cv::Vec3b>(i, j);
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			combine.at<cv::Vec3b>(i + n, j + m) = debugImage.at<cv::Vec3b>(i, j);
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			combine.at<cv::Vec3b>(i, j + m)[0] = depthImage.at<ushort>(i, j)>>8;
			combine.at<cv::Vec3b>(i, j + m)[1] = depthImage.at<ushort>(i, j) >> 8;
			combine.at<cv::Vec3b>(i, j + m)[2] = depthImage.at<ushort>(i, j) >> 8;
		}
	}

	//imshow("combine", combine);
#ifdef SAVEVIDEO
	output_dst.write(combine);
	output_dst.write(combine);
	output_dst.write(combine);
	output_dst.write(combine);
#endif
}

//void downSamplingDepthImage(Mat&input, Mat&output)
//{
//	int n = input.rows;
//	int m = input.cols;
//	int nn = n / 2;
//	int mm = m / 2;
//	output.create(nn, mm, CV_16U );
//
//	unsigned short* pInput = (unsigned short*)input.data;
//	unsigned short* pOutput = (unsigned short*)output.data;
//
//	for (int i = 0; i < nn; i++ )
//	{
//		for (int j = 0; j < mm; j++)
//		{
//			int y[4] = {(i << 1), (i << 1), (i << 1) + 1, (i << 1) + 1};
//			int x[4] = { (j << 1), (j << 1)+1, (j << 1), (j << 1) + 1 };
//			int sum = 0;
//			for (int  k = 0; k < 4; k++)
//			{
//				unsigned short t1 = pInput[y[k] * m + x[k]];
//				sum += t1;
//			}
//			pOutput[i*mm + j] = (unsigned short)(sum >> 2);
//			//pOutput[i*mm + j] = pInput[y[0] * m + x[0]];
//		}
//	}
//}

inline unsigned long long absUnsignedLongLong(unsigned long long a, unsigned long long b){
	if (a > b) return a - b;
	else return b - a;
}

void init()
{
	InitFIleList();

	//TUM Freiburg 1 sequences
	//CAMER_PARAMETERS cameraParameters(517.3, 516.5, 318.6, 255.3);

	//TUM Freiburg 3 sequences
	double fx = 535.4;
	double fy = 539.2;
	double cx = 320.1;
	double cy = 247.6;

#ifdef DOWNSAMPLING
	fx /= 2.0;
	fy /= 2.0;
	cx = (cx + 0.5) / 2.0 - 0.5;
	cy = (cy + 0.5) / 2.0 - 0.5;
#endif

	cameraParameters.setParameters(fx, fy, cx, cy);
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

//testDataGenerator TDG;
Mat rgbImage;
Mat depthImage[maxPyramidLevel*bufferSize];
Mat grayImage[maxPyramidLevel*bufferSize];
int bufferHead = 0;
STATE tmpState;
STATE* lastFrame;

void frameToFrameDenseTracking(Matrix3d& R_k_c, Vector3d& T_k_c)
{
	Matrix3d nextR = Matrix3d::Identity();
	Vector3d nextT = Vector3d::Zero();
	slidingWindows.denseTrackingWithoutSuperpixel(lastFrame, grayImage, bufferHead, nextR, nextT);

	T_k_c = nextR * T_k_c + nextT;
	R_k_c = nextR * R_k_c;
}

void keyframeToFrameDenseTracking(int bufferHead, Matrix3d& R_k_c, Vector3d& T_k_c)
{
	STATE* keyframe = &slidingWindows.states[slidingWindows.tail];
	slidingWindows.denseTrackingWithoutSuperpixel(keyframe, grayImage, bufferHead, R_k_c, T_k_c);
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
	map<unsigned long long, int>::iterator iter;

	Matrix3d R_k_c;//R_k^(k+1)
	Matrix3d R_c_0;
	Vector3d T_k_c;//T_k^(k+1)
	Vector3d T_c_0;

	ofstream fileOutput("result.txt");
	
	init();
	bufferHead = 0;
	for ( int i = 1; i < numImage; i += 1 )
	{
		printf("id : %d\n", i);
		char tmp[256];

		//read rgb image
		strcpy(tmp, rgbDataPath);
		strcat(tmp, rgbFileNameList[i]);
		rgbImage = imread(tmp, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImage, grayImage[0], CV_BGR2GRAY);
		//cv::undistort(rgbImage, rgbRectImage, cameraParameters.cameraMatrix, cameraParameters.distCoeffs);

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
		if (i + 1 < numImage && absUnsignedLongLong(depthImageTimeStamp[i + 1], rgbImageTimeStamp[i]) < minS){
			k = i + 1;
		}
		strcat(tmp, depthFileNameList[k]);
		depthImage[0] = imread(tmp, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

		depthImage[0].convertTo(depthImage[0], CV_32F);
		depthImage[0] /= depthFactor;


		//imshow("rgbImge", rgbImage ) ;
		//imshow("rgbRectImage", rgbRectImage);
		//waitKey(0);


#ifdef DOWNSAMPLING
		pyrDownMeanSmooth<uchar>(grayImage[0], grayImage[0]);
		pyrDownMedianSmooth<float>(depthImage[0], depthImage[0]);
#endif

		for (int kk = 1; kk < maxPyramidLevel; kk++){
			pyrDownMeanSmooth<uchar>(grayImage[kk - 1], grayImage[kk]);
			pyrDownMedianSmooth<float>(depthImage[kk - 1], depthImage[kk]);
		}


		//for (int x = 20; x < 30; x++){
		//	cout << depthImage.at<float>(307, x) << endl;
		//}
		//ushort* pDepth = (ushort*)depthImage.data;
		//for (int x = 0; x < 20; x++){
		//	cout << pDepth[x] << " " << (int)depthImage.at<ushort>(0, x) << endl;
		//}
		//downSamplingDepthImage(depthImage, downDepthImage );
		//for (int x = 0; x < 20; x++){
		//	cout << depth[307*640+x] <<  " " <<  (int)depthImage.at<ushort>(307, x) << endl;
		//}


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
			firstFrameTtoVICON << groundTruth[timeID].tx, groundTruth[timeID].ty, groundTruth[timeID].tz;

			//cout << firstFrameTtoVICON << endl;
			slidingWindows.insertKeyFrame(grayImage, depthImage, bufferHead, Matrix3d::Identity(), Vector3d::Zero());

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
		keyframeToFrameDenseTracking(bufferHead, R_k_c, T_k_c);
#endif

		//t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
		//printf("cal time: %f\n", t);

		R_c_0 = slidingWindows.states[slidingWindows.tail].R_k0*R_k_c.transpose();
		T_c_0 = R_c_0*(
			R_k_c*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - T_k_c);

		////update current  calculation
		//currentR = slidingWindows.states[slidingWindows.tail].R_k0*nextR.transpose();
		//currentT = currentR*(
		//	nextR*(slidingWindows.states[slidingWindows.tail].R_k0.transpose())*slidingWindows.states[slidingWindows.tail].T_k0 - nextT);
		//fileOutput << (firstFrameRtoVICON*currentT).transpose() << endl;
		//fileOutput << groundTruth[timeID].tx - firstFrameTtoVICON(0) << " "
		//	<< groundTruth[timeID].ty - firstFrameTtoVICON(1) << " " << groundTruth[timeID].tz - firstFrameTtoVICON(2) << endl;

		//insert key frame
		
		//MatrixXd rotatedAngles = currentR.eulerAngles(0, 1, 2);
		/*if (
			fabs(rotatedAngles(0,0)) > angularThreshold
			|| fabs(rotatedAngles(0, 1)) > angularThreshold
			|| fabs(rotatedAngles(0, 2)) > angularThreshold
			|| currentT.norm() > translationThreshold )*/
		if ((i % 10) == 1)
		{
			//double t = (double)cvGetTickCount();

			slidingWindows.insertKeyFrame(grayImage, depthImage, bufferHead, R_c_0, T_c_0 );

			cout << "estimate position[before BA]:\n" 
				<< slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;


			//slidingWindows.PhotometricBA();

			//t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
			//printf("BA cal time: %f\n", t);

			//slidingWindows.planeDection();

			R_k_c = Matrix3d::Identity();
			T_k_c = Vector3d::Zero();

			lastFrame = &slidingWindows.states[slidingWindows.tail];

			//cout << "estimate position[after BA]:\n" << (firstFrameRtoVICON.transpose()*slidingWindows.states[slidingWindows.tail].T_k0 + firstFrameTtoVICON).transpose() << endl;
			//cout << "ground truth position:\n" << groundTruth[timeID].tx << " " << groundTruth[timeID].ty << " " << groundTruth[timeID].tz << endl;
			//fileOutput << (firstFrameRtoVICON.transpose()*slidingWindows.states[slidingWindows.tail].T_k0 + firstFrameTtoVICON).transpose() << endl;
			//fileOutput << groundTruth[timeID].tx << " " << groundTruth[timeID].ty << " " << groundTruth[timeID].tz << endl;

			cout << "estimate position[after BA]:\n" 
				<< slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;

			Vector3d groundTruthT;
			groundTruthT << groundTruth[timeID].tx, groundTruth[timeID].ty, groundTruth[timeID].tz;
			groundTruthT = firstFrameRtoVICON.transpose()*(groundTruthT - firstFrameTtoVICON);

			cout << "ground truth position:\n" 
				<< groundTruthT.transpose() << endl;

			fileOutput << slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;
			fileOutput << groundTruthT.transpose() << endl;

			Quaterniond q;
			Matrix3d truthR;
			q.x() = groundTruth[timeID].qx;
			q.y() = groundTruth[timeID].qy;
			q.z() = groundTruth[timeID].qz;
			q.w() = groundTruth[timeID].qw;
			truthR = q.toRotationMatrix();

			double estimateEularAngels[3];
			double groundEularAngels[3];

			RtoEulerAngles(firstFrameRtoVICON*slidingWindows.states[slidingWindows.tail].R_k0, estimateEularAngels);
			RtoEulerAngles(truthR, groundEularAngels);
			//RtoEulerAngles(slidingWindows.states[slidingWindows.tail].R_k0, estimateEularAngels);
			//RtoEulerAngles(firstFrameRtoVICON.transpose()*truthR, groundEularAngels);
			
			cout << "estimate angels:\n" << estimateEularAngels[0] << " " << estimateEularAngels[1] << " " << estimateEularAngels[2] << endl;
			cout << "ground truth angels:\n" << groundEularAngels[0] << " " << groundEularAngels[1] << " " << groundEularAngels[2] << endl;

			fileOutput << estimateEularAngels[0] << " " << estimateEularAngels[1] << " " << estimateEularAngels[2]  << endl;
			fileOutput << groundEularAngels[0] << " " << groundEularAngels[1] << " " << groundEularAngels[2] << endl;

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
	fileOutput.close();

	return 0;
}
