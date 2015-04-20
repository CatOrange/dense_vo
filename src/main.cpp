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
char filePath[256] = "D:\\Dataset\\rgbd_dataset_freiburg3_structure_texture_far\\" ;
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
Mat depthImage[maxPyramidLevel];
Mat grayImage[maxPyramidLevel];
Mat gradientMapForDebug;
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

bool keyframeToFrameDenseTracking( Matrix3d& R_k_c, Vector3d& T_k_c)
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

Mat src, dst;
int spatialRad = 10, colorRad = 10, maxPryLevel = 1;
//const Scalar& colorDiff=Scalar::all(1);

void meanshift_seg()
{
	////调用meanshift图像金字塔进行分割
	double t = (double)cvGetTickCount();
	pyrMeanShiftFiltering(src, dst, spatialRad, colorRad, maxPryLevel);
	t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
	printf("cal time: %f\n", t);

	RNG rng = theRNG();
	Mat mask(dst.rows + 2, dst.cols + 2, CV_8UC1, Scalar::all(0));
	for (int i = 0; i<dst.rows; i++)    //opencv图像等矩阵也是基于0索引
	for (int j = 0; j<dst.cols; j++)
	if (mask.at<uchar>(i + 1, j + 1) == 0)
	{
		Scalar newcolor(rng(256), rng(256), rng(256));
		floodFill(dst, mask, Point(j, i), newcolor, 0, Scalar::all(1), Scalar::all(1));
	}
	imshow("dst", dst);
}

void meanshift_seg_s(int i, void *)
{
	spatialRad = i;
	meanshift_seg();
}

void meanshift_seg_c(int i, void *)
{
	colorRad = i;
	meanshift_seg();
}

void meanshift_seg_m(int i, void *)
{
	maxPryLevel = i;
	meanshift_seg();
}

void testFun()
{
	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("dst", WINDOW_AUTOSIZE);

	src = imread("D:\\depth image\\DS.png");
	CV_Assert(!src.empty());

	spatialRad = 10;
	colorRad = 10;
	maxPryLevel = 1;

	//虽然createTrackbar函数的参数onChange函数要求其2个参数形式为onChange(int,void*)
	//但是这里是系统响应函数，在使用createTrackbar函数时，其调用的函数可以不用写参数，甚至
	//括号都不用写，但是其调用函数的实现过程中还是需要满足(int,void*)2个参数类型
	createTrackbar("spatialRad", "dst", &spatialRad, 80, meanshift_seg_s);
	createTrackbar("colorRad", "dst", &colorRad, 60, meanshift_seg_c);
	createTrackbar("maxPryLevel", "dst", &maxPryLevel, 5, meanshift_seg_m);

	//    meanshift_seg(0,0);

	imshow("src", src);
	/*char c=(char)waitKey();
	if(27==c)
	return 0;*/
	imshow("dst", src);
	imshow("flood", src);
	waitKey();
}

kMeansClusteringDepthSpace clusteringDepth;
bool mask[320*240];
float depthMap[320 * 240];
int K = 50;
int iterNum = 2;
Mat out;
RNG rng(100);
vector<uchar>R;
vector<uchar>G;
vector<uchar>B;
Vector<LOCAL_SURFACE>localSurface;
bool vst[240][320];
//vector<MatrixXd>A;
//vector<VectorXd>b;
//MatrixXd coefficients;
vector<int>cntPoints;
Mat intepolationImage;
Mat intepolationWithConstrain;
Mat depthGradientMap;
int priorWeight = 10 ;

void fitting(int, void *)
{
	if (K == 0){
		return;
	}

	double t = (double)cvGetTickCount();

	int rows = 240;
	int cols = 320;

	
	clusteringDepth.runClustering(K, iterNum);

	int currentNumOfSuperpixel = clusteringDepth.seedsNum;
	printf("currentNumOfSuperpixel = %d\n", currentNumOfSuperpixel);

	R.clear();
	G.clear();
	B.clear();
	for (int i = 0; i < currentNumOfSuperpixel; i++){
		R.push_back( rng.uniform(0, 255) );
		G.push_back( rng.uniform(0, 255) );
		B.push_back( rng.uniform(0, 255) );
	}

	//init matrix size
	cntPoints.clear();
	cntPoints.resize(currentNumOfSuperpixel);
	localSurface.clear();
	localSurface.resize( currentNumOfSuperpixel ) ;
	for (int i = 0; i < currentNumOfSuperpixel; i++)
	{
		localSurface[i].neigbourList.clear();
		localSurface[i].listOfU.clear();
		localSurface[i].listOfU.reserve( clusteringDepth.numOfEachLabels[i + 1] );
		localSurface[i].listOfV.clear();
		localSurface[i].listOfV.reserve( clusteringDepth.numOfEachLabels[i + 1] );
		localSurface[i].b.resize( clusteringDepth.numOfEachLabels[i + 1] );
		localSurface[i].num = clusteringDepth.numOfEachLabels[i + 1];
		cntPoints[i] =  0 ;
	}

	//cntPoints.assign(cntPoints.size(), 0);
	cvtColor(src, out, CV_GRAY2RGB);
	double prop = 0.7;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (clusteringDepth.labels[i][j] < 1){
				continue;
			}
			int id = clusteringDepth.labels[i][j] - 1;

			out.at<cv::Vec3b>(i, j)[0] = out.at<cv::Vec3b>(i, j)[0] * prop + R[id] * (1 - prop);
			out.at<cv::Vec3b>(i, j)[1] = out.at<cv::Vec3b>(i, j)[1] * prop + G[id] * (1 - prop);
			out.at<cv::Vec3b>(i, j)[2] = out.at<cv::Vec3b>(i, j)[2] * prop + B[id] * (1 - prop);

			localSurface[id].listOfU.push_back(i);
			localSurface[id].listOfV.push_back(j);
			localSurface[id].b(cntPoints[id]) = depthMap[INDEX(i, j, rows, cols)];
			cntPoints[id]++;
		}
	}

	memset(vst, false, sizeof(vst));
	int stackTop;
	vector<int>stack(rows*cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (clusteringDepth.labels[i][j] < 1){
				continue;
			}
			if (vst[i][j] == true){
				continue;
			}
			int id = clusteringDepth.labels[i][j] - 1;
			stackTop = 0;
			stack[stackTop++] = (i << 16) | j;
			while (stackTop > 0)
			{
				stackTop--;
				int currentY = stack[stackTop] >> 16;
				int currentX = stack[stackTop] & 0xffff;
				if (vst[currentY][currentX] == true){
					continue;
				}
				vst[currentY][currentX] = true;
				for (int k = 0; k < 8; k++)
				{
					int ty = currentY + dy8[k];
					int tx = currentX + dx8[k];
					if (ty < 0 || ty >= rows || tx < 0 || tx >= cols){
						continue;
					}
					if (vst[ty][tx] == true){
						continue;
					}
					if (clusteringDepth.labels[ty][tx] < 1){
						continue;
					}
					int tmpID = clusteringDepth.labels[ty][tx] - 1;
					if (tmpID == id){
						stack[stackTop++] = (ty << 16) | tx;
					}
					else {
						NEIGBOURHOOD tt;
						tt.u = ty;
						tt.v = tx;
						tt.linkID = tmpID;
						localSurface[id].neigbourList.push_back(tt);
					}
				}
			}
		}
	}

	Mat debugMap;
	cvtColor(src, debugMap, CV_GRAY2BGR );
	for (int id = 0; id < currentNumOfSuperpixel; id++)
	{
		int numOfNeigbourList = localSurface[id].neigbourList.size();
		for (int i = 0; i < numOfNeigbourList; i++)
		{
			int u = localSurface[id].neigbourList[i].u;
			int v = localSurface[id].neigbourList[i].v;
			
			debugMap.at<cv::Vec3b>(u, v)(0) = 0;
			debugMap.at<cv::Vec3b>(u, v)(1) = 255;
			debugMap.at<cv::Vec3b>(u, v)(2) = 0;
		}
	}

	imshow("debugMap", debugMap);

	//with constrain
	intepolationWithConstrain = src.clone();
	MatrixXd constrainATA = MatrixXd::Zero(currentNumOfSuperpixel * 3, currentNumOfSuperpixel*3 );
	VectorXd constrainB = VectorXd::Zero(currentNumOfSuperpixel * 3);
	vector<int> idList;

	//fitting the surface
	MatrixXd AT;
	MatrixXd ATA;
	LLT<MatrixXd> lltATA;
	ComputationInfo info;
	for (int id = 0; id < currentNumOfSuperpixel; id++)
	{
		int numOfPixels = localSurface[id].num;
		localSurface[id].A.resize(numOfPixels, MODEL_ORDER);
		for (int i = 0; i < numOfPixels; i++)
		{
			int y = localSurface[id].listOfU[i];
			int x = localSurface[id].listOfV[i];
			localSurface[id].A(i, 0) = 1.0 ;
			localSurface[id].A(i, 1) = x;
			localSurface[id].A(i, 2) = y;
			if (MODEL_ORDER > 3){
				localSurface[id].A(i, 3) = y*x;
			}
		}
		AT = localSurface[id].A.transpose() ;
		ATA = AT*localSurface[id].A;
		lltATA = ATA.llt();
		info = lltATA.info();
		if (info == Success){
			localSurface[id].coefficients = lltATA.solve(AT* localSurface[id].b);
		}
		else{
			localSurface[id].coefficients = VectorXd::Zero(MODEL_ORDER);
			puts("fitting fail!");
		}

		//construct constrain ATA
		constrainATA.block(id * MODEL_ORDER, id * MODEL_ORDER, MODEL_ORDER, MODEL_ORDER) += ATA;
		constrainB.segment(id * MODEL_ORDER, MODEL_ORDER) += AT* localSurface[id].b;
		int numOfNeigbourList = localSurface[id].neigbourList.size();
		//idList.clear();
		//idList.push_back(id);
		for (int i = 0; i < numOfNeigbourList; i++)
		{
			int u = localSurface[id].neigbourList[i].u;
			int v = localSurface[id].neigbourList[i].v;
			int linkID = localSurface[id].neigbourList[i].linkID;
			if ( depthGradientMap.at<float>(u, v) < 5)
			{
				if (MODEL_ORDER == 3)
				{
					constrainATA(id * MODEL_ORDER + 1, id * MODEL_ORDER + 1) += priorWeight;
					constrainATA(id * MODEL_ORDER + 1, linkID * MODEL_ORDER + 1) -= priorWeight;
					constrainATA(linkID * MODEL_ORDER + 1, id * MODEL_ORDER + 1) -= priorWeight;
					constrainATA(linkID * MODEL_ORDER + 1, linkID * MODEL_ORDER + 1) += priorWeight;

					constrainATA(id * MODEL_ORDER + 2, id * MODEL_ORDER + 2) += priorWeight;
					constrainATA(id * MODEL_ORDER + 2, linkID * MODEL_ORDER + 2) -= priorWeight;
					constrainATA(linkID * MODEL_ORDER + 2, id *MODEL_ORDER + 2) -= priorWeight;
					constrainATA(linkID * MODEL_ORDER + 2, linkID * MODEL_ORDER + 2) += priorWeight;
				}
				else if (MODEL_ORDER == 4)
				{

				}
			}
		}
	}

	LLT<MatrixXd> lltConstrainATA = constrainATA.llt();
	if (lltConstrainATA.info() != Success){
		puts("fail");
	}
	VectorXd coefficients = lltConstrainATA.solve(constrainB);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (clusteringDepth.labels[i][j] < 1){
				continue;
			}
			int id = clusteringDepth.labels[i][j] - 1;

			VectorXd tt(MODEL_ORDER);
			tt(0) = 1.0;
			tt(1) = j;
			tt(2) = i;
			if (MODEL_ORDER > 3){
				tt(3) = j*i;
			}
			double d = tt.transpose() * localSurface[id].coefficients;
			intepolationImage.at<uchar>(i, j) = (uchar)d;

			double d2 = tt.transpose() *coefficients.segment(id * MODEL_ORDER, MODEL_ORDER);
			intepolationWithConstrain.at<uchar>(i, j) = (uchar)d2;
		}
	}
	imshow("intepolationImage", intepolationImage);
	imshow("intepolationWithConstrain", intepolationWithConstrain);

	

	//display centers
	for (int i = 0; i < currentNumOfSuperpixel; i++)
	{
		int y = clusteringDepth.kSeedY[i];
		int x = clusteringDepth.kSeedX[i];
		circle(out, Point(x, y), 2, Scalar(0, 0, 0));
	}

	imshow("out", out);

	t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
	printf("cal time: %f\n", t);
}

void testFun2()
{
	namedWindow("changing_K", WINDOW_AUTOSIZE);
	
	src = imread("D:\\depth image\\rpy_DS.png");
	out = src.clone();

	cvtColor(src, src, CV_RGB2GRAY);
	intepolationImage = src.clone();

	imshow("src", src);

	Scharr(src, depthGradientMap, CV_32F, 1, 0);
	depthGradientMap /= 32;

	//int y = 51;
	//int x = 51;
	//for (int i = -1; i <= 1; i++)
	//{
	//	for (int j = -1; j <= 1; j++)
	//	{
	//		int ty = y + i;
	//		int tx = x + j;
	//		printf("%d ", src.at<uchar>(ty, tx) );
	//	}
	//	cout << endl;
	//}
	//cout << depthGradientMap.at<float>(y, x);

	Mat src2;

	src.convertTo(src2, CV_32F);
	//src /= 255.0;

	memset(mask, true, sizeof(mask));
	int rows = 240;
	int cols = 320;

	//printf("%d %d\n", src.rows, src.cols);

	clusteringDepth.setCameraParameters(rows, cols, 0, 0, 0, 0);
	clusteringDepth.setMask(&mask[0]);
	memcpy(depthMap, (float*)src2.data, rows*cols*sizeof(float));
	clusteringDepth.setInputData(depthMap);

	createTrackbar("ClusteringNum", "changing_K", &K, 500, fitting );
	createTrackbar("IterationNum", "changing_K", &iterNum, 20, fitting);
	createTrackbar("priorWeight", "changing_K", &priorWeight, 100, fitting);

	imshow("out", out);
	imshow("src", src);
	imshow("intepolationImage", intepolationImage);

	cv::waitKey(0);

	return;
}


int main()
{
	testFun2();
	return 0;
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
	bool insertKeyFrameFlag = true;
	map<unsigned long long, int>::iterator iter;

	Matrix3d R_k_c;//R_k^(k+1)
	Matrix3d R_c_0;
	Vector3d T_k_c;//T_k^(k+1)
	Vector3d T_c_0;

	ofstream fileOutput("result.txt");
	
	init();
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
			slidingWindows.insertKeyFrame(grayImage, depthImage, gradientMapForDebug, Matrix3d::Identity(), Vector3d::Zero());
			slidingWindows.planeDection();

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
		insertKeyFrameFlag = keyframeToFrameDenseTracking(R_k_c, T_k_c);
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
		if ( (i%10) == 1 )
		{
			

			slidingWindows.insertKeyFrame(grayImage, depthImage, gradientMapForDebug, R_c_0, T_c_0 );

			//imshow("gradientImage", gradientMapForDebug);
			//cvMoveWindow("gradientImage", 500, 0);

			cout << "estimate position[before BA]:\n" 
				<< slidingWindows.states[slidingWindows.tail].T_k0.transpose() << endl;

			double t = (double)cvGetTickCount();

			slidingWindows.PhotometricBA();

			t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
			printf("BA cal time: %f\n", t);

			slidingWindows.planeDection();

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


		//imshow("currentImage", grayImage[0]);
		//cvMoveWindow("currentImage", 0, 0);
		//waitKey(1);
	}
	fileOutput.close();

	return 0;
}
