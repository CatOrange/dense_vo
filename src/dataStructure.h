#ifndef __DATASTRUCTURE_H
#define __DATASTRUCTURE_H

#include "variableDefinition.h"
#include "utility.h"
#include "Eigen/Dense"
#include <iostream>
#include <list>
#include <vector>
using namespace Eigen;

/*

for AHC clustering

*/
struct NODE{
	double sumX, sumY, sumZ, sumX2, sumY2, sumZ2, sumXY, sumXZ, sumYZ;
	double normal[3];
	double MSE;
	int id, num;
	bool valid;

	bool operator < (const NODE &  a)const{
		return MSE < a.MSE;
	}
};

struct HEAPNODE{
	double MSE;
	int id;

	bool operator < (const HEAPNODE &  a)const{
		return MSE < a.MSE;
	}
};

struct GRADIENTNODE{
	unsigned short u, v;
	double cost;
	bool operator < (const GRADIENTNODE &a)const{
		return cost > a.cost;
	}
};

class minHeap
{
public:
	int size;
	HEAPNODE data[queueSize];

	minHeap(){
		size = 0;
	}
	~minHeap(){
	}

	void clear(){
		size = 0;
	}

	void down(int x)
	{
		HEAPNODE tmp = data[x];
		for (int left, right;;)
		{
			left = x << 1, right = left + 1;
			if (right <= size && data[right] < data[left] && data[right] < tmp){
				data[x] = data[right];
				x = right;
			}
			else if (left <= size && data[left] < tmp){
				data[x] = data[left];
				x = left;
			}
			else break;
		}
		data[x] = tmp;
	}

	void push(HEAPNODE x)
	{
		data[++size] = x;
		int i, f;
		for (i = size; i > 1;)
		{
			f = i >> 1;
			if (x < data[f]) {
				data[i] = data[f];
				i = f;
			}
			else break;
		}
		data[i] = x;
	}

	void pop()
	{
		if (size > 1){
			data[1] = data[size];
			down(1);
		}
		size--;
	}
};

class unionSet
{
	int parent[unisetSize];

public:
	unionSet(){
		;
	}
	~unionSet(){
		;
	}

	void unionSetInit(int n)
	{
		for (int i = 0; i < n; i++){
			parent[i] = i;
		}
	}

	int find(int x)
	{
		int k, j, r;
		r = x;
		while (r != parent[r]){
			r = parent[r];
		}
		k = x;
		while (k != r){
			j = parent[k];
			parent[k] = r;
			k = j;
		}
		return r;
	}

	void merge(int x, int y)
	{
		int rx = find(x), ry = find(y);
		parent[ry] = rx;
	}
};

struct CAMER_PARAMETERS
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Matrix3d Ric;
	Eigen::Vector3d Tic;
	cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64FC1);
	double fy[maxPyramidLevel], fx[maxPyramidLevel], cy[maxPyramidLevel], cx[maxPyramidLevel];

	//CAMER_PARAMETERS(double input_fx, double input_fy, double input_cx, double input_cy )
	//{
	//	fx[0] = input_fx;
	//	fy[0] = input_fy;
	//	cx[0] = input_cx;
	//	cy[0] = input_cy;

	//#ifdef DOWNSAMPLING
	//		fx[0] /= 2.0;
	//		fy[0] /= 2.0;
	//		cx[0] = (cx[0] + 0.5) / 2.0 - 0.5;
	//		cy[0] = (cy[0] + 0.5) / 2.0 - 0.5;
	//#endif

	//initPyramidParameters();

	//	cameraMatrix.at<double>(0, 0) = input_fx;
	//	cameraMatrix.at<double>(0, 2) = input_cx;
	//	cameraMatrix.at<double>(1, 1) = input_fy;
	//	cameraMatrix.at<double>(1, 2) = input_cy;
	//	cameraMatrix.at<double>(2, 2) = 1.0;
	//}
	CAMER_PARAMETERS(){
	}

	void setExtrinsics(const Eigen::Matrix3d& R, const Eigen::Vector3d& T)
	{
		Ric = R;
		Tic = T;
	}

	void setParameters(double input_fx, double input_fy, double input_cx, double input_cy)
	{
		fx[0] = input_fx;
		fy[0] = input_fy;
		cx[0] = input_cx;
		cy[0] = input_cy;

		initPyramidParameters();

		cameraMatrix.at<double>(0, 0) = input_fx;
		cameraMatrix.at<double>(0, 2) = input_cx;
		cameraMatrix.at<double>(1, 1) = input_fy;
		cameraMatrix.at<double>(1, 2) = input_cy;
		cameraMatrix.at<double>(2, 2) = 1.0;
	}

	void setDistortionCoff(double d0, double d1, double d2, double d3, double d4)
	{
		distCoeffs.at<double>(0, 0) = d0;
		distCoeffs.at<double>(0, 1) = d1;
		distCoeffs.at<double>(0, 2) = d2;
		distCoeffs.at<double>(0, 3) = d3;
		distCoeffs.at<double>(0, 4) = d4;
	}

	void initPyramidParameters()
	{
		for (int i = 1; i < maxPyramidLevel; i++)
		{
			fx[i] = fx[i - 1] / 2.0;
			fy[i] = fy[i - 1] / 2.0;
			cx[i] = (cx[i - 1] + 0.5) / 2.0 - 0.5;
			cy[i] = (cy[i - 1] + 0.5) / 2.0 - 0.5;
		}
	}
};

#endif