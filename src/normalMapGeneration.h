#ifndef __NORMALMAPGENERATION_H
#define __NORMALMAPGENERATION_H
#include "variableDefinition.h"
#include "dataStructure.h"
#include "utility.h"
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace cv;

//#define PRINTF_TIME 

class normalMapGerneration
{
public:
	double fx, fy, cx, cy;
	int n, m;
	float* depth;
	bool* mask;
	double integralDepth[IMAGE_HEIGHT][IMAGE_WIDTH];
	double smoothDepth[IMAGE_HEIGHT][IMAGE_WIDTH];
	int integralNum[IMAGE_HEIGHT][IMAGE_WIDTH];
	bool valid[IMAGE_HEIGHT][IMAGE_WIDTH];
	double distanceMap[IMAGE_HEIGHT][IMAGE_WIDTH];
	double normalMap[IMAGE_HEIGHT][IMAGE_WIDTH][3];

	normalMapGerneration()
	{
		depth = NULL;
		mask = NULL;
	}
	~normalMapGerneration()
	{

	}

	void setCameraParameters(int input_n, int input_m,
		double input_fx, double input_fy, double input_cx, double input_cy){
		n = input_n;
		m = input_m;
		fx = input_fx;
		fy = input_fy;
		cx = input_cx;
		cy = input_cy;
	}

	void setInputData(float* depthMap){
		depth = depthMap;
	}

	void setMask(bool* maskMap){
		mask = maskMap;
	}

	inline void computeAverageDepth(int yLeftTop, int yRightDown, int xLeftTop, int xRightDown, double& sumDepth, int& sumNum)
	{
		//sum depth
		sumDepth = integralDepth[yRightDown][xRightDown];
		sumNum = integralNum[yRightDown][xRightDown];
		if (xLeftTop > 0){
			sumDepth -= integralDepth[yRightDown][xLeftTop - 1];
			sumNum -= integralNum[yRightDown][xLeftTop - 1];
		}
		if (yLeftTop > 0){
			sumDepth -= integralDepth[yLeftTop - 1][xRightDown];
			sumNum -= integralNum[yLeftTop - 1][xRightDown];
		}
		if (xLeftTop > 0 && yLeftTop > 0){
			sumDepth += integralDepth[yLeftTop - 1][xLeftTop - 1];
			sumNum += integralNum[yLeftTop - 1][xLeftTop - 1];
		}
	}

	inline void crossProduct(double u[3], double v[3], double output[3])
	{
		output[0] = u[1] * v[2] - u[2] * v[1];
		output[1] = u[2] * v[0] - u[0] * v[2];
		output[2] = u[0] * v[1] - u[1] * v[0];

		//normalized
		double norm = output[0] * output[0] + output[1] * output[1] + output[2] * output[2];
		norm = sqrt(norm);

		output[0] /= norm;
		output[1] /= norm;
		output[2] /= norm;
	}

	void computeIntegral( int height, int width)
	{
		int n = height;
		int m = width;

		//calculate the integral image
		integralDepth[0][0] = (double)depth[0] / depthFactor;
		integralNum[0][0] = (depth[0] > zeroThreshold);
		valid[0][0] = (depth[0] > zeroThreshold);
		for (int j = 1; j < m; j++){
			integralDepth[0][j] = depth[j] / depthFactor + integralDepth[0][j - 1];
			integralNum[0][j] = (depth[j] > zeroThreshold) + integralNum[0][j - 1];
			valid[0][j] = (depth[j] > zeroThreshold);
		}
		for (int i = 1; i < n; i++){
			integralDepth[i][0] = depth[i*width] / depthFactor + integralDepth[i - 1][0];
			integralNum[i][0] = (depth[i*width] > zeroThreshold) + integralNum[i - 1][0];
			valid[i][0] = (depth[i*width] > zeroThreshold);
		}
		for (int i = 1; i < n; i++)
		{
			for (int j = 1; j < m; j++)
			{
				integralDepth[i][j] = integralDepth[i - 1][j] + integralDepth[i][j - 1] - integralDepth[i - 1][j - 1] + depth[i*width + j] / depthFactor;
				integralNum[i][j] = integralNum[i - 1][j] + integralNum[i][j - 1] - integralNum[i - 1][j - 1] + (depth[i*width + j] > zeroThreshold);
				valid[i][j] = (depth[i*width + j] > zeroThreshold);
			}
		}
	}

	void initDistanceMap(int height, int width)
	{
		int n = height;
		int m = width;

		for (int i = 0; i < n; i++){
			distanceMap[i][0] = distanceMap[i][m - 1] = 0;
			valid[i][0] = valid[i][m - 1] = false;
		}
		for (int j = 0; j < m; j++){
			distanceMap[0][j] = distanceMap[n - 1][j] = 0;
			valid[0][j] = valid[n - 1][j] = false;
		}
		int nn = n - 1;
		int mm = m - 1;
		for (int i = 0; i < nn; i++)
		{
			for (int j = 0; j < mm; j++)
			{
				distanceMap[i][j] = height + width;
				double f_DC = 0.0028 * (double)depth[i*width + j] * (double)depth[i*width + j] / 25000000.0;
				double t_DC = f_DC * Gamma;
				int current = i*width + j;
				int right = i*width + j + 1;
				int down = (i + 1)*width + j;
				if (valid[i][j] == false || valid[i][j + 1] == false || fabs( depth[right] - depth[current] ) / 5000.0 > t_DC){
					distanceMap[i][j] = 0;
					distanceMap[i][j + 1] = 0;
				}
				if (valid[i][j] == false || valid[i + 1][j] == false || fabs( depth[down] - depth[current] ) / 5000.0 > t_DC){
					distanceMap[i][j] = 0;
					distanceMap[i + 1][j] = 0;
				}
			}
		}
	}

	void distanceTransform( int height, int width)
	{
		int n = height;
		int m = width;
		int nn = n - 1;
		int mm = m - 1;

		double maxValue = 0;

		//distance tranform
		for (int i = 1; i < nn; i++)
		{
			for (int j = 1; j < mm; j++)
			{
				if (valid[i][j] == false){
					continue;
				}
				double upLeft = distanceMap[i - 1][j - 1] + 1.4f;
				double up = distanceMap[i - 1][j] + 1.0f;
				double upRight = distanceMap[i - 1][j + 1] + 1.4f;
				double left = distanceMap[i][j - 1] + 1.0f;
				double minValue = std::min(std::min(upLeft, up), std::min(left, upRight));
				if (minValue < distanceMap[i][j]){
					distanceMap[i][j] = minValue;
				}
			}
		}
		for (int i = nn - 1; i > 0; i--)
		{
			for (int j = mm - 1; j > 0; j--)
			{
				if (valid[i][j] == false){
					continue;
				}
				double lowerRight = distanceMap[i + 1][j + 1] + 1.4f;
				double lower = distanceMap[i + 1][j] + 1.0f;
				double lowerLeft = distanceMap[i + 1][j - 1] + 1.4f;
				double right = distanceMap[i][j + 1] + 1.0f;
				double minValue = std::min(std::min(lowerRight, lower), std::min(lowerLeft, right));
				if (minValue < distanceMap[i][j]){
					distanceMap[i][j] = minValue;
				}
			}
		}
	}

	void computeNormalByIntegral()
	{
		//1. caculate integral image
#ifdef PRINTF_TIME
		double t1 = (double)cvGetTickCount();
#endif

		computeIntegral(n, m);

#ifdef PRINTF_TIME
		printf("t1= %f\n", ((double)cvGetTickCount() - t1) / (cvGetTickFrequency() * 1000));
#endif


		//2. init distance map
#ifdef PRINTF_TIME
		double t2 = (double)cvGetTickCount();
#endif

		initDistanceMap(n, m);

#ifdef PRINTF_TIME
		printf("t2= %f\n", ((double)cvGetTickCount() - t2) / (cvGetTickFrequency() * 1000));
#endif

		/*Mat display(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
		for (int j = 0; j < width; j++)
		{
		display.at<uchar>(i, j) = (uchar)(distanceMap[i][j] * 255 / (height + width) );
		}
		}
		imshow("distance map - original", display);
		*/


		//3. distance transform
#ifdef PRINTF_TIME
		double t3 = (double)cvGetTickCount();
#endif

		distanceTransform(n, m);

#ifdef PRINTF_TIME
		printf("t3= %f\n", ((double)cvGetTickCount() - t3) / (cvGetTickFrequency() * 1000));
#endif


		//4. calculate smooth depth
#ifdef PRINTF_TIME
		double t4 = (double)cvGetTickCount();
#endif

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				valid[i][j] &= mask[INDEX(i, j, n, m)] ;
				if (valid[i][j] == false){
					smoothDepth[i][j] = 0;
					continue;
				}
				int d = int(distanceMap[i][j] / SQR2);
				if (d > 0){
					int xLeftTop = j - d;
					int yLeftTop = i - d;
					int xRightDown = j + d;
					int yRightDown = i + d;
					double sumDepth;
					int sumNum;
					computeAverageDepth(yLeftTop, yRightDown, xLeftTop, xRightDown, sumDepth, sumNum);
					smoothDepth[i][j] = sumDepth / sumNum;
				}
				else{
					smoothDepth[i][j] = (double)depth[INDEX(i, j, n, m)] / depthFactor;
					valid[i][j] = false;
				}
			}
		}
#ifdef PRINTF_TIME
		printf("t4= %f\n", ((double)cvGetTickCount() - t4) / (cvGetTickFrequency() * 1000));
#endif

		//6. calculate normal map
#ifdef PRINTF_TIME
		double t5 = (double)cvGetTickCount();
#endif

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				if (valid[i][j] == false){
					continue;
				}
				double v_x[3];
				double v_y[3];
				int d = int(distanceMap[i][j] / SQR2);
				int left = j - d;
				int right = j + d;
				int up = i - d;
				int down = i + d;

				v_x[0] = (right - cx)*smoothDepth[i][right] / fx - (left - cx)*smoothDepth[i][left] / fx;
				v_x[1] = (i - cy)*smoothDepth[i][right] / fy - (i - cy)*smoothDepth[i][left] / fy;
				v_x[2] = smoothDepth[i][right] - smoothDepth[i][left];

				v_y[0] = (j - cx)*smoothDepth[down][j] / fx - (j - cx)*smoothDepth[up][j] / fx;
				v_y[1] = (down - cy)*smoothDepth[down][j] / fy - (up - cy)*smoothDepth[up][j] / fy;
				v_y[2] = smoothDepth[down][j] - smoothDepth[up][j];

				crossProduct(v_x, v_y, normalMap[i][j]);
			}
		}

#ifdef PRINTF_TIME
		printf("t5= %f\n", ((double)cvGetTickCount() - t5) / (cvGetTickFrequency() * 1000));
#endif

	}
};

#endif